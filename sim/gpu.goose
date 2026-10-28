// goosestrike on the GPU: the host of the WebGPU solver.
//
// The set-up is that of strike.goose: setup() builds the panel and the bird
// in its f64 arrays. This file copies them to GPU buffers in f32, makes a
// bind group for each kernel of web/strike.wgsl, and records the steps. One
// explicit time step is ten kernels in sequence (step_kernels). The WebGPU
// calls go through the C of sim/gpu.c, since Goose cannot build the
// webgpu.h descriptors or take a callback.
//
// The GPU keeps the whole state. gpu_run records some steps and a copy of
// the results to a read-back buffer. When the page sees that the buffer is
// mapped (gpu_poll), the results go into the globals of strike.goose, so
// its getters (get_time, buf_nodes and the others) give the GPU results.
// There are two read-back buffers, so the GPU can do the next run while
// the page waits for the results of the last one.

import strike;

extern fn gw_init(wgsl: const u8[:]) -> i32;
extern fn gw_pipeline(entry: const u8[:]) -> i32;
extern fn gw_reset();
extern fn gw_buffer(size: i64, kind: i32) -> i32;
extern fn gw_write(buf: i32, offset: i64, data: const u8[:]);
extern fn gw_bind(pipe: i32, pairs: const i32[:]) -> i32;
extern fn gw_dispatch(pipe: i32, group: i32, n: i64);
extern fn gw_copy(src: i32, src_off: i64, dst: i32, dst_off: i64, size: i64);
extern fn gw_submit();
extern fn gw_map(buf: i32);
extern fn gw_mapped(buf: i32) -> i32;
extern fn gw_read(buf: i32, offset: i64, out: f32[:]);
extern fn gw_unmap(buf: i32);

// ---- buffers: binding n of web/strike.wgsl is buffer n ----
let B_U = 0;        // parameters (uniform)
let B_S = 1;        // the scalar state: time, dt, sums
let B_PART = 2;     // particles: (x, rho), (v, p)
let B_PACC = 3;
let B_PFC = 4;
let B_PBIN = 5;
let B_PHEAD = 6;
let B_NINFO = 7;    // nodes: (x0, rotary inertia), (fixed, skin)
let B_NX = 8;       // nodes: (x, mass)
let B_NVW = 9;
let B_NFC = 10;
let B_ADJ = 11;     // the element corners of each node
let B_EINFO = 12;   // elements: nodes, (thickness, material)
let B_ESTATE = 13;
let B_EFLAG = 14;
let B_EFOR = 15;
let B_EBIN = 16;
let B_EHEAD = 17;
let B_RED = 18;     // partial sums, 2 per workgroup
let B_HIST = 19;
let B_RBUF = 20;    // what the page draws
let B_READ = 21;    // the read-back copies: B_READ and B_READ + 1
let NB = 23;

let WG = 64;            // workgroup size of the kernels
let NU = 52;            // floats in the parameters
let NS = 24;            // floats in the state
let HIST_CAP = 16384;   // steps that the history keeps
let RUN_MAX = 512;      // steps per gpu_run

// ---- kernels ----
let KN_CLEAR = 0;
let KN_BIN_P = 1;
let KN_SPH = 2;
let KN_SHELL = 3;
let KN_BIN_E = 4;
let KN_CONTACT = 5;
let KN_INT_P = 6;
let KN_INT_N = 7;
let KN_GEOM = 8;
let KN_REDUCE = 9;
let KN_RENDER = 10;
let NK = 11;

var gbuf: i32[NB];          // the handle of each buffer in sim/gpu.c
var kpipe: i32[NK];
var kbind: i32[NK];
var gu: f32[NU];            // parameters
var gs: f32[NS];            // the state, as read back
var ghist: f32[4 * HIST_CAP];   // per step: time, contact force, sensor force, max deflection
var up: f32[MAXP * 8];      // upload staging, f32
var upi: i32[MAXN + 1 + 4 * MAXE];
var cur: i32[MAXN];
var submitted: i64 = 0;     // steps recorded since the set-up
var oldest: i64 = 0;        // the read-back copy that gpu_poll reads next
var pending: i64 = 0;       // runs not read back (0, 1 or 2)
var read_first: i64[2];     // the history entries in each read-back copy
var read_count: i64[2];

fn groups(n: i64) -> i64 { (n + WG - 1) / WG }

// Compiles the kernels. wgsl: the text of web/strike.wgsl.
export fn gpu_init(wgsl: const u8[:]) -> i32 {
    if gw_init(wgsl) == 0 { return 0; }
    kpipe[KN_CLEAR] = gw_pipeline("clear");
    kpipe[KN_BIN_P] = gw_pipeline("bin_particles");
    kpipe[KN_SPH] = gw_pipeline("sph");
    kpipe[KN_SHELL] = gw_pipeline("shell");
    kpipe[KN_BIN_E] = gw_pipeline("bin_elements");
    kpipe[KN_CONTACT] = gw_pipeline("contact");
    kpipe[KN_INT_P] = gw_pipeline("integrate_particles");
    kpipe[KN_INT_N] = gw_pipeline("integrate_nodes");
    kpipe[KN_GEOM] = gw_pipeline("elem_geom");
    kpipe[KN_REDUCE] = gw_pipeline("reduce");
    kpipe[KN_RENDER] = gw_pipeline("render");
    return 1;
}

fn make(b: i64, bytes: i64, kind: i32) { gbuf[b] = gw_buffer(bytes, kind); }

fn upload(b: i64, floats: i64) { gw_write(gbuf[b], 0, bytes_of(up[0..floats])); }

// A bind group with the buffers that kernel k uses (the bindings it
// declares in web/strike.wgsl).
fn bind(k: i64, bufs: const i64[:]) {
    var pairs: i32[32];
    for i in 0..bufs.len {
        pairs[2 * i] = bufs[i] as i32;
        pairs[2 * i + 1] = gbuf[bufs[i]];
    }
    kbind[k] = gw_bind(kpipe[k], pairs[0..2 * bufs.len]);
}

fn rbuf_floats() -> i64 { 4 * np + 3 * nn + ne }

// Sets up a run: the CPU set-up of strike.goose (set_skin, then setup), then
// the state in GPU buffers, and the time step before the first step.
// Returns the particle count, as setup. The render data and the energies are
// those of the set-up until the first gpu_poll.
export fn gpu_setup(m: i32, cells: i32, spacing: f64, speed: f64) -> i32 {
    let n = setup(m, cells, spacing, speed);
    gw_reset();
    let gp = groups(np);
    let gn = groups(nn);
    let ge = groups(ne);
    make(B_U, 4 * NU, 1);
    make(B_S, 4 * NS, 0);
    make(B_PART, 32 * np, 0);
    make(B_PACC, 16 * np, 0);
    make(B_PFC, 16 * np, 0);
    make(B_PBIN, 16 * np, 0);
    make(B_PHEAD, 4 * NHASH, 0);
    make(B_NINFO, 32 * nn, 0);
    make(B_NX, 16 * nn, 0);
    make(B_NVW, 32 * nn, 0);
    make(B_NFC, 12 * nn, 0);
    make(B_ADJ, 4 * (nn + 1 + 4 * ne), 0);
    make(B_EINFO, 32 * ne, 0);
    make(B_ESTATE, 128 * ne, 0);
    make(B_EFLAG, 4 * ne, 0);
    make(B_EFOR, 128 * ne, 0);
    make(B_EBIN, 32 * ne, 0);
    make(B_EHEAD, 4 * NHASH, 0);
    make(B_RED, 32 * (gp + gn + ge), 0);
    make(B_HIST, 16 * HIST_CAP, 0);
    make(B_RBUF, 4 * rbuf_floats(), 0);
    for c in 0..2 { make(B_READ + c, 4 * NS + 16 * RUN_MAX + 4 * rbuf_floats(), 2); }

    // The new buffers are zero. Only the non-zero state is uploaded.
    for i in 0..np {
        for k in 0..3 {
            up[8 * i + k] = px[3 * i + k] as f32;
            up[8 * i + 4 + k] = pv[3 * i + k] as f32;
        }
        up[8 * i + 3] = prho[i] as f32;
        up[8 * i + 7] = pp[i] as f32;
    }
    upload(B_PART, 8 * np);
    for i in 0..nn {
        for k in 0..3 { up[8 * i + k] = nx[3 * i + k] as f32; }
        up[8 * i + 3] = ninert[i] as f32;
        up[8 * i + 4] = nfix[i];
        up[8 * i + 5] = nskin[i];
        up[8 * i + 6] = 0.0;
        up[8 * i + 7] = 0.0;
    }
    upload(B_NINFO, 8 * nn);
    for i in 0..nn {
        for k in 0..3 { up[4 * i + k] = nx[3 * i + k] as f32; }
        up[4 * i + 3] = nmass[i] as f32;
    }
    upload(B_NX, 4 * nn);
    for e in 0..ne {
        for k in 0..4 { up[8 * e + k] = en[4 * e + k]; }
        up[8 * e + 4] = eth[e] as f32;
        up[8 * e + 5] = emat[e];
        up[8 * e + 6] = 0.0;
        up[8 * e + 7] = 0.0;
    }
    upload(B_EINFO, 8 * ne);
    // the corners 4 e + k of each node, after the first index of each node
    for i in 0..nn + 1 { upi[i] = 0; }
    for c in 0..4 * ne { upi[en[c] + 1] += 1; }
    for i in 0..nn { upi[i + 1] += upi[i]; }
    for i in 0..nn { cur[i] = upi[i]; }
    for c in 0..4 * ne {
        let i = en[c];
        upi[nn + 1 + cur[i]] = c as i32;
        cur[i] += 1;
    }
    gw_write(gbuf[B_ADJ], 0, bytes_of(upi[0..nn + 1 + 4 * ne]));

    // the parameters: see struct Params in web/strike.wgsl
    gu[0] = mode; gu[1] = np; gu[2] = nn; gu[3] = ne;
    gu[4] = pm as f32; gu[5] = ph as f32; gu[6] = prad as f32; gu[7] = sensor_r as f32;
    gu[8] = 3.0e38;   // t_end: gpu_run sets it
    gu[9] = HIST_CAP; gu[10] = gp; gu[11] = gp + gn;
    for k in 0..2 {
        gu[12 + 4 * k] = mat_e[k] as f32; gu[13 + 4 * k] = mat_nu[k] as f32;
        gu[14 + 4 * k] = mat_sy[k] as f32; gu[15 + 4 * k] = mat_h[k] as f32;
    }
    gu[20] = mat_epf[0] as f32; gu[21] = mat_epf[1] as f32; gu[22] = mat_rho[0] as f32; gu[23] = mat_rho[1] as f32;
    gu[24] = RHO_BIRD; gu[25] = C0_BIRD; gu[26] = GAMMA_TAIT; gu[27] = NHASH;
    gu[28] = SPH_ALPHA; gu[29] = SPH_BETA; gu[30] = K_CONTACT; gu[31] = HG_R;
    gu[32] = KAPPA_S; gu[33] = CFL_SPH; gu[34] = CFL_SHELL; gu[35] = max(t_skin, T_FRM) as f32;
    for g in 0..8 {
        gu[36 + g] = if g < NIP { gp_z[g] as f32 } else { 0.0 };
        gu[44 + g] = if g < NIP { gp_w[g] as f32 } else { 0.0 };
    }
    gw_write(gbuf[B_U], 0, bytes_of(gu));
    // the state: see struct State
    for i in 0..NS { gs[i] = 0.0; }
    gs[5] = -1.0;   // t_contact: no contact yet
    gs[17] = 1.0;   // running
    gs[19] = get_bird_ke() as f32;
    gw_write(gbuf[B_S], 0, bytes_of(gs));

    bind(KN_CLEAR, [B_U, B_PHEAD, B_NFC, B_EHEAD]);
    bind(KN_BIN_P, [B_U, B_S, B_PART, B_PBIN, B_PHEAD]);
    bind(KN_SPH, [B_U, B_S, B_PART, B_PACC, B_PBIN, B_PHEAD]);
    bind(KN_SHELL, [B_U, B_S, B_NX, B_NVW, B_EINFO, B_ESTATE, B_EFLAG, B_EFOR]);
    bind(KN_BIN_E, [B_U, B_S, B_EFLAG, B_EBIN, B_EHEAD]);
    bind(KN_CONTACT, [B_U, B_S, B_PART, B_PFC, B_NX, B_NFC, B_EINFO, B_EBIN, B_EHEAD]);
    bind(KN_INT_P, [B_U, B_S, B_PART, B_PACC, B_PFC, B_RED]);
    bind(KN_INT_N, [B_U, B_S, B_NINFO, B_NX, B_NVW, B_NFC, B_ADJ, B_EFOR, B_RED]);
    bind(KN_GEOM, [B_U, B_NX, B_EINFO, B_ESTATE, B_EFLAG, B_EBIN, B_RED]);
    bind(KN_REDUCE, [B_U, B_S, B_RED, B_HIST]);
    bind(KN_RENDER, [B_U, B_PART, B_NX, B_ESTATE, B_EFLAG, B_RBUF]);

    // before the first step: the element geometry and the first time step
    if mode == 0 { dispatch(KN_GEOM, ge); }
    dispatch(KN_INT_P, gp);
    if mode == 0 { dispatch(KN_INT_N, gn); }
    dispatch(KN_REDUCE, 1);
    gw_submit();
    submitted = 0;
    oldest = 0;
    pending = 0;
    fill_render();
    return n;
}

fn dispatch(k: i64, n: i64) { gw_dispatch(kpipe[k], kbind[k], n); }

// One time step: run_phase 0 to 4 of strike.goose. The time step of
// phase 0 comes from reduce at the end of the step before.
fn step_kernels() {
    let gp = groups(np);
    let ge = groups(ne);
    dispatch(KN_CLEAR, groups(max(NHASH, 3 * nn)));
    dispatch(KN_BIN_P, gp);
    dispatch(KN_SPH, gp);
    if mode == 0 {
        dispatch(KN_SHELL, ge);
        dispatch(KN_BIN_E, ge);
    }
    dispatch(KN_CONTACT, gp);
    dispatch(KN_INT_P, gp);
    if mode == 0 {
        dispatch(KN_INT_N, groups(nn));
        dispatch(KN_GEOM, ge);
    }
    dispatch(KN_REDUCE, 1);
}

// Records `steps` steps (at most RUN_MAX), then a copy of the state, of
// their history and of what the page draws, and maps that copy. The GPU
// does no step at or after t_end (s): give the same t_end to each run of a
// set-up. Returns the steps recorded, or 0 while two runs are not read back.
export fn gpu_run(steps: i32, t_end: f64) -> i32 {
    if pending == 2 { return 0; }
    if gu[8] != t_end as f32 {
        gu[8] = t_end as f32;
        gw_write(gbuf[B_U], 32, bytes_of(gu[8..9]));
    }
    let k = clamp(steps as i64, 0, RUN_MAX);
    for s in 0..k { step_kernels(); }
    dispatch(KN_RENDER, groups(max(max(np, nn), ne)));
    let c = (oldest + pending) % 2;
    let rb = gbuf[B_READ + c];
    gw_copy(gbuf[B_S], 0, rb, 0, 4 * NS);
    read_first[c] = min(submitted, HIST_CAP);
    read_count[c] = min(submitted + k, HIST_CAP) - read_first[c];
    gw_copy(gbuf[B_HIST], 16 * read_first[c], rb, 4 * NS, 16 * read_count[c]);
    gw_copy(gbuf[B_RBUF], 0, rb, 4 * NS + 16 * RUN_MAX, 4 * rbuf_floats());
    gw_submit();
    gw_map(rb);
    submitted += k;
    pending += 1;
    return k as i32;
}

// 1 when the copy of the oldest gpu_run that is not read back is mapped:
// then this puts it in the globals of strike.goose and the history. 0 while
// the GPU works or when no run is pending, -1 if the map failed.
export fn gpu_poll() -> i32 {
    if pending == 0 { return 0; }
    let rb = gbuf[B_READ + oldest];
    let state = gw_mapped(rb);
    if state != 1 { return state; }
    let a = read_first[oldest];
    gw_read(rb, 0, gs[0..NS]);
    gw_read(rb, 4 * NS, ghist[4 * a..4 * (a + read_count[oldest])]);
    let o = 4 * NS + 16 * RUN_MAX;
    gw_read(rb, o, up[0..4 * np]);
    for i in 0..np {
        for k in 0..3 { r_part[3 * i + k] = up[4 * i + k]; }
        r_pp[i] = up[4 * i + 3];
    }
    gw_read(rb, o + 16 * np, r_node[0..3 * nn]);
    gw_read(rb, o + 16 * np + 12 * nn, r_eep[0..ne]);
    gw_unmap(rb);
    oldest = (oldest + 1) % 2;
    pending -= 1;
    sim_time = gs[0];
    dt = gs[1];
    nsteps = gs[4] as i64;
    t_contact = gs[5];
    contact_fz = gs[6];
    center_fz = gs[7];
    impulse = gs[8];
    panel_work = gs[9];
    hg_work = gs[10];
    bird_work = gs[11];
    n_failed = gs[12] as i64;
    n_failed_skin = gs[13] as i64;
    max_ep = gs[14];
    max_defl = gs[15];
    return 1;
}

// 0 when the GPU reached t_end
export fn gpu_running() -> i32 { gs[17] as i32 }
// the runs that gpu_poll did not read back
export fn gpu_pending() -> i32 { pending as i32 }
export fn gpu_bird_ke() -> f64 { gs[19] }
export fn gpu_panel_ke() -> f64 { gs[20] }
export fn gpu_hist_cap() -> i32 { HIST_CAP }
// per step from the first: time, contact force, sensor force, max deflection
export fn gpu_buf_hist() -> const u8[:] { bytes_of(ghist) }
