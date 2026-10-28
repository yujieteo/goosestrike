// goosestrike on the GPU: one explicit time step of sim/strike.goose as
// compute kernels, in f32. sim/gpu.goose makes the buffers, fills them from
// the set-up of strike.goose and dispatches the kernels in this sequence:
//
//   clear, bin_particles, sph, shell, bin_elements, contact,
//   integrate_particles, integrate_nodes, elem_geom, reduce
//
// Each kernel does the work of one function of strike.goose for one
// particle, node or element. Where strike.goose adds a force of an element
// to its nodes, shell writes the force per element corner and
// integrate_nodes gathers them. The contact forces on the nodes are added
// with atomics. The sums and the maxima (the energies, the deflection, the
// next time step) are made per workgroup and then by reduce, in one
// workgroup, which also advances the time. All kernels stop when the time
// reaches t_end, so the host can send more steps than it needs.
//
// Binding n is buffer n of sim/gpu.goose. Some buffers have two or three
// views (plain, atomic, read-only). One kernel uses one view of each.

struct Params {
    n: vec4f,      // mode (0: panel, 1: rigid wall), particles, nodes, elements
    sph: vec4f,    // particle mass, smoothing length, contact radius, sensor radius
    run: vec4f,    // t_end, history capacity, first node partial, first element partial
    m0: vec4f,     // 2024-T3 (skin): E, nu, yield stress, hardening modulus
    m1: vec4f,     // 7075-T6 (stringers, frames)
    mx: vec4f,     // failure plastic strain of 0 and 1, density of 0 and 1
    bird: vec4f,   // density, sound speed, Tait exponent, hash buckets
    k: vec4f,      // SPH alpha, SPH beta, contact stiffness factor, hourglass factor
    k2: vec4f,     // shear correction, SPH CFL, shell CFL, frame thickness
    gz: array<vec4f, 2>,   // Gauss points through the thickness
    gw: array<vec4f, 2>,   // and their weights
}

struct State {
    time: f32, dt: f32, dt_prev: f32, dtm: f32,
    steps: f32, t_contact: f32, contact_fz: f32, center_fz: f32,
    impulse: f32, panel_work: f32, hg_work: f32, bird_work: f32,
    failed: f32, failed_skin: f32, max_ep: f32, max_defl: f32,
    ecs: f32, running: f32, primed: f32, bird_ke: f32,
    panel_ke: f32, pad0: f32, pad1: f32, pad2: f32,
}

struct EBin { c: vec3f, r: f32, cell: vec3i, next: i32 }

@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read> S: State;
@group(0) @binding(1) var<storage, read_write> Sw: State;
@group(0) @binding(2) var<storage, read_write> part: array<vec4f>;   // per particle: (x, rho), (v, p)
@group(0) @binding(2) var<storage, read> part_r: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> pacc: array<vec4f>;   // SPH acceleration, drho/dt
@group(0) @binding(3) var<storage, read> pacc_r: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> pfc: array<vec4f>;    // contact force, 1 inside the sensor disc
@group(0) @binding(4) var<storage, read> pfc_r: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> pbin: array<vec4i>;   // cell, next in the bucket
@group(0) @binding(5) var<storage, read> pbin_r: array<vec4i>;
@group(0) @binding(6) var<storage, read_write> phead: array<i32>;
@group(0) @binding(6) var<storage, read_write> phead_a: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read> phead_r: array<i32>;
@group(0) @binding(7) var<storage, read> ninfo: array<vec4f>;        // per node: (x0, rotary inertia), (fixed, skin, 0, 0)
@group(0) @binding(8) var<storage, read_write> nx: array<vec4f>;     // position, mass
@group(0) @binding(8) var<storage, read> nx_r: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> nvw: array<vec4f>;    // per node: velocity, angular velocity
@group(0) @binding(9) var<storage, read> nvw_r: array<vec4f>;
@group(0) @binding(10) var<storage, read_write> nfc: array<u32>;     // contact force, f32 bits, 3 per node
@group(0) @binding(10) var<storage, read_write> nfc_a: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read> nfc_r: array<u32>;
@group(0) @binding(11) var<storage, read> adj: array<u32>;           // per node: first corner, then the corners 4 e + k
@group(0) @binding(12) var<storage, read> einfo: array<vec4f>;       // per element: nodes, (thickness, material, 0, 0)
@group(0) @binding(13) var<storage, read_write> estate: array<f32>;  // per element, 32: see shell
@group(0) @binding(13) var<storage, read> estate_r: array<f32>;
@group(0) @binding(14) var<storage, read_write> eflag: array<u32>;   // 1: eroded
@group(0) @binding(14) var<storage, read> eflag_r: array<u32>;
@group(0) @binding(15) var<storage, read_write> efor: array<vec4f>;  // per element corner: force, moment
@group(0) @binding(15) var<storage, read> efor_r: array<vec4f>;
@group(0) @binding(16) var<storage, read_write> ebin: array<EBin>;
@group(0) @binding(16) var<storage, read> ebin_r: array<EBin>;
@group(0) @binding(17) var<storage, read_write> ehead: array<i32>;
@group(0) @binding(17) var<storage, read_write> ehead_a: array<atomic<i32>>;
@group(0) @binding(17) var<storage, read> ehead_r: array<i32>;
@group(0) @binding(18) var<storage, read_write> red: array<vec4f>;   // partials, 2 per workgroup
@group(0) @binding(18) var<storage, read> red_r: array<vec4f>;
@group(0) @binding(19) var<storage, read_write> hist: array<vec4f>;  // per step: time, contact force, sensor force, max deflection
@group(0) @binding(20) var<storage, read_write> rbuf: array<f32>;    // for the page: particles (x, p), nodes (x), element strain

const WG = 64u;
var<workgroup> wa: array<vec4f, 64>;
var<workgroup> wb: array<vec4f, 64>;
var<workgroup> wc: array<vec4f, 64>;
var<workgroup> wd: array<vec4f, 64>;
var<workgroup> we: array<vec4f, 64>;

fn n_part() -> u32 { return u32(U.n.y); }
fn n_node() -> u32 { return u32(U.n.z); }
fn n_elem() -> u32 { return u32(U.n.w); }
fn groups(n: u32) -> u32 { return (n + WG - 1u) / WG; }

// ---------------------------------------------------------------------------
// neighbour search: a hashed grid with an exact-cell test (bin_particles and
// bin_elements in strike.goose). The hash differs, the neighbours do not.

fn cell_of(v: vec3f, size: f32) -> vec3i { return vec3i(floor(v / size)); }

fn hash3(c: vec3i) -> u32 {
    let u = bitcast<vec3u>(c);
    return ((u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u)) & (u32(U.bird.w) - 1u);
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i < u32(U.bird.w) { phead[i] = -1; ehead[i] = -1; }
    if i < 3u * n_node() { nfc[i] = 0u; }
}

@compute @workgroup_size(64)
fn bin_particles(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= n_part() || S.running == 0.0 { return; }
    let c = cell_of(part_r[2u * i].xyz, 2.0 * U.sph.y);
    let next = atomicExchange(&phead_a[hash3(c)], i32(i));
    pbin[i] = vec4i(c, next);
}

// ---------------------------------------------------------------------------
// SPH: continuity, Tait pressure, momentum with Monaghan viscosity
// (sph_forces). Each particle sums over its own neighbours.

fn sound(rho: f32) -> f32 {
    let x = rho / U.bird.x;
    return U.bird.y * x * x * x;
}

fn tait(rho: f32) -> f32 {
    let b = U.bird.x * U.bird.y * U.bird.y / U.bird.z;
    let x = rho / U.bird.x;
    let x2 = x * x;
    // cavitation: the gelatin carries no tension
    return max(b * (x2 * x2 * x2 * x - 1.0), 0.0);
}

@compute @workgroup_size(64)
fn sph(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= n_part() || S.running == 0.0 { return; }
    let pm = U.sph.x;
    let h = U.sph.y;
    let sigma = 1.0 / (3.14159265358979 * h * h * h);
    let h2 = 4.0 * h * h;
    let xi = part_r[2u * i];
    let vi = part_r[2u * i + 1u];
    let ri = xi.w;
    let pri = vi.w / (ri * ri);
    let ci = sound(ri);
    let c0 = pbin_r[i].xyz;
    var a = vec3f(0.0);
    var drho = 0.0;
    for (var dk = -1; dk <= 1; dk++) {
        for (var dj = -1; dj <= 1; dj++) {
            for (var di = -1; di <= 1; di++) {
                let c = c0 + vec3i(di, dj, dk);
                var j = phead_r[hash3(c)];
                while j >= 0 {
                    let b = pbin_r[j];
                    if u32(j) != i && all(b.xyz == c) {
                        let xj = part_r[2 * j];
                        let d = xi.xyz - xj.xyz;
                        let r2 = dot(d, d);
                        if r2 < h2 && r2 > 0.0 {
                            let r = sqrt(r2);
                            let q = r / h;
                            let dw = select(-0.75 * sigma / h * (2.0 - q) * (2.0 - q),
                                            sigma / h * (-3.0 * q + 2.25 * q * q), q < 1.0);
                            let g = dw * d / r;
                            let vj = part_r[2 * j + 1];
                            let v = vi.xyz - vj.xyz;
                            drho += pm * dot(v, g);
                            let rj = xj.w;
                            var visc = 0.0;
                            let vr = dot(v, d);
                            if vr < 0.0 {
                                let mu = h * vr / (r2 + 0.01 * h * h);
                                let cbar = 0.5 * (ci + sound(rj));
                                let rbar = 0.5 * (ri + rj);
                                visc = (-U.k.x * cbar * mu + U.k.y * mu * mu) / rbar;
                            }
                            a -= pm * (pri + vj.w / (rj * rj) + visc) * g;
                        }
                    }
                    j = b.w;
                }
            }
        }
    }
    pacc[i] = vec4f(a, drho);
}

// ---------------------------------------------------------------------------
// shells (shell_element, plane_stress_return)

fn material(mi: u32) -> vec4f { return select(U.m0, U.m1, mi == 1u); }

// Plane-stress J2 return (Simo and Taylor) of the trial stress s.
// Returns the stress and the plastic strain increment.
fn plane_stress_return(s: vec3f, ep: f32, e: f32, nu: f32, sy0: f32, hh: f32) -> vec4f {
    let a1 = s.x + s.y;
    let a2 = s.x - s.y;
    let a3 = s.z;
    let j2 = a1 * a1 / 12.0 + a2 * a2 / 4.0 + a3 * a3;
    let ry = sy0 + hh * ep;
    if 3.0 * j2 <= ry * ry { return vec4f(s, 0.0); }
    let g2 = e / (1.0 + nu);
    let ka = e / (3.0 * (1.0 - nu));
    let aa = a1 * a1 / 12.0;
    let bb = a2 * a2 / 4.0 + a3 * a3;
    let s23 = sqrt(2.0 / 3.0);
    var dg = 0.0;
    for (var it = 0; it < 30; it++) {
        let d1 = 1.0 + ka * dg;
        let d2 = 1.0 + g2 * dg;
        let jj = aa / (d1 * d1) + bb / (d2 * d2);
        let djj = -2.0 * ka * aa / (d1 * d1 * d1) - 2.0 * g2 * bb / (d2 * d2 * d2);
        let qn = sqrt(2.0 * jj);
        let rr = sy0 + hh * (ep + s23 * dg * qn);
        let phi = jj - rr * rr / 3.0;
        let ddep = s23 * (qn + dg * djj / qn);
        let dphi = djj - 2.0 / 3.0 * rr * hh * ddep;
        let step = phi / dphi;
        dg = max(dg - step, 0.0);
        // f32: stop at a relative change of 1e-6 (strike.goose: 1e-12 in f64)
        if abs(step) <= 1.0e-6 * (abs(dg) + 1.0e-20) { break; }
    }
    let d1 = 1.0 + ka * dg;
    let d2 = 1.0 + g2 * dg;
    let b1 = a1 / d1;
    let b2 = a2 / d2;
    let qn = sqrt(2.0 * (b1 * b1 / 12.0 + b2 * b2 / 4.0 + a3 * a3 / (d2 * d2)));
    return vec4f(0.5 * (b1 + b2), 0.5 * (b1 - b2), a3 / d2, s23 * dg * qn);
}

// estate per element: 0-14 stress (xx, yy, xy) at the 5 Gauss points,
// 15-19 plastic strain, 20-21 transverse shear stress, 22-26 hourglass
// forces and moments, 27 the largest plastic strain, 28 the hourglass work
// of this step.
@compute @workgroup_size(64)
fn shell(@builtin(global_invocation_id) id: vec3u) {
    let e = id.x;
    if e >= n_elem() || S.running == 0.0 { return; }
    let st = 32u * e;
    if eflag[e] != 0u {
        // eroded: no more forces
        for (var k = 0u; k < 8u; k++) { efor[8u * e + k] = vec4f(0.0); }
        estate[st + 28u] = 0.0;
        return;
    }
    let step_dt = S.dtm;
    let nd = vec4u(einfo[2u * e]);
    let ep = einfo[2u * e + 1u];
    let x0 = nx_r[nd.x].xyz; let x1 = nx_r[nd.y].xyz; let x2 = nx_r[nd.z].xyz; let x3 = nx_r[nd.w].xyz;
    // local frame from the diagonals
    let e3 = normalize(cross(x2 - x0, x3 - x1));
    let e1 = normalize((x1 - x0) - dot(x1 - x0, e3) * e3);
    let e2 = cross(e3, e1);
    let c = 0.25 * (x0 + x1 + x2 + x3);
    let xl = vec4f(dot(x0 - c, e1), dot(x1 - c, e1), dot(x2 - c, e1), dot(x3 - c, e1));
    let yl = vec4f(dot(x0 - c, e2), dot(x1 - c, e2), dot(x2 - c, e2), dot(x3 - c, e2));
    var vx: vec4f; var vy: vec4f; var vz: vec4f; var tx: vec4f; var ty: vec4f;
    for (var k = 0u; k < 4u; k++) {
        let u = nvw_r[2u * nd[k]].xyz;
        let w = nvw_r[2u * nd[k] + 1u].xyz;
        vx[k] = dot(u, e1); vy[k] = dot(u, e2); vz[k] = dot(u, e3);
        tx[k] = dot(w, e1); ty[k] = dot(w, e2);
    }
    let area = 0.5 * ((xl.z - xl.x) * (yl.w - yl.y) + (xl.y - xl.w) * (yl.z - yl.x));
    let inv2a = 0.5 / area;
    let b1 = vec4f(yl.y - yl.w, yl.z - yl.x, yl.w - yl.y, yl.x - yl.z) * inv2a;
    let b2 = vec4f(xl.w - xl.y, xl.x - xl.z, xl.y - xl.w, xl.z - xl.x) * inv2a;
    // membrane, bending and transverse shear rates
    let dxx = dot(b1, vx);
    let dyy = dot(b2, vy);
    let gxy = dot(b2, vx) + dot(b1, vy);
    let kxx = dot(b1, ty);
    let kyy = -dot(b2, tx);
    let kxy = dot(b2, ty) - dot(b1, tx);
    let gxz = dot(b1, vz) + 0.25 * (ty.x + ty.y + ty.z + ty.w);
    let gyz = dot(b2, vz) - 0.25 * (tx.x + tx.y + tx.z + tx.w);
    let t = ep.x;
    let mi = u32(ep.y);
    let m = material(mi);
    let em = m.x; let nu = m.y;
    let c11 = em / (1.0 - nu * nu);
    let gm = 0.5 * em / (1.0 + nu);
    var nn = vec3f(0.0);   // membrane forces xx, yy, xy
    var mm = vec3f(0.0);   // moments
    var epm = 0.0;
    for (var g = 0u; g < 5u; g++) {
        let z = U.gz[g / 4u][g % 4u] * 0.5 * t;
        let w = U.gw[g / 4u][g % 4u] * 0.5 * t;
        let ex = (dxx + z * kxx) * step_dt;
        let ey = (dyy + z * kyy) * step_dt;
        let exy = (gxy + z * kxy) * step_dt;
        let s = st + 3u * g;
        let trial = vec3f(estate[s] + c11 * (ex + nu * ey), estate[s + 1u] + c11 * (ey + nu * ex), estate[s + 2u] + gm * exy);
        let r = plane_stress_return(trial, estate[st + 15u + g], em, nu, m.z, m.w);
        estate[s] = r.x; estate[s + 1u] = r.y; estate[s + 2u] = r.z;
        let epg = estate[st + 15u + g] + r.w;
        estate[st + 15u + g] = epg;
        epm = max(epm, epg);
        nn += w * r.xyz;
        mm += w * z * r.xyz;
    }
    estate[st + 27u] = epm;
    let kappa = U.k2.x;
    estate[st + 20u] += kappa * gm * gxz * step_dt;
    estate[st + 21u] += kappa * gm * gyz * step_dt;
    let qx = t * estate[st + 20u];
    let qy = t * estate[st + 21u];
    // hourglass control (Belytschko-Tsay), stiffness form
    let hx = xl.x - xl.y + xl.z - xl.w;
    let hy = yl.x - yl.y + yl.z - yl.w;
    let gam = vec4f(1.0, -1.0, 1.0, -1.0) - hx * b1 - hy * b2;
    let bb = dot(b1, b1) + dot(b2, b2);
    var q = array<f32, 5>(dot(gam, vx), dot(gam, vy), dot(gam, vz), dot(gam, tx), dot(gam, ty));
    let hgr = U.k.w;
    let cm = hgr * em * t * area * bb / 8.0;
    let cth = hgr * em * t * t * t * area * bb / 192.0;
    let cw = hgr * kappa * gm * t * t * t * bb / 12.0;
    var cs = array<f32, 5>(cm, cm, cw, cth, cth);
    var hgw = 0.0;
    var hg: array<f32, 5>;
    for (var k = 0u; k < 5u; k++) {
        hg[k] = estate[st + 22u + k] + cs[k] * q[k] * step_dt;
        estate[st + 22u + k] = hg[k];
        hgw += hg[k] * q[k];
    }
    estate[st + 28u] = step_dt * hgw;
    // corner forces and moments, then to the global frame
    let fx = area * (b1 * nn.x + b2 * nn.z) + gam * hg[0];
    let fy = area * (b2 * nn.y + b1 * nn.z) + gam * hg[1];
    let fz = area * (b1 * qx + b2 * qy) + gam * hg[2];
    let mx = area * (-b2 * mm.y - b1 * mm.z - 0.25 * qy) + gam * hg[3];
    let my = area * (b1 * mm.x + b2 * mm.z + 0.25 * qx) + gam * hg[4];
    for (var k = 0u; k < 4u; k++) {
        efor[8u * e + 2u * k] = vec4f(-(fx[k] * e1 + fy[k] * e2 + fz[k] * e3), 0.0);
        efor[8u * e + 2u * k + 1u] = vec4f(-(mx[k] * e1 + my[k] * e2), 0.0);
    }
    let epf = select(U.mx.x, U.mx.y, mi == 1u);
    if epm >= epf { eflag[e] = 1u; }
}

// ---------------------------------------------------------------------------
// contact (bin_elements, contact, closest_tri)

@compute @workgroup_size(64)
fn bin_elements(@builtin(global_invocation_id) id: vec3u) {
    let e = id.x;
    if e >= n_elem() || S.running == 0.0 || eflag_r[e] != 0u { return; }
    let c = cell_of(ebin[e].c, S.ecs);
    ebin[e].cell = c;
    ebin[e].next = atomicExchange(&ehead_a[hash3(c)], i32(e));
}

// Closest point of triangle abc to p (Ericson, Real-Time Collision
// Detection, 5.1.5). Returns the barycentric weights of a, b and c.
fn closest_tri(p: vec3f, a: vec3f, b: vec3f, c: vec3f) -> vec3f {
    let ab = b - a; let ac = c - a; let ap = p - a;
    let d1 = dot(ab, ap); let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 { return vec3f(1.0, 0.0, 0.0); }
    let bp = p - b;
    let d3 = dot(ab, bp); let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 { return vec3f(0.0, 1.0, 0.0); }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return vec3f(1.0 - v, v, 0.0);
    }
    let cp = p - c;
    let d5 = dot(ab, cp); let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 { return vec3f(0.0, 0.0, 1.0); }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return vec3f(1.0 - w, 0.0, w);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return vec3f(0.0, 1.0 - w, w);
    }
    let den = 1.0 / (va + vb + vc);
    let v = vb * den;
    let w = vc * den;
    return vec3f(1.0 - v - w, v, w);
}

fn add_nfc(i: u32, v: f32) {
    var old = atomicLoad(&nfc_a[i]);
    loop {
        let r = atomicCompareExchangeWeak(&nfc_a[i], old, bitcast<u32>(bitcast<f32>(old) + v));
        if r.exchanged { break; }
        old = r.old_value;
    }
}

// Penalty contact: the force on each particle, and on the nodes it pushes.
@compute @workgroup_size(64)
fn contact(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= n_part() || S.running == 0.0 { return; }
    let p = part_r[2u * i].xyz;
    let pm = U.sph.x;
    let prad = U.sph.z;
    let dt = S.dt;
    var f = vec3f(0.0);
    if U.n.x == 1.0 {
        let pen = prad - p.z;
        if pen > 0.0 { f = vec3f(0.0, 0.0, U.k.z * pm / (dt * dt) * pen); }
    } else {
        let c0 = cell_of(p, S.ecs);
        var best = -1;
        var bestd = 1.0e30;
        var bw = vec4f(0.0);
        var bq = vec3f(0.0);
        for (var dk = -1; dk <= 1; dk++) {
            for (var dj = -1; dj <= 1; dj++) {
                for (var di = -1; di <= 1; di++) {
                    let c = c0 + vec3i(di, dj, dk);
                    var e = ehead_r[hash3(c)];
                    while e >= 0 {
                        let b = ebin_r[e];
                        if all(b.cell == c) {
                            let gap = prad + 0.5 * einfo[2 * e + 1].x;
                            let dc = p - b.c;
                            let lim = b.r + gap;
                            if dot(dc, dc) < lim * lim {
                                let nd = vec4u(einfo[2 * e]);
                                let xa = nx_r[nd.x].xyz; let xb = nx_r[nd.y].xyz;
                                let xc = nx_r[nd.z].xyz; let xd = nx_r[nd.w].xyz;
                                // the quad as triangles abc and acd
                                let w1 = closest_tri(p, xa, xb, xc);
                                let q1 = xa * w1.x + xb * w1.y + xc * w1.z;
                                let w2 = closest_tri(p, xa, xc, xd);
                                let q2 = xa * w2.x + xc * w2.y + xd * w2.z;
                                let s1 = dot(p - q1, p - q1);
                                let s2 = dot(p - q2, p - q2);
                                let sd = min(s1, s2);
                                if sd < gap * gap && sqrt(sd) - gap < bestd {
                                    bestd = sqrt(sd) - gap;
                                    best = e;
                                    if s1 <= s2 { bw = vec4f(w1, 0.0); bq = q1; }
                                    else { bw = vec4f(w2.x, 0.0, w2.y, w2.z); bq = q2; }
                                }
                            }
                        }
                        e = b.next;
                    }
                }
            }
        }
        if best >= 0 {
            let ln = length(p - bq);
            if ln >= 1.0e-12 {
                let nrm = (p - bq) / ln;
                let nd = vec4u(einfo[2 * best]);
                // the reduced mass of the particle and the nodes it pushes
                let mn = dot(bw, vec4f(nx_r[nd.x].w, nx_r[nd.y].w, nx_r[nd.z].w, nx_r[nd.w].w));
                let mr = pm * mn / (pm + mn);
                f = U.k.z * mr / (dt * dt) * (-bestd) * nrm;
                for (var k = 0u; k < 4u; k++) {
                    if bw[k] != 0.0 {
                        add_nfc(3u * nd[k], -bw[k] * f.x);
                        add_nfc(3u * nd[k] + 1u, -bw[k] * f.y);
                        add_nfc(3u * nd[k] + 2u, -bw[k] * f.z);
                    }
                }
            }
        }
    }
    let r = U.sph.w;
    pfc[i] = vec4f(f, select(0.0, 1.0, p.x * p.x + p.y * p.y < r * r));
}

// ---------------------------------------------------------------------------
// central differences (integrate), with the partial sums for reduce.
// Before the first step (primed = 0) these kernels only make the partials.

fn psum(a: vec4f, b: vec4f) -> vec4f { return vec4f(max(a.x, b.x), a.yzw + b.yzw); }

@compute @workgroup_size(64)
fn integrate_particles(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) l: u32,
                       @builtin(workgroup_id) wid: vec3u) {
    let i = id.x;
    let pm = U.sph.x;
    var a0 = vec4f(0.0);   // max(c + |v|), contact force, sensor force, bird work
    var a1 = vec4f(0.0);   // bird kinetic energy
    if i < n_part() {
        var x = part[2u * i];
        var v = part[2u * i + 1u];
        if S.primed == 1.0 && S.running == 1.0 {
            let pa = pacc_r[i];
            let fc = pfc_r[i];
            let a = pa.xyz + fc.xyz / pm;
            let vm = v.xyz + 0.5 * a * S.dtm;
            a0.w = -pm * dot(pa.xyz, vm) * S.dtm;
            v = vec4f(v.xyz + a * S.dtm, 0.0);
            x = vec4f(x.xyz + v.xyz * S.dt, x.w + pa.w * S.dt);
            v.w = tait(x.w);
            part[2u * i] = x;
            part[2u * i + 1u] = v;
            a0.y = fc.z;
            a0.z = fc.z * fc.w;
        }
        a0.x = sound(x.w) + length(v.xyz);
        a1.x = 0.5 * pm * dot(v.xyz, v.xyz);
    }
    wa[l] = a0; wb[l] = a1;
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        workgroupBarrier();
        if l < s { wa[l] = psum(wa[l], wa[l + s]); wb[l] += wb[l + s]; }
    }
    if l == 0u { red[2u * wid.x] = wa[0]; red[2u * wid.x + 1u] = wb[0]; }
}

fn nsum(a: vec4f, b: vec4f) -> vec4f { return vec4f(a.x + b.x, max(a.y, b.y), a.z + b.z, 0.0); }

@compute @workgroup_size(64)
fn integrate_nodes(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) l: u32,
                   @builtin(workgroup_id) wid: vec3u) {
    let n = id.x;
    let nn = n_node();
    var a0 = vec4f(0.0);   // panel work, deflection, panel kinetic energy
    if n < nn {
        let i0 = ninfo[2u * n];
        let i1 = ninfo[2u * n + 1u];
        var x = nx[n];
        var v = nvw[2u * n].xyz;
        var w = nvw[2u * n + 1u].xyz;
        if S.primed == 1.0 && S.running == 1.0 && i1.x == 0.0 {
            var f = vec3f(0.0);
            var mo = vec3f(0.0);
            for (var s = adj[n]; s < adj[n + 1u]; s++) {
                let corner = adj[nn + 1u + s];
                f += efor_r[2u * corner].xyz;
                mo += efor_r[2u * corner + 1u].xyz;
            }
            let fc = vec3f(bitcast<f32>(nfc_r[3u * n]), bitcast<f32>(nfc_r[3u * n + 1u]), bitcast<f32>(nfc_r[3u * n + 2u]));
            let a = (f + fc) / x.w;
            let al = mo / i0.w;
            let vm = v + 0.5 * a * S.dtm;
            let wm = w + 0.5 * al * S.dtm;
            a0.x = -(dot(f, vm) + dot(mo, wm)) * S.dtm;
            v += a * S.dtm;
            w += al * S.dtm;
            x = vec4f(x.xyz + v * S.dt, x.w);
            nx[n] = x;
            nvw[2u * n] = vec4f(v, 0.0);
            nvw[2u * n + 1u] = vec4f(w, 0.0);
        }
        if i1.y == 1.0 { a0.y = length(x.xyz - i0.xyz); }
        a0.z = 0.5 * x.w * dot(v, v) + 0.5 * i0.w * dot(w, w);
    }
    wa[l] = a0;
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        workgroupBarrier();
        if l < s { wa[l] = nsum(wa[l], wa[l + s]); }
    }
    if l == 0u { red[2u * (u32(U.run.z) + wid.x)] = wa[0]; }
}

fn esum(a: vec4f, b: vec4f) -> vec4f { return vec4f(min(a.x, b.x), max(a.y, b.y), a.z + b.z, max(a.w, b.w)); }

// After the nodes move: the centre and radius of each element for the
// contact search, and its stable time step (shell_dt).
@compute @workgroup_size(64)
fn elem_geom(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) l: u32,
             @builtin(workgroup_id) wid: vec3u) {
    let e = id.x;
    var a0 = vec4f(1.0, 0.0, 0.0, 0.0);   // min dt, max radius, hourglass work, max plastic strain
    var a1 = vec4f(0.0);                  // eroded, eroded in the skin
    if e < n_elem() {
        let nd = vec4u(einfo[2u * e]);
        let ep = einfo[2u * e + 1u];
        let xa = nx_r[nd.x].xyz; let xb = nx_r[nd.y].xyz; let xc = nx_r[nd.z].xyz; let xd = nx_r[nd.w].xyz;
        let c = 0.25 * (xa + xb + xc + xd);
        let r = max(max(length(xa - c), length(xb - c)), max(length(xc - c), length(xd - c)));
        ebin[e].c = c;
        ebin[e].r = r;
        a0.y = r;
        let mi = u32(ep.y);
        if eflag_r[e] == 0u {
            let area = 0.5 * length(cross(xc - xa, xd - xb));
            let lmax = max(max(length(xa - xb), length(xb - xc)), max(length(xc - xd), length(xd - xa)));
            let m = material(mi);
            let rho = select(U.mx.z, U.mx.w, mi == 1u);
            a0.x = area / lmax / sqrt(m.x / (rho * (1.0 - m.y * m.y)));
        } else {
            a1 = vec4f(1.0, select(0.0, 1.0, mi == 0u), 0.0, 0.0);
        }
        a0.z = estate_r[32u * e + 28u];
        a0.w = estate_r[32u * e + 27u];
    }
    wa[l] = a0; wb[l] = a1;
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        workgroupBarrier();
        if l < s { wa[l] = esum(wa[l], wa[l + s]); wb[l] += wb[l + s]; }
    }
    if l == 0u { red[2u * (u32(U.run.w) + wid.x)] = wa[0]; red[2u * (u32(U.run.w) + wid.x) + 1u] = wb[0]; }
}

// The end of a step (or the set-up, when primed = 0): the sums, the time,
// the history, and the time step and contact cell size of the next step.
@compute @workgroup_size(64)
fn reduce(@builtin(local_invocation_index) l: u32) {
    let gp = groups(n_part());
    let gn = groups(n_node());
    let ge = groups(n_elem());
    let on = u32(U.run.z);
    let oe = u32(U.run.w);
    var p0 = vec4f(0.0); var p1 = vec4f(0.0);
    var q0 = vec4f(0.0);
    var e0 = vec4f(1.0, 0.0, 0.0, 0.0); var e1 = vec4f(0.0);
    for (var g = l; g < gp; g += WG) { p0 = psum(p0, red_r[2u * g]); p1 += red_r[2u * g + 1u]; }
    for (var g = l; g < gn; g += WG) { q0 = nsum(q0, red_r[2u * (on + g)]); }
    for (var g = l; g < ge; g += WG) { e0 = esum(e0, red_r[2u * (oe + g)]); e1 += red_r[2u * (oe + g) + 1u]; }
    wa[l] = p0; wb[l] = p1; wc[l] = q0; wd[l] = e0; we[l] = e1;
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        workgroupBarrier();
        if l < s {
            wa[l] = psum(wa[l], wa[l + s]); wb[l] += wb[l + s];
            wc[l] = nsum(wc[l], wc[l + s]);
            wd[l] = esum(wd[l], wd[l + s]); we[l] += we[l + s];
        }
    }
    if l != 0u || Sw.running == 0.0 { return; }
    p0 = wa[0]; p1 = wb[0]; q0 = wc[0]; e0 = wd[0]; e1 = we[0];
    let panel = U.n.x == 0.0;
    if Sw.primed == 1.0 {
        // a step ended: contact, then integrate, of strike.goose
        Sw.contact_fz = p0.y;
        Sw.center_fz = p0.z;
        if p0.y > 0.0 && Sw.t_contact < 0.0 { Sw.t_contact = Sw.time; }
        Sw.impulse += p0.y * Sw.dt;
        Sw.bird_work += p0.w;
        if panel {
            Sw.panel_work += q0.x;
            Sw.hg_work += e0.z;
            Sw.max_defl = max(Sw.max_defl, q0.y);
            Sw.max_ep = e0.w;
            Sw.failed = e1.x;
            Sw.failed_skin = e1.y;
        }
        Sw.dt_prev = Sw.dt;
        Sw.time += Sw.dt;
        let k = u32(Sw.steps);
        if k < u32(U.run.y) { hist[k] = vec4f(Sw.time, Sw.contact_fz, Sw.center_fz, Sw.max_defl); }
        Sw.steps += 1.0;
    }
    Sw.bird_ke = p1.x;
    Sw.panel_ke = q0.z;
    // stable_dt for the next step
    var d = U.k2.y * U.sph.y / p0.x;
    if panel { d = min(d, U.k2.z * e0.x); }
    Sw.dtm = select(0.5 * (d + Sw.dt_prev), d, Sw.steps == 0.0);
    Sw.dt = d;
    // a particle in contact lies within the element radius + gap of the centroid
    Sw.ecs = e0.y + U.sph.z + 0.5 * U.k2.w + 1.0e-4;
    Sw.primed = 1.0;
    Sw.running = select(0.0, 1.0, Sw.time < U.run.x);
}

// What the page draws: particle positions and pressures, node positions,
// and the largest plastic strain of each element (-1: eroded).
@compute @workgroup_size(64)
fn render(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let np = n_part();
    let nn = n_node();
    if i < np {
        let x = part_r[2u * i];
        rbuf[4u * i] = x.x; rbuf[4u * i + 1u] = x.y; rbuf[4u * i + 2u] = x.z;
        rbuf[4u * i + 3u] = part_r[2u * i + 1u].w;
    }
    if i < nn {
        let x = nx_r[i];
        rbuf[4u * np + 3u * i] = x.x; rbuf[4u * np + 3u * i + 1u] = x.y; rbuf[4u * np + 3u * i + 2u] = x.z;
    }
    if i < n_elem() {
        rbuf[4u * np + 3u * nn + i] = select(estate_r[32u * i + 27u], -1.0, eflag_r[i] != 0u);
    }
}
