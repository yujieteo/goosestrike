// goosestrike: explicit dynamics of a 4 lb bird striking a stiffened
// aluminium fuselage panel (14 CFR 25.571(e)(1)).
//
// Bird: smoothed particle hydrodynamics (SPH), a cylinder with hemispherical
// ends (L/D = 2) of water-like gelatin, Tait equation of state.
// Panel: Belytschko-Lin-Tsay 4-node shells, one-point in-plane quadrature,
// 5 Gauss points through the thickness, J2 plasticity with linear isotropic
// hardening (plane stress return of Simo and Taylor), stiffness hourglass
// control, element erosion at the failure plastic strain.
// Contact: penalty, each bird particle against its nearest shell element
// (or against a rigid wall at z = 0 for the validation case).
// Time integration: central differences with lumped masses and a stable
// time step from the smallest element and the SPH sound speed.
//
// Units are SI throughout: m, kg, s, N, Pa.
//
// All storage is fixed-size globals in static data. Nothing is allocated
// after start-up, and the time loop uses no data stack at all, which is
// what lets the program run on WebAssembly's linear memory.

import math;
import std;

// ---- capacities ----
let MAXP = 60000;           // SPH particles
let MAXN = 24000;           // shell nodes
let MAXE = 24000;           // shell elements
let NIP = 5;                // Gauss points through the thickness
let NHASH = 131072;         // hash buckets (a power of two)

// ---- the load case: 14 CFR 25.571(e)(1) ----
let BIRD_MASS = 4.0 * 0.45359237;      // 4 lb in kg
let V_C = 350.0 * 1852.0 / 3600.0;    // 350 KCAS in m/s, VMO of the A320 (EASA TCDS A.064)
let RHO_BIRD = 950.0;       // gelatin with 10 % porosity, kg/m^3
let C0_BIRD = 1480.0;       // bulk sound speed of water, m/s
let GAMMA_TAIT = 7.0;       // Tait exponent for water

// ---- the panel (representative narrow-body crown panel) ----
let R_FUS = 1.975;          // fuselage outer radius, m (3.95 m width)
let PANEL_LX = 1.0668;      // axial length: two 21 in frame bays, m
let PANEL_LS = 0.760;       // arc length: four 190 mm stringer bays, m
let T_SKIN = 0.0016;        // 0.063 in skin
let STR_H = 0.0254;         // stringer web height
let STR_F = 0.015;          // stringer free flange width
let T_STR = 0.0016;
let FRM_H = 0.0762;         // frame web height
let FRM_F = 0.025;          // frame inner flange width
let T_FRM = 0.0016;

// ---- materials: 0 = 2024-T3 (skin), 1 = 7075-T6 (stringers, frames) ----
// E, nu, rho, yield, ultimate, elongation at break (ASM data sheets).
var mat_e: f64[2] = [73.1e9, 71.7e9];
var mat_nu: f64[2] = [0.33, 0.33];
var mat_rho: f64[2] = [2780.0, 2810.0];
var mat_sy: f64[2] = [345.0e6, 503.0e6];
var mat_su: f64[2] = [483.0e6, 572.0e6];
var mat_el: f64[2] = [0.18, 0.11];
var mat_h: f64[2] = [0.0, 0.0];      // plastic hardening modulus, set in setup
var mat_epf: f64[2] = [0.0, 0.0];    // plastic strain at failure, set in setup

let HG_R = 0.05;            // hourglass stiffness coefficient
let KAPPA_S = 0.8333333333333334;  // transverse shear correction 5/6
let SPH_ALPHA = 1.0;        // Monaghan artificial viscosity
let SPH_BETA = 2.0;
let H_FACTOR = 1.2;         // smoothing length / particle spacing
let CFL_SPH = 0.25;
let CFL_SHELL = 0.9;
let K_CONTACT = 0.1;        // penalty stiffness as a fraction of m / dt^2

var gp_z: f64[5] = [-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664];
var gp_w: f64[5] = [0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189];

// ---- state ----
var mode: i64 = 0;          // 0: panel, 1: rigid wall at z = 0
var sim_time: f64 = 0.0;
var dt: f64 = 0.0;
var dt_prev: f64 = 0.0;
var nsteps: i64 = 0;
var t_contact: f64 = -1.0;

// particles
var np: i64 = 0;
var pm: f64 = 0.0;          // particle mass
var ph: f64 = 0.0;          // smoothing length
var prad: f64 = 0.0;        // contact radius
var pdx: f64 = 0.0;         // lattice spacing
var bird_d: f64 = 0.0;
var bird_l: f64 = 0.0;
var bird_v: f64 = 0.0;
var px: f64[MAXP * 3];
var pv: f64[MAXP * 3];
var pa: f64[MAXP * 3];
var prho: f64[MAXP];
var pdrho: f64[MAXP];
var pp: f64[MAXP];
var pc: f64[MAXP];
var pfc: f64[MAXP * 3];   // contact force on the particle
var pcell: i64[MAXP * 3];
var pnext: i32[MAXP];
var phead: i32[NHASH];

// nodes
var nn: i64 = 0;
var nx: f64[MAXN * 3];
var nx0: f64[MAXN * 3];
var nv: f64[MAXN * 3];
var nw: f64[MAXN * 3];
var nf: f64[MAXN * 3];
var nmo: f64[MAXN * 3];
var nfc: f64[MAXN * 3];   // contact force on the node
var nmass: f64[MAXN];
var ninert: f64[MAXN];
var nfix: i32[MAXN];
var nskin: i32[MAXN];

// elements
var ne: i64 = 0;
var en: i32[MAXE * 4];
var eth: f64[MAXE];
var emat: i32[MAXE];
var efail: i32[MAXE];
var esig: f64[MAXE * 15];   // NIP x (xx, yy, xy), local frame
var eep: f64[MAXE * 5];      // plastic strain per Gauss point
var etau: f64[MAXE * 2];     // transverse shear stress
var ehg: f64[MAXE * 5];      // hourglass forces and moments
var eepmax: f64[MAXE];
var ecen: f64[MAXE * 3];
var erad: f64[MAXE];
var ecell: i64[MAXE * 3];
var enext: i32[MAXE];
var ehead: i32[NHASH];
var ecs: f64 = 0.0;         // contact cell size

// diagnostics
var contact_fz: f64 = 0.0;      // total contact force normal to the target, N
var center_fz: f64 = 0.0;       // the part of it inside the sensor disc
var sensor_r: f64 = 0.0;
var impulse: f64 = 0.0;         // integral of contact_fz dt
// Work is integrated with the mean of the old and the new velocity, so
// that the sums close the energy balance of the central difference scheme.
var panel_work: f64 = 0.0;      // internal work on the panel (hourglass included), J
var hg_work: f64 = 0.0;         // the hourglass part of it, J
var bird_work: f64 = 0.0;       // internal energy of the bird (compression and viscosity), J
var cw_bird: f64 = 0.0;         // work of the contact forces on the bird, J
var cw_panel: f64 = 0.0;        // and on the panel
var n_failed: i64 = 0;
var n_failed_skin: i64 = 0;     // the part of n_failed in the skin (material 0)
var max_ep: f64 = 0.0;
var max_defl: f64 = 0.0;

// render buffers (f32 for the page)
var r_node: f32[MAXN * 3];
var r_part: f32[MAXP * 3];
var r_pp: f32[MAXP];
var r_eep: f32[MAXE];

// ---------------------------------------------------------------------------
// mesh building

fn add_node(x: f64, y: f64, z: f64, fix: bool, skin: bool) -> i64 {
    assert(nn < MAXN);
    let i = nn;
    nx[3 * i] = x; nx[3 * i + 1] = y; nx[3 * i + 2] = z;
    nfix[i] = if fix { 1 } else { 0 };
    nskin[i] = if skin { 1 } else { 0 };
    nn += 1;
    return i;
}

fn add_elem(a: i64, b: i64, c: i64, d: i64, t: f64, m: i32) {
    assert(ne < MAXE);
    en[4 * ne] = a as i32; en[4 * ne + 1] = b as i32;
    en[4 * ne + 2] = c as i32; en[4 * ne + 3] = d as i32;
    eth[ne] = t;
    emat[ne] = m;
    ne += 1;
}

// A point of the panel: x along the fuselage, s along the arc of the outer
// skin, d inward from the skin. The outer normal at s = 0 is +z.
fn panel_point(x: f64, s: f64, d: f64) -> f64, f64, f64 {
    let r = R_FUS - d;
    let a = s / R_FUS;
    return x, r * sin(a), r * cos(a) - R_FUS;
}

fn node_at(x: f64, s: f64, d: f64, fix: bool, skin: bool) -> i64 {
    let a, b, c = panel_point(x, s, d);
    return add_node(a, b, c, fix, skin);
}

fn divisions(len: f64, h: f64) -> i64 {
    let k = (len / h + 0.5) as i64;
    return max(k, 1);
}

// Skin grid (gx x gs elements), four Z stringers, two Z frames. The edges
// x = +-LX/2 and s = +-LS/2 lie on the neighbouring frames and stringers
// and are clamped.
fn build_panel(gx: i64, gs: i64) {
    let dxe = PANEL_LX / gx;
    let dse = PANEL_LS / gs;
    let x0 = -0.5 * PANEL_LX;
    let s0 = -0.5 * PANEL_LS;
    for j in 0..gs + 1 {
        for i in 0..gx + 1 {
            let fix = i == 0 || i == gx || j == 0 || j == gs;
            node_at(x0 + i * dxe, s0 + j * dse, 0.0, fix, true);
        }
    }
    for j in 0..gs {
        for i in 0..gx {
            let a = j * (gx + 1) + i;
            add_elem(a, a + 1, a + gx + 2, a + gx + 1, T_SKIN, 0);
        }
    }
    let he = 0.5 * (dxe + dse);
    // stringers at s = -3, -1, 1, 3 x LS/8; a Z: web, then a free flange
    let nw = divisions(STR_H, he);
    let nfl = divisions(STR_F, he);
    for k in 0..4 {
        let js = gs / 8 * (2 * k + 1);
        let s = s0 + js * dse;
        let base = nn;
        for l in 1..nw + 1 {
            for i in 0..gx + 1 {
                node_at(x0 + i * dxe, s, STR_H * l / nw, i == 0 || i == gx, false);
            }
        }
        for w in 1..nfl + 1 {
            for i in 0..gx + 1 {
                node_at(x0 + i * dxe, s + STR_F * w / nfl, STR_H, i == 0 || i == gx, false);
            }
        }
        for l in 0..nw + nfl {
            for i in 0..gx {
                let lo = if l == 0 { js * (gx + 1) + i } else { base + (l - 1) * (gx + 1) + i };
                let hi = base + l * (gx + 1) + i;
                add_elem(lo, lo + 1, hi + 1, hi, T_STR, 1);
            }
        }
    }
    // frames at x = -LX/4 and +LX/4; a Z: web, then an inner flange
    let nfw = divisions(FRM_H, he);
    let nff = divisions(FRM_F, he);
    for k in 0..2 {
        let ifr = gx / 4 * (2 * k + 1);
        let x = x0 + ifr * dxe;
        let base = nn;
        for l in 1..nfw + 1 {
            for j in 0..gs + 1 {
                node_at(x, s0 + j * dse, FRM_H * l / nfw, j == 0 || j == gs, false);
            }
        }
        for w in 1..nff + 1 {
            for j in 0..gs + 1 {
                node_at(x + FRM_F * w / nff, s0 + j * dse, FRM_H, j == 0 || j == gs, false);
            }
        }
        for l in 0..nfw + nff {
            for j in 0..gs {
                let lo = if l == 0 { j * (gx + 1) + ifr } else { base + (l - 1) * (gs + 1) + j };
                let lo2 = if l == 0 { (j + 1) * (gx + 1) + ifr } else { lo + 1 };
                let hi = base + l * (gs + 1) + j;
                add_elem(lo, lo2, hi + 1, hi, T_FRM, 1);
            }
        }
    }
}

// Lumped translational masses and rotary inertias.
fn lump_masses() {
    for i in 0..nn { nmass[i] = 0.0; ninert[i] = 0.0; }
    for e in 0..ne {
        let a = en[4 * e]; let b = en[4 * e + 1]; let c = en[4 * e + 2]; let d = en[4 * e + 3];
        let area = quad_area(a, b, c, d);
        let t = eth[e];
        let m = mat_rho[emat[e]] * t * area * 0.25;
        // Rotary inertia scaled up from t^2/12 so that the rotational
        // modes do not set the time step (a standard explicit practice).
        let rot = m * (t * t + area) / 12.0;
        nmass[a] += m; nmass[b] += m; nmass[c] += m; nmass[d] += m;
        ninert[a] += rot; ninert[b] += rot; ninert[c] += rot; ninert[d] += rot;
    }
}

fn quad_area(a: i32, b: i32, c: i32, d: i32) -> f64 {
    // half the cross product of the diagonals
    let r1x = nx[3 * c] - nx[3 * a]; let r1y = nx[3 * c + 1] - nx[3 * a + 1]; let r1z = nx[3 * c + 2] - nx[3 * a + 2];
    let r2x = nx[3 * d] - nx[3 * b]; let r2y = nx[3 * d + 1] - nx[3 * b + 1]; let r2z = nx[3 * d + 2] - nx[3 * b + 2];
    let cx = r1y * r2z - r1z * r2y; let cy = r1z * r2x - r1x * r2z; let cz = r1x * r2y - r1y * r2x;
    return 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
}

// ---------------------------------------------------------------------------
// the bird

fn inside_capsule(x: f64, y: f64, z: f64, r: f64, half: f64) -> bool {
    let zc = clamp(z, -half, half);
    let dz = z - zc;
    return x * x + y * y + dz * dz <= r * r;
}

// A capsule along z with its nose at z = nose_z, flying toward -z.
fn build_bird(spacing: f64, speed: f64, nose_z: f64) {
    // L/D = 2 with hemispherical ends: volume = 5 pi D^3 / 12
    let vol = BIRD_MASS / RHO_BIRD;
    bird_d = pow(12.0 * vol / (5.0 * PI), 1.0 / 3.0);
    bird_l = 2.0 * bird_d;
    bird_v = speed;
    let r = 0.5 * bird_d;
    let half = 0.5 * (bird_l - bird_d);
    let n = (bird_l / spacing) as i64 + 2;
    let nr = (bird_d / spacing) as i64 + 2;
    let zc = nose_z + 0.5 * bird_l;
    np = 0;
    for k in 0..n {
        let z = -0.5 * bird_l + (k + 0.5) * spacing;
        for j in 0..nr {
            let y = -0.5 * nr * spacing + (j + 0.5) * spacing;
            for i in 0..nr {
                let x = -0.5 * nr * spacing + (i + 0.5) * spacing;
                if inside_capsule(x, y, z, r, half) {
                    assert(np < MAXP);
                    px[3 * np] = x; px[3 * np + 1] = y; px[3 * np + 2] = z + zc;
                    pv[3 * np] = 0.0; pv[3 * np + 1] = 0.0; pv[3 * np + 2] = -speed;
                    prho[np] = RHO_BIRD;
                    pp[np] = 0.0;
                    pc[np] = C0_BIRD;
                    np += 1;
                }
            }
        }
    }
    // The lattice volume differs a little from the capsule's: the particle
    // mass makes the total exactly 4 lb.
    pm = BIRD_MASS / np;
    pdx = pow(pm / RHO_BIRD, 1.0 / 3.0);
    ph = H_FACTOR * pdx;
    prad = 0.5 * pdx;
}

// ---------------------------------------------------------------------------
// set-up

// mode 0: the panel; mode 1: a rigid wall at z = 0.
// cells: skin elements along x (a multiple of 8). spacing: bird lattice, m.
export fn setup(m: i32, cells: i32, spacing: f64, speed: f64) -> i32 {
    for k in 0..2 {
        let ey = mat_sy[k] / mat_e[k];
        let eu = mat_el[k];
        let et = (mat_su[k] - mat_sy[k]) / (eu - ey);
        mat_h[k] = mat_e[k] * et / (mat_e[k] - et);
        mat_epf[k] = eu - mat_su[k] / mat_e[k];
    }
    mode = m;
    sim_time = 0.0; dt = 0.0; dt_prev = 0.0; nsteps = 0; t_contact = -1.0;
    contact_fz = 0.0; center_fz = 0.0; impulse = 0.0; panel_work = 0.0;
    hg_work = 0.0; bird_work = 0.0; cw_bird = 0.0; cw_panel = 0.0;
    n_failed = 0; n_failed_skin = 0; max_ep = 0.0; max_defl = 0.0;
    nn = 0; ne = 0;
    var top = 0.0;
    if mode == 0 {
        let gx = (cells / 8 * 8) as i64;
        let gs = (gx * PANEL_LS / PANEL_LX / 8.0 + 0.5) as i64 * 8;
        build_panel(gx, gs);
        top = 0.5 * T_SKIN;
    }
    for i in 0..3 * nn {
        nx0[i] = nx[i]; nv[i] = 0.0; nw[i] = 0.0;
    }
    for e in 0..ne {
        efail[e] = 0; eepmax[e] = 0.0;
        for k in 0..15 { esig[15 * e + k] = 0.0; }
        for k in 0..5 { eep[5 * e + k] = 0.0; ehg[5 * e + k] = 0.0; }
        etau[2 * e] = 0.0; etau[2 * e + 1] = 0.0;
    }
    lump_masses();
    // nose a little clear of the target, so contact starts after a few steps
    build_bird(spacing, speed, top + 0.5 * spacing + 0.002);
    sensor_r = 0.125 * bird_d;
    return np as i32;
}

// ---------------------------------------------------------------------------
// neighbour search: a hashed grid with an exact-cell test, so that two
// cells sharing a bucket are never visited twice

fn cell_of(v: f64, size: f64) -> i64 { floor(v / size) as i64 }

fn hash3(i: i64, j: i64, k: i64) -> i64 {
    let h = (i * 73856093) ^ (j * 19349663) ^ (k * 83492791);
    return h % NHASH;
}

fn bin_particles() {
    let size = 2.0 * ph;
    for b in 0..NHASH { phead[b] = -1; }
    for i in 0..np {
        let ci = cell_of(px[3 * i], size);
        let cj = cell_of(px[3 * i + 1], size);
        let ck = cell_of(px[3 * i + 2], size);
        pcell[3 * i] = ci; pcell[3 * i + 1] = cj; pcell[3 * i + 2] = ck;
        let b = hash3(ci, cj, ck);
        pnext[i] = phead[b];
        phead[b] = i as i32;
    }
}

// ---------------------------------------------------------------------------
// SPH: continuity, Tait pressure, momentum with Monaghan viscosity

fn sph_forces() {
    let h = ph;
    let sigma = 1.0 / (PI * h * h * h);
    let h2 = 4.0 * h * h;
    for i in 0..np {
        pdrho[i] = 0.0;
        pa[3 * i] = 0.0; pa[3 * i + 1] = 0.0; pa[3 * i + 2] = 0.0;
    }
    bin_particles();
    for i in 0..np {
        let xi = px[3 * i]; let yi = px[3 * i + 1]; let zi = px[3 * i + 2];
        let vxi = pv[3 * i]; let vyi = pv[3 * i + 1]; let vzi = pv[3 * i + 2];
        let ri = prho[i];
        let pri = pp[i] / (ri * ri);
        let ci = pc[i];
        let ci0 = pcell[3 * i]; let cj0 = pcell[3 * i + 1]; let ck0 = pcell[3 * i + 2];
        for dk in -1..2 {
            for dj in -1..2 {
                for di in -1..2 {
                    let ci1 = ci0 + di; let cj1 = cj0 + dj; let ck1 = ck0 + dk;
                    var j = phead[hash3(ci1, cj1, ck1)] as i64;
                    while j >= 0 {
                        if j > i && pcell[3 * j] == ci1 && pcell[3 * j + 1] == cj1 && pcell[3 * j + 2] == ck1 {
                            let dx = xi - px[3 * j]; let dy = yi - px[3 * j + 1]; let dz = zi - px[3 * j + 2];
                            let r2 = dx * dx + dy * dy + dz * dz;
                            if r2 < h2 && r2 > 0.0 {
                                let r = sqrt(r2);
                                let q = r / h;
                                let dw = if q < 1.0 { sigma / h * (-3.0 * q + 2.25 * q * q) }
                                         else { -0.75 * sigma / h * (2.0 - q) * (2.0 - q) };
                                let gx = dw * dx / r; let gy = dw * dy / r; let gz = dw * dz / r;
                                let vx = vxi - pv[3 * j]; let vy = vyi - pv[3 * j + 1]; let vz = vzi - pv[3 * j + 2];
                                let vdotg = vx * gx + vy * gy + vz * gz;
                                pdrho[i] += pm * vdotg;
                                pdrho[j] += pm * vdotg;
                                let rj = prho[j];
                                var visc = 0.0;
                                let vr = vx * dx + vy * dy + vz * dz;
                                if vr < 0.0 {
                                    let mu = h * vr / (r2 + 0.01 * h * h);
                                    let cbar = 0.5 * (ci + pc[j]);
                                    let rbar = 0.5 * (ri + rj);
                                    visc = (-SPH_ALPHA * cbar * mu + SPH_BETA * mu * mu) / rbar;
                                }
                                let coef = pm * (pri + pp[j] / (rj * rj) + visc);
                                pa[3 * i] -= coef * gx; pa[3 * i + 1] -= coef * gy; pa[3 * i + 2] -= coef * gz;
                                pa[3 * j] += coef * gx; pa[3 * j + 1] += coef * gy; pa[3 * j + 2] += coef * gz;
                            }
                        }
                        j = pnext[j];
                    }
                }
            }
        }
    }
}

fn tait(rho: f64) -> f64 {
    let b = RHO_BIRD * C0_BIRD * C0_BIRD / GAMMA_TAIT;
    let x = rho / RHO_BIRD;
    let x2 = x * x;
    let p = b * (x2 * x2 * x2 * x - 1.0);
    // cavitation: the gelatin carries no tension
    return max(p, 0.0);
}

// ---------------------------------------------------------------------------
// shells

// Plane-stress J2 return (Simo and Taylor). s: trial stress (xx, yy, xy).
// Returns the stress and the plastic strain increment.
fn plane_stress_return(sxx: f64, syy: f64, sxy: f64, ep: f64, e: f64, nu: f64, sy0: f64, hh: f64) -> f64, f64, f64, f64 {
    let a1 = sxx + syy;
    let a2 = sxx - syy;
    let a3 = sxy;
    let j2 = a1 * a1 / 12.0 + a2 * a2 / 4.0 + a3 * a3;
    let ry = sy0 + hh * ep;
    if 3.0 * j2 <= ry * ry { return sxx, syy, sxy, 0.0; }
    let g2 = e / (1.0 + nu);            // 2G
    let ka = e / (3.0 * (1.0 - nu));
    let aa = a1 * a1 / 12.0;
    let bb = a2 * a2 / 4.0 + a3 * a3;
    var dg = 0.0;
    var dep = 0.0;
    for it in 0..30 {
        let d1 = 1.0 + ka * dg;
        let d2 = 1.0 + g2 * dg;
        let jj = aa / (d1 * d1) + bb / (d2 * d2);
        let djj = -2.0 * ka * aa / (d1 * d1 * d1) - 2.0 * g2 * bb / (d2 * d2 * d2);
        let qn = sqrt(2.0 * jj);
        dep = sqrt(2.0 / 3.0) * dg * qn;
        let rr = sy0 + hh * (ep + dep);
        let phi = jj - rr * rr / 3.0;
        let ddep = sqrt(2.0 / 3.0) * (qn + dg * djj / qn);
        let dphi = djj - 2.0 / 3.0 * rr * hh * ddep;
        let step = phi / dphi;
        dg -= step;
        if dg < 0.0 { dg = 0.0; }
        if abs(step) <= 1.0e-12 * (abs(dg) + 1.0e-20) { break; }
    }
    let d1 = 1.0 + ka * dg;
    let d2 = 1.0 + g2 * dg;
    let b1 = a1 / d1;
    let b2 = a2 / d2;
    let qn = sqrt(2.0 * (b1 * b1 / 12.0 + b2 * b2 / 4.0 + a3 * a3 / (d2 * d2)));
    dep = sqrt(2.0 / 3.0) * dg * qn;
    return 0.5 * (b1 + b2), 0.5 * (b1 - b2), a3 / d2, dep;
}

// Internal forces of one Belytschko-Lin-Tsay element, added to nf and nmo
// with a minus sign. Updates the stresses with velocities v (n - 1/2).
fn shell_element(e: i64, step_dt: f64) {
    let n0 = en[4 * e] as i64; let n1 = en[4 * e + 1] as i64;
    let n2 = en[4 * e + 2] as i64; let n3 = en[4 * e + 3] as i64;
    var ids: i64[4] = [n0, n1, n2, n3];
    // local frame from the diagonals
    let r31x = nx[3 * n2] - nx[3 * n0]; let r31y = nx[3 * n2 + 1] - nx[3 * n0 + 1]; let r31z = nx[3 * n2 + 2] - nx[3 * n0 + 2];
    let r42x = nx[3 * n3] - nx[3 * n1]; let r42y = nx[3 * n3 + 1] - nx[3 * n1 + 1]; let r42z = nx[3 * n3 + 2] - nx[3 * n1 + 2];
    var e3x = r31y * r42z - r31z * r42y; var e3y = r31z * r42x - r31x * r42z; var e3z = r31x * r42y - r31y * r42x;
    let l3 = sqrt(e3x * e3x + e3y * e3y + e3z * e3z);
    e3x /= l3; e3y /= l3; e3z /= l3;
    var e1x = nx[3 * n1] - nx[3 * n0]; var e1y = nx[3 * n1 + 1] - nx[3 * n0 + 1]; var e1z = nx[3 * n1 + 2] - nx[3 * n0 + 2];
    let d13 = e1x * e3x + e1y * e3y + e1z * e3z;
    e1x -= d13 * e3x; e1y -= d13 * e3y; e1z -= d13 * e3z;
    let l1 = sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
    e1x /= l1; e1y /= l1; e1z /= l1;
    let e2x = e3y * e1z - e3z * e1y; let e2y = e3z * e1x - e3x * e1z; let e2z = e3x * e1y - e3y * e1x;
    let cx = 0.25 * (nx[3 * n0] + nx[3 * n1] + nx[3 * n2] + nx[3 * n3]);
    let cy = 0.25 * (nx[3 * n0 + 1] + nx[3 * n1 + 1] + nx[3 * n2 + 1] + nx[3 * n3 + 1]);
    let cz = 0.25 * (nx[3 * n0 + 2] + nx[3 * n1 + 2] + nx[3 * n2 + 2] + nx[3 * n3 + 2]);
    var xl: f64[4] = [0.0; 4]; var yl: f64[4] = [0.0; 4];
    var vx: f64[4] = [0.0; 4]; var vy: f64[4] = [0.0; 4]; var vz: f64[4] = [0.0; 4];
    var tx: f64[4] = [0.0; 4]; var ty: f64[4] = [0.0; 4];
    for k in 0..4 {
        let n = ids[k];
        let ddx = nx[3 * n] - cx; let ddy = nx[3 * n + 1] - cy; let ddz = nx[3 * n + 2] - cz;
        xl[k] = ddx * e1x + ddy * e1y + ddz * e1z;
        yl[k] = ddx * e2x + ddy * e2y + ddz * e2z;
        let ux = nv[3 * n]; let uy = nv[3 * n + 1]; let uz = nv[3 * n + 2];
        vx[k] = ux * e1x + uy * e1y + uz * e1z;
        vy[k] = ux * e2x + uy * e2y + uz * e2z;
        vz[k] = ux * e3x + uy * e3y + uz * e3z;
        let wx = nw[3 * n]; let wy = nw[3 * n + 1]; let wz = nw[3 * n + 2];
        tx[k] = wx * e1x + wy * e1y + wz * e1z;
        ty[k] = wx * e2x + wy * e2y + wz * e2z;
    }
    let area = 0.5 * ((xl[2] - xl[0]) * (yl[3] - yl[1]) + (xl[1] - xl[3]) * (yl[2] - yl[0]));
    let inv2a = 0.5 / area;
    var b1: f64[4] = [(yl[1] - yl[3]) * inv2a, (yl[2] - yl[0]) * inv2a, (yl[3] - yl[1]) * inv2a, (yl[0] - yl[2]) * inv2a];
    var b2: f64[4] = [(xl[3] - xl[1]) * inv2a, (xl[0] - xl[2]) * inv2a, (xl[1] - xl[3]) * inv2a, (xl[2] - xl[0]) * inv2a];
    // membrane, bending and transverse shear rates
    var dxx = 0.0; var dyy = 0.0; var gxy = 0.0;
    var kxx = 0.0; var kyy = 0.0; var kxy = 0.0;
    var gxz = 0.0; var gyz = 0.0;
    for k in 0..4 {
        dxx += b1[k] * vx[k];
        dyy += b2[k] * vy[k];
        gxy += b2[k] * vx[k] + b1[k] * vy[k];
        kxx += b1[k] * ty[k];
        kyy -= b2[k] * tx[k];
        kxy += b2[k] * ty[k] - b1[k] * tx[k];
        gxz += b1[k] * vz[k] + 0.25 * ty[k];
        gyz += b2[k] * vz[k] - 0.25 * tx[k];
    }
    let t = eth[e];
    let mi = emat[e];
    let em = mat_e[mi]; let nu = mat_nu[mi];
    let c11 = em / (1.0 - nu * nu);
    let gm = 0.5 * em / (1.0 + nu);
    var nxx = 0.0; var nyy = 0.0; var nxy = 0.0;
    var mxx = 0.0; var myy = 0.0; var mxy = 0.0;
    var epm = 0.0;
    for g in 0..NIP {
        let z = gp_z[g] * 0.5 * t;
        let w = gp_w[g] * 0.5 * t;
        let ex = (dxx + z * kxx) * step_dt;
        let ey = (dyy + z * kyy) * step_dt;
        let exy = (gxy + z * kxy) * step_dt;
        let s = 15 * e + 3 * g;
        let trx = esig[s] + c11 * (ex + nu * ey);
        let tr_y = esig[s + 1] + c11 * (ey + nu * ex);
        let trxy = esig[s + 2] + gm * exy;
        let sx, sy, sxy, dep = plane_stress_return(trx, tr_y, trxy, eep[5 * e + g], em, nu, mat_sy[mi], mat_h[mi]);
        esig[s] = sx; esig[s + 1] = sy; esig[s + 2] = sxy;
        eep[5 * e + g] += dep;
        epm = max(epm, eep[5 * e + g]);
        nxx += w * sx; nyy += w * sy; nxy += w * sxy;
        mxx += w * z * sx; myy += w * z * sy; mxy += w * z * sxy;
    }
    eepmax[e] = epm;
    etau[2 * e] += KAPPA_S * gm * gxz * step_dt;
    etau[2 * e + 1] += KAPPA_S * gm * gyz * step_dt;
    let qx = t * etau[2 * e];
    let qy = t * etau[2 * e + 1];
    // internal work, for the energy balance
    // hourglass control (Belytschko-Tsay), stiffness form
    let hx = xl[0] - xl[1] + xl[2] - xl[3];
    let hy = yl[0] - yl[1] + yl[2] - yl[3];
    var gam: f64[4] = [0.0; 4];
    var bb = 0.0;
    for k in 0..4 {
        let hk = if k % 2 == 0 { 1.0 } else { -1.0 };
        gam[k] = hk - hx * b1[k] - hy * b2[k];
        bb += b1[k] * b1[k] + b2[k] * b2[k];
    }
    var qmx = 0.0; var qmy = 0.0; var qw = 0.0; var qtx = 0.0; var qty = 0.0;
    for k in 0..4 {
        qmx += gam[k] * vx[k]; qmy += gam[k] * vy[k]; qw += gam[k] * vz[k];
        qtx += gam[k] * tx[k]; qty += gam[k] * ty[k];
    }
    let cm = HG_R * em * t * area * bb / 8.0;
    let cth = HG_R * em * t * t * t * area * bb / 192.0;
    let cw = HG_R * KAPPA_S * gm * t * t * t * bb / 12.0;
    let h5 = 5 * e;
    ehg[h5] += cm * qmx * step_dt;
    ehg[h5 + 1] += cm * qmy * step_dt;
    ehg[h5 + 2] += cw * qw * step_dt;
    ehg[h5 + 3] += cth * qtx * step_dt;
    ehg[h5 + 4] += cth * qty * step_dt;
    hg_work += step_dt * (ehg[h5] * qmx + ehg[h5 + 1] * qmy + ehg[h5 + 2] * qw + ehg[h5 + 3] * qtx + ehg[h5 + 4] * qty);
    // local nodal forces and moments, then back to global
    for k in 0..4 {
        let fx = area * (b1[k] * nxx + b2[k] * nxy) + gam[k] * ehg[h5];
        let fy = area * (b2[k] * nyy + b1[k] * nxy) + gam[k] * ehg[h5 + 1];
        let fz = area * (b1[k] * qx + b2[k] * qy) + gam[k] * ehg[h5 + 2];
        let mx = area * (-b2[k] * myy - b1[k] * mxy - 0.25 * qy) + gam[k] * ehg[h5 + 3];
        let my = area * (b1[k] * mxx + b2[k] * mxy + 0.25 * qx) + gam[k] * ehg[h5 + 4];
        let n = ids[k];
        nf[3 * n] -= fx * e1x + fy * e2x + fz * e3x;
        nf[3 * n + 1] -= fx * e1y + fy * e2y + fz * e3y;
        nf[3 * n + 2] -= fx * e1z + fy * e2z + fz * e3z;
        nmo[3 * n] -= mx * e1x + my * e2x;
        nmo[3 * n + 1] -= mx * e1y + my * e2y;
        nmo[3 * n + 2] -= mx * e1z + my * e2z;
    }
    if epm >= mat_epf[mi] {
        efail[e] = 1;
        n_failed += 1;
        if mi == 0 { n_failed_skin += 1; }
    }
}

fn shell_forces(step_dt: f64) {
    for i in 0..3 * nn { nf[i] = 0.0; nmo[i] = 0.0; }
    for e in 0..ne {
        if efail[e] == 0 { shell_element(e, step_dt); }
    }
}

// Stable time step of the shells: the smallest element over the plate
// wave speed (area over the longest side as the element length).
fn shell_dt() -> f64 {
    var best = 1.0;
    for e in 0..ne {
        if efail[e] != 0 { continue; }
        let a = en[4 * e]; let b = en[4 * e + 1]; let c = en[4 * e + 2]; let d = en[4 * e + 3];
        let area = quad_area(a, b, c, d);
        let lmax = max(max(dist(a, b), dist(b, c)), max(dist(c, d), dist(d, a)));
        let mi = emat[e];
        let cp = sqrt(mat_e[mi] / (mat_rho[mi] * (1.0 - mat_nu[mi] * mat_nu[mi])));
        best = min(best, area / lmax / cp);
    }
    return CFL_SHELL * best;
}

fn dist(a: i32, b: i32) -> f64 {
    let dx = nx[3 * a] - nx[3 * b]; let dy = nx[3 * a + 1] - nx[3 * b + 1]; let dz = nx[3 * a + 2] - nx[3 * b + 2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

// ---------------------------------------------------------------------------
// contact

// Closest point of triangle abc to p (Ericson, Real-Time Collision
// Detection, 5.1.5). Returns the barycentric weights of a, b and c.
fn closest_tri(p: f64[3], a: f64[3], b: f64[3], c: f64[3]) -> f64, f64, f64 {
    let ab = b - a; let ac = c - a; let ap = p - a;
    let d1 = dot3(ab, ap); let d2 = dot3(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 { return 1.0, 0.0, 0.0; }
    let bp = p - b;
    let d3 = dot3(ab, bp); let d4 = dot3(ac, bp);
    if d3 >= 0.0 && d4 <= d3 { return 0.0, 1.0, 0.0; }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return 1.0 - v, v, 0.0;
    }
    let cp = p - c;
    let d5 = dot3(ab, cp); let d6 = dot3(ac, cp);
    if d6 >= 0.0 && d5 <= d6 { return 0.0, 0.0, 1.0; }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return 1.0 - w, 0.0, w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return 0.0, 1.0 - w, w;
    }
    let den = 1.0 / (va + vb + vc);
    let v = vb * den;
    let w = vc * den;
    return 1.0 - v - w, v, w;
}

fn dot3(a: f64[3], b: f64[3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }

fn node3(n: i64) -> f64[3] { [nx[3 * n], nx[3 * n + 1], nx[3 * n + 2]] }

fn bin_elements() {
    for b in 0..NHASH { ehead[b] = -1; }
    var rmax = 0.0;
    for e in 0..ne {
        let a = en[4 * e] as i64; let b = en[4 * e + 1] as i64; let c = en[4 * e + 2] as i64; let d = en[4 * e + 3] as i64;
        let ctr = (node3(a) + node3(b) + node3(c) + node3(d)) * 0.25;
        ecen[3 * e] = ctr[0]; ecen[3 * e + 1] = ctr[1]; ecen[3 * e + 2] = ctr[2];
        var r = 0.0;
        for k in 0..4 {
            let q = node3(en[4 * e + k]) - ctr;
            r = max(r, sqrt(dot3(q, q)));
        }
        erad[e] = r;
        rmax = max(rmax, r);
    }
    // a particle in contact lies within erad + gap of the centroid
    ecs = rmax + prad + 0.5 * T_FRM + 1.0e-4;
    for e in 0..ne {
        if efail[e] != 0 { continue; }
        let ci = cell_of(ecen[3 * e], ecs);
        let cj = cell_of(ecen[3 * e + 1], ecs);
        let ck = cell_of(ecen[3 * e + 2], ecs);
        ecell[3 * e] = ci; ecell[3 * e + 1] = cj; ecell[3 * e + 2] = ck;
        let b = hash3(ci, cj, ck);
        enext[e] = ehead[b];
        ehead[b] = e as i32;
    }
}

// Penalty contact: forces on the particles and on the shell nodes.
fn contact(step_dt: f64) {
    contact_fz = 0.0;
    center_fz = 0.0;
    for i in 0..3 * np { pfc[i] = 0.0; }
    for i in 0..3 * nn { nfc[i] = 0.0; }
    if mode == 1 {
        let k = K_CONTACT * pm / (step_dt * step_dt);
        for i in 0..np {
            let pen = prad - px[3 * i + 2];
            if pen > 0.0 {
                let f = k * pen;
                pfc[3 * i + 2] = f;
                contact_fz += f;
                let rr = px[3 * i] * px[3 * i] + px[3 * i + 1] * px[3 * i + 1];
                if rr < sensor_r * sensor_r { center_fz += f; }
            }
        }
        return;
    }
    bin_elements();
    for i in 0..np {
        let p: f64[3] = [px[3 * i], px[3 * i + 1], px[3 * i + 2]];
        let ci0 = cell_of(p[0], ecs); let cj0 = cell_of(p[1], ecs); let ck0 = cell_of(p[2], ecs);
        var best = -1;
        var bestd = 1.0e30;
        var bw: f64[4] = [0.0; 4];
        var bq: f64[3] = [0.0; 3];
        for dk in -1..2 {
            for dj in -1..2 {
                for di in -1..2 {
                    let ci1 = ci0 + di; let cj1 = cj0 + dj; let ck1 = ck0 + dk;
                    var e = ehead[hash3(ci1, cj1, ck1)] as i64;
                    while e >= 0 {
                        if ecell[3 * e] == ci1 && ecell[3 * e + 1] == cj1 && ecell[3 * e + 2] == ck1 {
                            let gap = prad + 0.5 * eth[e];
                            let dc: f64[3] = [p[0] - ecen[3 * e], p[1] - ecen[3 * e + 1], p[2] - ecen[3 * e + 2]];
                            let lim = erad[e] + gap;
                            if dot3(dc, dc) < lim * lim {
                                let a = node3(en[4 * e]); let b = node3(en[4 * e + 1]);
                                let c = node3(en[4 * e + 2]); let d = node3(en[4 * e + 3]);
                                // the quad as triangles abc and acd
                                let u1, v1, w1 = closest_tri(p, a, b, c);
                                let q1 = a * u1 + b * v1 + c * w1;
                                let u2, v2, w2 = closest_tri(p, a, c, d);
                                let q2 = a * u2 + c * v2 + d * w2;
                                let r1 = p - q1; let r2 = p - q2;
                                let s1 = dot3(r1, r1); let s2 = dot3(r2, r2);
                                let sd = min(s1, s2);
                                if sd < gap * gap && sqrt(sd) - gap < bestd {
                                    bestd = sqrt(sd) - gap;
                                    best = e;
                                    if s1 <= s2 { bw = [u1, v1, w1, 0.0]; bq = q1; }
                                    else { bw = [u2, 0.0, v2, w2]; bq = q2; }
                                }
                            }
                        }
                        e = enext[e];
                    }
                }
            }
        }
        if best >= 0 {
            let pen = -bestd;
            var nrm = p - bq;
            let ln = sqrt(dot3(nrm, nrm));
            if ln < 1.0e-12 { continue; }
            nrm = nrm * (1.0 / ln);
            // the reduced mass of the particle and the nodes it pushes
            var mn = 0.0;
            for k in 0..4 { mn += bw[k] * nmass[en[4 * best + k]]; }
            let mr = pm * mn / (pm + mn);
            let kk = K_CONTACT * mr / (step_dt * step_dt);
            let f = kk * pen;
            pfc[3 * i] = f * nrm[0]; pfc[3 * i + 1] = f * nrm[1]; pfc[3 * i + 2] = f * nrm[2];
            contact_fz += f * nrm[2];
            let rr = p[0] * p[0] + p[1] * p[1];
            if rr < sensor_r * sensor_r { center_fz += f * nrm[2]; }
            for k in 0..4 {
                let n = en[4 * best + k] as i64;
                nfc[3 * n] -= bw[k] * f * nrm[0];
                nfc[3 * n + 1] -= bw[k] * f * nrm[1];
                nfc[3 * n + 2] -= bw[k] * f * nrm[2];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the time step

fn stable_dt() -> f64 {
    var vmax = 0.0;
    for i in 0..np {
        let v2 = pv[3 * i] * pv[3 * i] + pv[3 * i + 1] * pv[3 * i + 1] + pv[3 * i + 2] * pv[3 * i + 2];
        vmax = max(vmax, pc[i] + sqrt(v2));
    }
    var d = CFL_SPH * ph / vmax;
    if mode == 0 { d = min(d, shell_dt()); }
    return d;
}

fn step_once() {
    dt = stable_dt();
    // central differences: forces at x(n) from v(n - 1/2)
    let dtm = if nsteps == 0 { dt } else { 0.5 * (dt + dt_prev) };
    sph_forces();
    if mode == 0 { shell_forces(dtm); }
    contact(dt);
    if contact_fz > 0.0 && t_contact < 0.0 { t_contact = sim_time; }
    impulse += contact_fz * dt;
    // particles
    let ipm = 1.0 / pm;
    for i in 0..np {
        for k in 0..3 {
            let a = pa[3 * i + k] + pfc[3 * i + k] * ipm;
            let vm = pv[3 * i + k] + 0.5 * a * dtm;
            bird_work -= pm * pa[3 * i + k] * vm * dtm;
            cw_bird += pfc[3 * i + k] * vm * dtm;
            pv[3 * i + k] += a * dtm;
            px[3 * i + k] += pv[3 * i + k] * dt;
        }
        prho[i] += pdrho[i] * dt;
        pp[i] = tait(prho[i]);
        let x = prho[i] / RHO_BIRD;
        pc[i] = C0_BIRD * x * x * x;
    }
    // nodes
    if mode == 0 {
        var defl = 0.0;
        for n in 0..nn {
            if nfix[n] != 0 { continue; }
            let im = 1.0 / nmass[n];
            let ii = 1.0 / ninert[n];
            for k in 0..3 {
                let a = (nf[3 * n + k] + nfc[3 * n + k]) * im;
                let vm = nv[3 * n + k] + 0.5 * a * dtm;
                let al = nmo[3 * n + k] * ii;
                let wm = nw[3 * n + k] + 0.5 * al * dtm;
                panel_work -= (nf[3 * n + k] * vm + nmo[3 * n + k] * wm) * dtm;
                cw_panel += nfc[3 * n + k] * vm * dtm;
                nv[3 * n + k] += a * dtm;
                nw[3 * n + k] += al * dtm;
                nx[3 * n + k] += nv[3 * n + k] * dt;
            }
            if nskin[n] != 0 {
                let ux = nx[3 * n] - nx0[3 * n]; let uy = nx[3 * n + 1] - nx0[3 * n + 1]; let uz = nx[3 * n + 2] - nx0[3 * n + 2];
                defl = max(defl, sqrt(ux * ux + uy * uy + uz * uz));
            }
        }
        max_defl = max(max_defl, defl);
        var epx = 0.0;
        for e in 0..ne { epx = max(epx, eepmax[e]); }
        max_ep = epx;
    }
    dt_prev = dt;
    sim_time += dt;
    nsteps += 1;
}

// Advances by up to `steps` steps, stopping at `t_end` (s).
export fn advance(steps: i32, t_end: f64) -> i32 {
    var done: i32 = 0;
    while done < steps && sim_time < t_end {
        step_once();
        done += 1;
    }
    return done;
}

// ---------------------------------------------------------------------------
// what the page and the validation read

export fn get_time() -> f64 { sim_time }
export fn get_dt() -> f64 { dt }
export fn get_steps() -> i32 { nsteps as i32 }
export fn get_t_contact() -> f64 { t_contact }
export fn get_np() -> i32 { np as i32 }
export fn get_nn() -> i32 { nn as i32 }
export fn get_ne() -> i32 { ne as i32 }
export fn get_particle_mass() -> f64 { pm }
export fn get_spacing() -> f64 { pdx }
export fn get_smoothing() -> f64 { ph }
export fn get_bird_d() -> f64 { bird_d }
export fn get_bird_l() -> f64 { bird_l }
export fn get_bird_v() -> f64 { bird_v }
export fn get_contact_force() -> f64 { contact_fz }
export fn get_center_force() -> f64 { center_fz }
export fn get_sensor_r() -> f64 { sensor_r }
export fn get_impulse() -> f64 { impulse }
export fn get_panel_work() -> f64 { panel_work }
export fn get_hg_work() -> f64 { hg_work }
export fn get_contact_work_bird() -> f64 { cw_bird }
export fn get_contact_work_panel() -> f64 { cw_panel }
export fn get_bird_work() -> f64 { bird_work }
export fn get_failed() -> i32 { n_failed as i32 }
export fn get_failed_skin() -> i32 { n_failed_skin as i32 }
export fn get_max_ep() -> f64 { max_ep }
export fn get_max_defl() -> f64 { max_defl }
export fn get_eps_fail(m: i32) -> f64 { mat_epf[m] }

export fn get_bird_ke() -> f64 {
    var ke = 0.0;
    for i in 0..np {
        ke += 0.5 * pm * (pv[3 * i] * pv[3 * i] + pv[3 * i + 1] * pv[3 * i + 1] + pv[3 * i + 2] * pv[3 * i + 2]);
    }
    return ke;
}

export fn get_bird_pz() -> f64 {
    var s = 0.0;
    for i in 0..np { s += pm * pv[3 * i + 2]; }
    return s;
}

export fn get_panel_ke() -> f64 {
    var ke = 0.0;
    for n in 0..nn {
        ke += 0.5 * nmass[n] * (nv[3 * n] * nv[3 * n] + nv[3 * n + 1] * nv[3 * n + 1] + nv[3 * n + 2] * nv[3 * n + 2]);
        ke += 0.5 * ninert[n] * (nw[3 * n] * nw[3 * n] + nw[3 * n + 1] * nw[3 * n + 1] + nw[3 * n + 2] * nw[3 * n + 2]);
    }
    return ke;
}

export fn get_panel_mass() -> f64 {
    var m = 0.0;
    for n in 0..nn { m += nmass[n]; }
    return m;
}

// Fills the f32 render buffers from the state.
export fn fill_render() -> i32 {
    for i in 0..3 * nn { r_node[i] = nx[i] as f32; }
    for i in 0..3 * np { r_part[i] = px[i] as f32; }
    for i in 0..np { r_pp[i] = pp[i] as f32; }
    for e in 0..ne { r_eep[e] = if efail[e] != 0 { -1.0 } else { eepmax[e] as f32 }; }
    return 1;
}

export fn buf_nodes() -> const u8[:] { bytes_of(r_node) }
export fn buf_particles() -> const u8[:] { bytes_of(r_part) }
export fn buf_pressure() -> const u8[:] { bytes_of(r_pp) }
export fn buf_strain() -> const u8[:] { bytes_of(r_eep) }
export fn buf_elements() -> const u8[:] { bytes_of(en) }
