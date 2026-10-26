// Checks of the solver. The default: a rigid flat plate check of the bird
// model against Wilbeck's hydrodynamic theory, the Hugoniot pressure at
// first contact and the stagnation pressure of the steady flow, both at
// the impact centre. --panel: the fuselage panel case with its energy
// balance and damage.
//
// Usage: validate [--panel] [--csv] [spacing_um ...]
// Prints a summary per bird spacing (default 9000 6000 4500 um) and, with
// --csv, the history. --panel runs the fuselage panel case instead.

import strike;
import os;
import math;
import std;

// Shock state of the Tait gelatin hitting a rigid wall at speed v: the
// particle velocity behind the shock is v, and the jump conditions
// rho0 Us = rho1 (Us - v), p1 = rho0 Us v give p1 = p(rho1).
fn hugoniot(v: f64) -> f64, f64 {
    let b = RHO_BIRD * C0_BIRD * C0_BIRD / GAMMA_TAIT;
    var lo = RHO_BIRD * 1.000001;
    var hi = RHO_BIRD * 2.0;
    for it in 0..200 {
        let mid = 0.5 * (lo + hi);
        let f = b * (pow(mid / RHO_BIRD, GAMMA_TAIT) - 1.0) - RHO_BIRD * v * v * mid / (mid - RHO_BIRD);
        if f > 0.0 { hi = mid; } else { lo = mid; }
    }
    let rho1 = 0.5 * (lo + hi);
    let us = v * rho1 / (rho1 - RHO_BIRD);
    return RHO_BIRD * us * v, us;
}

fn run(spacing: f64, csv: bool) {
    let v = V_C;
    let n = setup(1, 8, spacing, v);
    let area = PI * get_sensor_r() * get_sensor_r();
    let tl = get_bird_l() / v;         // the squash-up time L / V
    let ph, us = hugoniot(v);
    let ps = 0.5 * RHO_BIRD * v * v;
    var peak = 0.0;
    var t_peak = 0.0;
    var plat = 0.0;
    var plat_n = 0;
    let t_end = 1.5 * tl;
    var last_print = -1.0;
    while get_time() < t_end {
        advance(1, t_end);
        let tc = get_t_contact();
        if tc < 0.0 { continue; }
        let t = get_time() - tc;
        let p = get_center_force() / area;
        if p > peak { peak = p; t_peak = t; }
        // steady flow: between 1/4 and 3/4 of the squash-up time
        if t > 0.25 * tl && t < 0.75 * tl { plat += p; plat_n += 1; }
        if csv && t - last_print >= 2.0e-6 {
            print("csv,", spacing * 1000.0, ",", t * 1.0e6, ",", p / 1.0e6, ",", get_contact_force() / 1000.0);
            last_print = t;
        }
    }
    let pm = plat / plat_n;
    let mom = BIRD_MASS * v;
    print("spacing_mm=", spacing * 1000.0, " particles=", n, " h_mm=", get_smoothing() * 1000.0,
          " steps=", get_steps(), " dt_ns=", get_dt() * 1.0e9,
          " sensor_r_mm=", get_sensor_r() * 1000.0);
    print("  hugoniot_MPa=", ph / 1.0e6, " (Us=", us, " m/s) peak_MPa=", peak / 1.0e6,
          " at_us=", t_peak * 1.0e6, " error_pct=", 100.0 * (peak - ph) / ph);
    print("  stagnation_MPa=", ps / 1.0e6, " plateau_MPa=", pm / 1.0e6,
          " error_pct=", 100.0 * (pm - ps) / ps);
    print("  impulse/(m v)=", get_impulse() / mom, " squash_up_us=", tl * 1.0e6,
          " energy_balance=", (get_bird_ke() + get_bird_work()) / (0.5 * BIRD_MASS * v * v));
}

// The panel case: energy balance and damage. Prints the history with --csv.
fn panel_run(spacing: f64, csv: bool) {
    // shell elements about 2.1 bird spacings long, a multiple of 8 along x
    let cells = ((PANEL_LX / (2.1 * spacing) / 8.0 + 0.5) as i32) * 8;
    let n = setup(0, cells, spacing, V_C);
    let ke0 = get_bird_ke();
    let t_end = 3.0e-3;
    var next = 0.0;
    var fpeak = 0.0;
    var worst = 0.0;
    while get_time() < t_end {
        advance(1, t_end);
        fpeak = max(fpeak, get_contact_force());
        let bal = (get_bird_ke() + get_panel_ke() + get_panel_work() + get_bird_work()) / ke0;
        worst = max(worst, abs(bal - 1.0));
        if csv && get_time() >= next {
            print("csv,", spacing * 1000.0, ",", get_time() * 1.0e6, ",", get_contact_force() / 1000.0, ",",
                  get_max_defl() * 1000.0, ",", get_max_ep(), ",", get_failed(), ",", get_bird_ke(), ",",
                  get_panel_ke(), ",", get_panel_work(), ",", get_bird_work());
            next += 2.0e-5;
        }
    }
    print("panel spacing_mm=", spacing * 1000.0, " skin_cells=", cells, " nodes=", get_nn(),
          " shells=", get_ne(), " particles=", n, " steps=", get_steps(), " panel_kg=", get_panel_mass());
    print("  peak_force_kN=", fpeak / 1000.0, " max_deflection_mm=", get_max_defl() * 1000.0,
          " max_plastic_strain=", get_max_ep(), " eroded_shells=", get_failed(), " eroded_skin=", get_failed_skin());
    print("  bird_KE0_J=", ke0, " panel_work_J=", get_panel_work(), " hourglass_J=", get_hg_work(),
          " bird_internal_J=", get_bird_work(), " bird_KE_end_J=", get_bird_ke(),
          " worst_energy_balance_error=", worst);
}

fn main() {
    var csv = false;
    var panel = false;
    var spacings: f64[>..] = [];
    let av = args();
    for i in 1..av.len {
        if av[i] == "--csv" { csv = true; continue; }
        if av[i] == "--panel" { panel = true; continue; }
        let x, ok = parse_int(av[i]);
        assert(ok);
        spacings.push(x / 1.0e6);
    }
    if panel {
        if spacings.len == 0 { spacings.push(0.009); }
        for sp in spacings { panel_run(sp, csv); }
        return;
    }
    if spacings.len == 0 { spacings.push(0.009); spacings.push(0.006); spacings.push(0.0045); }
    for s in spacings { run(s, csv); }
}
