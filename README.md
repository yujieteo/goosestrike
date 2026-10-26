# goosestrike

A 4 lb bird hits a stiffened aluminium fuselage panel at V<sub>C</sub>, the load case of 14 CFR 25.571(e)(1).
The solver is explicit dynamics, written in [Goose](https://github.com/aardappel/goose), compiled through C to WebAssembly.
It runs in a browser page that shows the bird as a cartoon goose, shows the Goose code that runs, and exports the strike as an animated GIF.

![The goose hits the panel](docs/strike.png)

Run it:

```sh
python3 -m http.server -d web 8000
# then open http://localhost:8000/
```

The repository has the built `web/strike.wasm`, so the page needs no build step.
This is not a certification analysis. The page and this README say where the model is coarse.

## The load case

14 CFR 25.571(e)(1), as amended to Amendment 25-148 ([text at law.cornell.edu](https://www.law.cornell.edu/cfr/text/14/25.571)):

> (e) Damage-tolerance (discrete source) evaluation. The airplane must be capable of successfully completing a flight during which likely structural damage occurs as a result of—
> (1) Impact with a 4-pound bird when the velocity of the airplane relative to the bird along the airplane's flight path is equal to V<sub>C</sub> at sea level or 0.85V<sub>C</sub> at 8,000 feet, whichever is more critical;

| Quantity | Value | Source |
|---|---|---|
| Bird mass | 4 lb = 1.81436948 kg | the rule, and 1 lb = 0.45359237 kg |
| V<sub>C</sub> | 350 KCAS | 14 CFR 25.1505: V<sub>MO</sub> must not be greater than V<sub>C</sub>. The A320 has V<sub>MO</sub> = 350 kt ([EASA TCDS EASA.A.064](https://www.easa.europa.eu/en/document-library/type-certificates/aircraft-cs-25-cs-22-cs-23-cs-vla-cs-lsa/easaa064-airbus-a318-a319-a320), Issue 62, Section 1 A320). Thus 350 KCAS is the lowest V<sub>C</sub> that the rule allows for this aeroplane. |
| Impact speed | 180.06 m/s | 350 KCAS = 350 KTAS at sea level ISA. 1 kt = 1852/3600 m/s. |
| Altitude | sea level | 0.85 V<sub>C</sub> at 8000 ft is about 0.96 V<sub>C</sub> true airspeed, so the sea-level case is more critical. |
| Direction | normal to the skin, at the middle of the panel | an idealized worst case |

## The bird: physics and costume

The physics bird is the standard substitute bird of bird strike analysis.
The cartoon goose is only paint on it.

| Physics | Value | Source |
|---|---|---|
| Shape | cylinder with hemispherical ends, length/diameter = 2 | Heimbs 2011, Perdikoulis et al. 2021 |
| Size | diameter 113.1 mm, length 226.2 mm | from the mass and the density |
| Material | gelatin with 10 % porosity, 950 kg/m³ | Perdikoulis et al. 2021, after Barber et al. |
| Equation of state | Tait for water: p = (ρ<sub>0</sub>c<sub>0</sub>²/γ)((ρ/ρ<sub>0</sub>)<sup>γ</sup> − 1), c<sub>0</sub> = 1480 m/s, γ = 7, no tension (p ≥ 0) | Heimbs 2011 |

The costume (white feathers, eyes, beak, wings, feet, head tuft and blush) follows the particles, but it has no mass and no effect on the result.
The flight before contact is a cartoon. The simulation starts with the nose 2 mm plus half a particle spacing from the skin.

## The panel

The geometry is a representative assumption, not a drawing of a real aeroplane.

| Part | Value |
|---|---|
| Fuselage radius | 1.975 m (A320 class) |
| Panel | 1.0668 m along the fuselage (two 533 mm frame bays) × 0.760 m around it (four 190 mm stringer bays) |
| Edges | clamped, on the next frames and stringers |
| Skin | 2024-T3, 1.6 mm |
| Stringers | 4 Z sections, 7075-T6, web 25.4 mm, flange 15 mm, 1.6 mm |
| Frames | 2 Z sections, 7075-T6, web 76.2 mm, flange 25 mm, 1.6 mm |

Material data, from ASM Aerospace Specification Metals (data sheets for 2024-T3 and 7075-T6 sheet, as published on MatWeb):

| | ρ (kg/m³) | E (GPa) | ν | yield (MPa) | ultimate (MPa) | elongation |
|---|---|---|---|---|---|---|
| 2024-T3 | 2780 | 73.1 | 0.33 | 345 | 483 | 18 % |
| 7075-T6 | 2810 | 71.7 | 0.33 | 503 | 572 | 11 % |

The plasticity is bilinear: a straight line from yield to the ultimate stress at the elongation.
An element erodes when its plastic strain reaches the elongation less the elastic strain at the ultimate stress: 17.3 % for 2024-T3 and 10.2 % for 7075-T6.

## The method

All of it is in `sim/strike.goose`. Units are SI.

- Bird: smoothed particle hydrodynamics (SPH). Cubic spline kernel, h = 1.2 × particle spacing, continuity equation for the density, Monaghan artificial viscosity (α = 1, β = 2).
- Panel: Belytschko-Lin-Tsay 4-node shells with 5 Gauss points through the thickness. J2 plasticity with linear hardening and the plane-stress return of Simo and Taylor. Stiffness hourglass control. Lumped masses.
- Contact: penalty. Each particle pushes on its nearest shell element, with a stiffness of 0.1 × reduced mass / dt².
- Time: central differences. The stable time step comes from the smallest shell element and from the SPH sound speed (CFL 0.25 for SPH, 0.9 for the shells).
- Memory: all state is fixed-size global arrays in static data. The time loop allocates nothing.

The energy balance closes: bird kinetic energy + bird internal energy + panel kinetic energy + panel internal work stays within 0.4 % of the initial kinetic energy in all runs below.

## Accuracy: the rigid wall check

Wilbeck (1978) gives two pressures at the centre of a soft body that hits a rigid wall:

- the Hugoniot shock pressure at first contact, P<sub>H</sub> = ρ<sub>0</sub> U<sub>s</sub> v. Here U<sub>s</sub> comes from the jump conditions with the same Tait equation of state.
- the stagnation pressure of the steady flow after the shock, P<sub>s</sub> = ½ ρ<sub>0</sub> v².

At 180.06 m/s, P<sub>H</sub> = 314.4 MPa (U<sub>s</sub> = 1838 m/s) and P<sub>s</sub> = 15.40 MPa.
The solver measures the force on a disc of radius D/8 (14.2 mm) at the impact centre.
"Peak" is the maximum. "Plateau" is the mean from 0.25 to 0.75 of the squash-up time L/v = 1260 µs.

| Bird spacing | Particles | Peak (MPa) | Peak vs P<sub>H</sub> | Plateau (MPa) | Plateau vs P<sub>s</sub> | Impulse / m v | Energy balance |
|---|---|---|---|---|---|---|---|
| 9 mm | 2560 | 180.5 | −42.6 % | 19.37 | +25.8 % | 1.0079 | 0.9997 |
| 6 mm | 8760 | 317.4 | +0.9 % | 19.67 | +27.7 % | 0.9955 | 1.0006 |
| 4.5 mm | 20902 | 322.9 | +2.7 % | 17.42 | +13.1 % | 0.9935 | 1.0000 |
| 3.5 mm | 44412 | 350.1 | +11.3 % | 15.90 | +3.2 % | 0.9921 | 1.0000 |

The impulse is m v within 1 %, and the energy balance closes within 0.1 %.
The plateau converges to P<sub>s</sub> as the spacing decreases, to +3.2 % at 3.5 mm.
The peak converges less well. At 9 mm, the smoothing length (10.9 mm) smears the shock, and the peak is 43 % low.
At 6 mm and 4.5 mm the peak is within 3 %, and at 3.5 mm it is 11 % high.
The peak is the maximum of one time step on a 14 mm disc, so it is noisy. Use it as an order-of-magnitude check, and the plateau as the convergence check.
These numbers come from `validate`, the native build of the same solver (see below). The page runs the same check in the browser with the "Run the wall check" button.

## The panel result

Impact at 180.06 m/s, run to 3 ms. Bird kinetic energy at the start: 29.41 kJ.

| Bird spacing | Skin cells along the fuselage | Shells | Peak contact force | Max deflection | Eroded skin shells | Eroded stiffener shells | Panel work | Hourglass part | Worst energy balance error |
|---|---|---|---|---|---|---|---|---|---|
| 9 mm | 56 | 3088 | 240.1 kN | 150.8 mm | 0 | 36 | 18.53 kJ | 0.34 kJ | 0.32 % |
| 6 mm | 88 | 7712 | 229.3 kN | 152.1 mm | 8 | 62 | 18.92 kJ | 0.25 kJ | 0.20 % |
| 4.5 mm | 112 | 12960 | 209.1 kN | 152.9 mm | 25 | 152 | 19.29 kJ | 0.27 kJ | 0.11 % |

The shell size is about 2.1 bird spacings. The panel mass is 5.05 kg.

The deflection converges.
The skin damage does not: at 9 mm, no skin element reaches the failure strain, but at 6 mm and 4.5 mm some do.
Element erosion depends on the element size, so a tear prediction needs a failure model calibrated to the mesh. This model has none.
The stringers and frames fail locally at all three meshes.
The rule accepts damage. It asks that the aeroplane completes the flight with it, and that needs a residual strength analysis, which this project does not do.

## The browser trade

The page offers the three meshes. The solver time per step grows with the particle count, and the stable time step decreases with the element size.
The times below are for the wasm in Node 24 on one core of the development machine. The native build is about 1.6 times faster.

| Page preset | Bird spacing | Steps to 3 ms | wasm time per step | Solver time for the run |
|---|---|---|---|---|
| Potato goose | 9 mm | 1774 | 5.7 ms | about 10 s |
| Good goose | 6 mm | 2679 | 13.1 ms | about 35 s |
| Very good goose | 4.5 mm | 3594 | 31.3 ms | about 2 minutes |

The 9 mm mesh is the fast default. The trade: its peak pressure is 43 % low, its plateau is 26 % high, and it shows no skin erosion. But its deflection is within 1.5 % of the 4.5 mm result, and its peak force is 15 % higher.

## The page

- The code view shows `web/strike.goose`, the copy of `sim/strike.goose` that `build.sh` writes.
  The page runs each time step in five phases, one call of the wasm export `step_phase` each: time step, SPH bird, shells (with plasticity and erosion), contact and integration.
  `run_phase` in the source has the five phases. The lit lines are the phase that ran last, as `step_phase` returns it, and the page measures the time in each phase.
  A test checks that the phases one at a time give the same result as whole steps.
- "Export GIF" saves the last run as an animated GIF, 480 px wide: about ten frames a second of the approach, then one frame each 25 µs of simulated time.
  Each frame has a caption with the time and the colour bar. The encoder is `web/gif.mjs` (median cut palette, LZW), with no dependency.
- The colour maps are continuous ramps, interpolated in OKLab: plastic strain on a square-root scale from the bare metal (0 %) to 18 %, so that the first yield shows, and bird pressure on a log scale from 0.1 MPa to 400 MPa.
- The look (fonts, palette) is that of [teoyujie.org](https://teoyujie.org). The fonts are in `web/fonts/`: Source Serif 4 and DM Sans, with their OFL licences.

## Goose on WebAssembly

Goose has no wasm port yet (its `docs/design/wasm.md` is a proposal).
The C that Goose writes compiles with Emscripten, with one risk: Goose reserves a large address range (2 GiB by default) for each data stack, with a guard gap behind it that stops a stack overflow.
WebAssembly linear memory has no reservation and no guard pages. Emscripten's `mmap` allocates the full size, and `mprotect` does nothing.

The handling:

1. The solver keeps all state in static global arrays. The generated C of the time loop (`step_once`, `step_phase` and the functions they call) has no data-stack use. You can check this: in the code of the program, `gen/strike.c` uses `->top` only in `gs_init_globals`.
2. `gs_init_globals` builds each zeroed global array in a data-stack temporary and copies it. The largest temporary is `esig`, 24000 × 15 × 8 B = 2.88 MB, and each temporary is released before the next one.
3. Thus `build.sh` sets `GS_STACK_RESERVE` to 4 MiB and the guard gap to 64 KiB. If a later change makes a larger global array, raise the reserve.

## Build

The build needs the Goose compiler and Emscripten. On NixOS, these commands work (no compiler needs to be on the PATH):

```sh
# 1. the Goose compiler, at the commit that this repository uses
git clone https://github.com/aardappel/goose ../goose
git -C ../goose checkout a108677d300cb0e8bbb842c35a353abe67a20c6c
( cd ../goose && nix shell nixpkgs#cmake nixpkgs#clang nixpkgs#lld nixpkgs#gnumake --command \
    bash -c 'cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ && cmake --build build -j16' )

# 2. the C (gen/), the wasm (web/strike.mjs, web/strike.wasm) and web/strike.goose, with Emscripten 5.0.6
nix shell nixpkgs#emscripten --command env GOOSE=$PWD/../goose/build/goose ./build.sh

# 3. the native validation program
( cd sim && nix shell nixpkgs#clang --command ../../goose/build/goose build -o ../build/validate validate.goose )
build/validate 9000 6000 4500 3500       # rigid wall, bird spacings in µm
build/validate --panel 9000 6000 4500    # the panel
```

Add `--csv` to `validate` to print the time history.

`gen/strike.c` and `gen/strike.h` are the C that the Goose compiler generates from `sim/strike.goose`. They are C, not C++, and nobody edits them by hand.
They are in the repository so that you can read the code that Emscripten compiles to `web/strike.wasm`.
`build.sh` writes them, and writes `gen/sha256sums` with the SHA-256 of `sim/strike.goose`, the C and the wasm.

## Tests

```sh
nix shell nixpkgs#nodejs --command npm test
```

The tests (`test/strike.test.mjs`) load the committed `web/strike.wasm` in Node and check:

- that the C in `gen/` and the wasm come from the current `sim/strike.goose` (the sums in `gen/sha256sums`), and that `web/strike.goose` is the same as `sim/strike.goose`,
- the load case constants and the Wilbeck reference pressures,
- the rigid wall impulse (m v within 2 %) and energy balance (within 1 %),
- the panel energy balance (within 1 %), the panel mass, and that a new set-up gives the same result,
- that the phases one at a time (`step_phase`) give the same result as whole steps (`advance`).

CI runs the same tests on each push.

## Sources

- 14 CFR 25.571(e)(1) and 25.1505. Electronic Code of Federal Regulations, through Amendment 25-148.
- EASA Type-Certificate Data Sheet EASA.A.064 (Airbus A318, A319, A320, A321), Issue 62.
- R. L. Wilbeck, *Impact behavior of low strength projectiles*, AFML-TR-77-134, 1978 (DTIC ADA060423).
- S. Heimbs, Computational methods for bird strike simulations: a review, *Computers & Structures* 89 (2011) 2093–2112, doi:10.1016/j.compstruc.2011.08.007.
- C. Perdikoulis, E. Theotokoglou, G. Giannopoulos, Bird strike virtual testing simulations and results, for preliminary airframe design structural optimization, WCCM-ECCOMAS 2020 (2021).
- ASM Aerospace Specification Metals, data sheets for aluminium 2024-T3 and 7075-T6.

## Licence

MIT (see `LICENSE`), for the code of this repository.
Goose is Apache-2.0. This repository does not include Goose source, but the generated `gen/strike.c` and `web/strike.wasm` contain the Goose runtime that the compiler puts in its output. Thus `web/GOOSE-LICENSE.txt` has the Goose licence.
The fonts in `web/fonts/` are under the SIL Open Font License (the `-OFL.txt` files).
