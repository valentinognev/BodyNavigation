# CADAC HYPER5 (Round3 Hyper + Target + Satellite) — design

Date: 2026-09-06
Status: approved (chat). Second slice plan 1. Depends on first-slice kernel + Round3.

Parent: `docs/superpowers/specs/2026-09-04-cadac-python-design.md`

## Goal

Run CADAC HYPER5 from JSONC: types `HYPER5` and `TARGET3`, with `SATELLITE3` registered. First e2e is Demo 4.7 / `input.asc` (RR3X + Truck_t1, 25 s). Numerics regression-close to `CADAC_Simulations/HYPER5_250113/HYPER5/`.

## Non-goals

HYPER6 / Round6. Targeting e2e (sat demos). Unused C++ guidance 30/33/40/43. AIM5, CRUISE5, MAGSIX, SAM6, SRAAM6, AGM6, ROCKET6. `cadac translate-asc` CLI. Monte Carlo. CADAC packet-slot indexing.

## Layout

```
Python/src/cadac/vehicles/hyper5/
  vehicle.py      # Hyper5 type="HYPER5"
  aero.py
  propulsion.py
  forces.py
  control.py
  guidance.py
  seeker.py
  intercept.py
  targeting.py
  target.py       # Target3 type="TARGET3"
  satellite.py    # Satellite3 type="SATELLITE3"
Python/cases/hyper5/          # translated input.asc + aero (prop if present)
Python/tests/unit/test_hyper5_*.py test_target3_*.py test_satellite3_*.py
Python/tests/e2e/test_hyper5_pronav.py
```

Reuse existing `Round3Environment` / `Round3Newton` (ISO 62). Do not copy Cruise3 or Plane5 vehicle modules; port HYPER5 C++.

## JSONC types

| `type` | Class | Decks |
|---|---|---|
| `HYPER5` | `Hyper5` | aero required; prop required if `mprop!=0` else optional |
| `TARGET3` | `Target3` | none |
| `SATELLITE3` | `Satellite3` | none |

`TARGET3` in this slice is HYPER5 Target. CRUISE5 Target, if different, binds later as `(family, type)`.

Unknown types still `ValueError`. `CRUISE3` / `PLANE` / `PLANE6` stay registered.

## Kernel changes

- `run_loop`: if a scenario module name is not on that vehicle, skip (CADAC dummy methods). Do not `KeyError`.
- `run_scenario`: require aero/prop decks only when the type needs them. Do not force decks on `TARGET3` / `SATELLITE3`.
- Constructor factory may differ per type (Target/Satellite take name + events only).
- `plot_rows` / `plot.csv`: vehicle slot 0 only (Demo 4.7 Hyper). Do not interleave Target rows. Columns: plot-flagged store names (PLANE path), not HYPER3 `PLOT_COLUMNS`.
- Seeker/targeting read combus **by field name**, not C++ `Packet data[i]` slots.

## Hyper5 modules

CADAC MODULES (Demo 4.7): `environment`, `aerodynamics`, `propulsion`, `forces`, `newton`, `seeker`, `guidance`, `control`, `intercept`. `targeting` is defined on Hyper; include it in the vehicle so JSONC may list it (Demo 4.7 omits it from MODULES — define still runs).

**Aerodynamics** (`hyper_modules.cpp`): 2D `look_up` `cn_rr3x_vs_alphax_mach`, `ca_rr3x_vs_alphax_mach`; `cd`/`cl` from cn/ca and alpha; `cla` from ±2 deg. `area` from JSONC.

**Propulsion:** `mprop` 0 none; 1 fixed-phi; 2 auto-phi q-hold; 3 keep-phi. Else `ValueError`. Port C++ `Hyper::propulsion`.

**Forces:** FSPV as HYPER5 `Hyper::forces` (same algebra as Cruise3/Plane5). FSPV is forces-owned; newton does not define it (skip-if-exists if already defined).

**Control:** helpers `control_heading`, `control_flightpath`, `control_bank`, `control_load`, `control_lateral`, `control_altitude`. Dispatcher `mcontrol` in `{0,3,4,6,16,36,40,44}` as C++ (HYPER5 `input*.asc`; `mcontrol 03` is int 3). `0` zeros `phimvx`/`alphax`. Else `ValueError`. Then `TBV=cadtbv(phimv,alpha)`, `TBG=TBV@TVG`.

**Guidance:** `0` zeros commands and return. `44` `guidance_point`; `66` `guidance_pronav`; `70` `guidance_arc`. Else `ValueError`. Write `alcomx`/`ancomx` as C++ (clips via control limiters).

**Seeker:** `0` return. `1` acquire by ground range vs `acq_range`, then set `mseeker=3` and track (LOS, `range_go`, `STBG`, closing speed). Else `ValueError`. Target kinematics from combus names: `lonx`, `latx`, `alt`, `psivgx`, `thtvgx`, `dvbe`, `VBEG`, `SBII`.

**Intercept:** no `sys.exit` / `cout`. `halt`, ground (`alt<=0`), and closest-approach vs tracked target: `vehicle.health=0` and `ctx.combus[slot].status=0`. `write` latch as C++.

**Targeting:** `mtargeting==0` return; `==1` satellite visibility / closest target as C++ `Hyper::targeting` (needs `SATELLITE3` packets). Else `ValueError`. First e2e does not list this module.

## Target3

Modules: `environment`, `newton`, `forces`, `intercept` only.

**Forces:** Coriolis + centrifugal − gravity + `fwd_accel`/`side_accel` as `Target::forces`. Demo 4.7 leaves accels at 0.

**Intercept:** copy combus status into `targ_health` (1 alive, 0 dead, −1 hit).

**com_names:** union of Round3 fields with `com` in outputs (at least `time`, `mach`, `lonx`, `latx`, `alt`, `dvbe`, `psivgx`, `thtvgx`, `VBEG`, `SBII`).

## Satellite3

Modules: `environment`, `newton`, `forces`. `sat_thrust`, `sat_mass` (default 100). `FSPV` from thrust/mass as C++. No seeker. No sat-targeting e2e in this plan.

## First case

Translate `CADAC_Simulations/HYPER5_250113/HYPER5/input.asc` (same as `input_Demo_4_7_pro_nav.asc`) + `hyper5_aero_deck.asc` to `Python/cases/hyper5/`. `end_time` 25. Vehicles: `HYPER5` RR3X, `TARGET3` Truck_t1. `mprop=0`, `mcontrol=44`, `mguidance=66`, `mseeker=1`, `acq_range=6000`.

## Testing

Unit: `rtol=1e-12`, `atol=1e-14` vs CADAC formulas. TDD each task.

E2E: `tests/e2e/test_hyper5_pronav.py`. Skip if `tests/e2e/goldens/hyper5/plot.csv` absent (do not create the file). Else `run_scenario` on the translated case; compare Hyper plot columns present in both; skip sentinel `time=-1`; CSV `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`.

Regression: HYPER3 e2e and FALCON5/FALCON6 unit tests on commits that touch kernel/cli/plot.

## Process

Grok implementer `cursor-grok-4.6-high` per task; Grok task reviewer; Grok whole-plan review. No Composer Fast / Kimi / Fast. Controller does not patch physics. Isolated worktree; no commit to main without user ask.

## Python style

Named state, numpy `@`, CADAC numeric order copied not “improved.” Same module **name** binds Hyper5 vs Target3 vs Cruise3 by vehicle type.
