# CADAC CRUISE5 (Round3 UAV + Target + Satellite) — design

Date: 2026-09-06
Status: approved (chat). Depends on kernel + Round3 + HYPER5 (cadine, skip-missing-modules, plot slot 0).

Parent: `docs/superpowers/specs/2026-09-04-cadac-python-design.md`

## Goal

Run CADAC CRUISE5 from JSONC: family `cruise5` with CADAC types `CRUISE3` (UAV), `TARGET3` (Tank), `SATELLITE3` (Sat). First e2e is `input_1.asc` (UAV + Tank_t1 + Sat_s1, 410 s). Numerics regression-close to `CADAC_Simulations/CRUISE5_250115/CRUISE5/`.

## Non-goals

HYPER3 `Cruise3` / HYPER5 `Hyper5` / `Target3` / `Satellite3` imports. AIM5, MAGSIX, Round6. `input.asc` nine-vehicle case (later/non-goal; not first e2e). Unused C++ guidance `03/33/40/60/06/66/70`. Unused C++ `mcontrol` `1/10/11/3/4/40/6/16/36`. Seeker acquire/track (`mseeker` 1/3). Monte Carlo. CADAC packet-slot indexing. Overwriting global `_VEHICLE_TYPES` `CRUISE3`/`TARGET3`/`SATELLITE3`. Retargeting unknown-type tests that use `AIM5` (AIM5 plan owns AIM5→ROTOR).

## Layout

```
Python/src/cadac/vehicles/cruise5/
  vehicle.py      # Cruise5 type="CRUISE3"
  aero.py
  propulsion.py
  forces.py
  control.py
  guidance.py
  seeker.py
  intercept.py
  targeting.py
  target.py       # Cruise5Target type="TARGET3"
  satellite.py    # Cruise5Satellite type="SATELLITE3"
Python/cases/cruise5/          # translated input_1.asc + cruise3_*_deck.asc
Python/tests/unit/test_cruise5_*.py
Python/tests/e2e/test_cruise5_input1.py
```

Reuse existing `Round3Environment` / `Round3Newton` (ISO 62) and `cadac.math.earth.cadine`. Do not copy or import HYPER3 `Cruise3` or HYPER5 vehicle modules; port CRUISE5 C++ (`cruise_modules.cpp`, `target_modules.cpp`, `satellite_modules.cpp`, functions files).

## Family dispatch

Canonical API (same as AIM5). `VehicleSpec.family: str | None = None` is the source of truth for lookup. `RunConfig` does not store a separate family field.

JSONC: optional scenario-level `"family"` is the default for vehicles that omit `"family"`. A vehicle-level `"family"` key wins. `load_scenario` writes the **resolved** value onto each `VehicleSpec`.

- If `spec.family` is set: look up **only** `_VEHICLE_FAMILIES[(family, type)]`. Never fall through to `_VEHICLE_TYPES`. Missing pair → `ValueError` that includes **type and family**.
- If `spec.family` is `None`: look up **only** `_VEHICLE_TYPES[type]` (HYPER3 `CRUISE3`, HYPER5 `TARGET3`/`SATELLITE3` stay working). Missing → `ValueError` that includes type.

`translate_scenario_asc(src, dst_dir, family=None)` writes `"family"` on **each vehicle** when `family` is given. Omit the key when `family` is `None`.

Idempotent with AIM5: if this API already exists, keep it. Do **not** reassign `_VEHICLE_FAMILIES = {}` (would wipe AIM5 pairs). Register the three cruise5 pairs only at vehicle-registry time.

CRUISE5 registers **only** the family map (never overwrite global `CRUISE3`/`TARGET3`/`SATELLITE3`):

| `(family, type)` | Class |
|---|---|
| `("cruise5", "CRUISE3")` | `Cruise5` |
| `("cruise5", "TARGET3")` | `Cruise5Target` |
| `("cruise5", "SATELLITE3")` | `Cruise5Satellite` |

Plot columns: HYPER3 class `cadac.vehicles.cruise3.vehicle.Cruise3` keeps `PLOT_COLUMNS`. `Cruise5` (`type=="CRUISE3"`) uses flagged plot names (PLANE / HYPER5 path). Dispatch by **class**, not the type string.

## JSONC types (family `cruise5`)

| `type` | Class | Decks |
|---|---|---|
| `CRUISE3` | `Cruise5` | aero **and** prop required |
| `TARGET3` | `Cruise5Target` | none. Constructor `(name, events=None)` |
| `SATELLITE3` | `Cruise5Satellite` | none. Constructor `(name, events=None)` |

Unknown `(family, type)` or unknown `type` when `family` is None still `ValueError` (type, and family when set). Do not retarget unknown-type tests (AIM5 plan owns AIM5→ROTOR).

## Kernel / CLI (this slice)

- Family dispatch as above. Idempotent if AIM5 already added the same API: do not change the signature; do not reset `_VEHICLE_FAMILIES`; only register the three cruise5 pairs (vehicle-registry task) and translate with `family="cruise5"`.
- `run_loop` skip-missing-modules and plot slot 0 already exist (HYPER5). Reuse.
- Seeker/targeting/intercept read combus **by `Packet.type`** (`TARGET3` / `SATELLITE3`) and **field names**, not C++ `id.find("t")` / `data[i]` slots.
- No `sys.exit` / `cout`. Intercept sets `vehicle.health=0` and `ctx.combus[slot].status=0`.

## Cruise5 modules

CADAC MODULES (`input_1.asc`): `environment`, `aerodynamics`, `propulsion`, `forces`, `newton`, `targeting`, `seeker`, `guidance`, `control`, `intercept`. Include all on the UAV so define always runs. Target/Satellite skip names they lack (existing kernel).

**Aerodynamics** (`Cruise::aerodynamics`): 1D `look_up` `cd0_vs_mach`, `cl0_vs_mach`, `cla_vs_mach`, `ckk_vs_mach`, `cla0_vs_mach`. `cl=cla0+cla*alphax`; `cd=cd0+ckk*(cl-cl0)**2`; `cl_ov_cd=cl/cd`. Default `area=0.929` (C++ `def_aerodynamics`; `input_1.asc` does not set `area`). `alphax` is control-owned (do not define it in aero). Do not import HYPER3 `Cruise3Aero`.

**Propulsion:** `mprop` 0 none (thrust=0, fuel flow 0, still `cg_vs_mass`, **early return** — do not integrate fuel); 1 `thrust_com`; 2 idle `fidle_vs_alt_mach` + `iff_vs_alt`; 3 max `tav_vs_alt_mach` + `ff_vs_thrust_alt_mach`; `mprop>3` Mach-hold (C++ sets `mprop=4`, then 5 if below idle / 6 if above max). Fuel `fmasse` via `integrate` of `ff`; `mass=mass_init-fmasse`; `fuelmass=fuel_init-fmasse`; `fuelmass<=0` → `thrust=0`. `mprop<0` → `ValueError`. Initialize `mass=mass_init`. Tables from `cruise3_prop_deck.asc`. No forces.

**Forces:** FSPV as CRUISE5 `Cruise::forces` (same algebra as Cruise3/Hyper5). Skip-if-exists (Round3 newton does not define FSPV). Plot-flagged.

**Control:** helpers `control_bank`, `control_load`, `control_lateral`, `control_altitude` ported from CRUISE5 `cruise_modules.cpp` (not Plane5 / Hyper5 classes). Dispatcher `mcontrol` in `{0,44,46}` as C++ (`input_1.asc`: 46 midcourse, event 3 sets 44). `0` zeros `phimvx`/`alphax`. Else `ValueError`. Then `TBV=cadtbv(phimvx*RAD, alphax*RAD)`, `TBG=TBV@TVG`. `input_1` bank `tphi=0.5`, `philimx=70`.

**Guidance:** `0` return without writing commands (C++ returns before store load). Locked C++ mapping (before limiters): `30` → `alcomx=ALGV[1]/grav`, `ancomx` stays 0; `43` → `alcomx=APGV[1]/grav`, `ancomx=-ALGV[2]/grav` (point lateral, line pitch — C++ after last waypoint). Else `ValueError` (do not port 03/33/40/60/06/66/70). Then clips via `anposlimx`/`anneglimx`/`allimx` as C++ `Cruise::guidance`. `cadine` for waypoint inertial. `wp_flag` CADAC `sign` (`<0 → -1` else `+1`).

**Seeker:** `mseeker==0` return. Else `ValueError`. Define C++ `def_seeker` fields so intercept can read defaults. Do not port acquire/track in this slice (`input_1` keeps `mseeker=0`).

**Intercept:** no `sys.exit` / `cout`. Ground `alt<=0` and `write` → health/status 0. `mguidance in {33,43}` and `alt<=wp_alt` and `write` → miss `||SWBG||`, health/status 0. `write` latch. Waypoint overfly (`mguidance` 30/40/70 and `wp_flag==-1`) is diagnostic only in C++ (console); Python does not print and does not kill. Optional port of `mseeker==3` closest-approach (dead code until seeker track exists); no `halt` field (CRUISE5 C++ has none).

**Targeting:** `mtargeting==0` return. `==1` satellite visibility / closest TARGET3 as C++ `Cruise::targeting` / `targeting_satellite` / `targeting_grnd_ranges` (port C++ `angle(SBII,STII)` visibility, not SSII). Writes `wp_lonx`/`wp_latx`/`wp_alt` from that target. `BIG=1e10` module-level (C++ `global_constants.hpp`); do not add to `cadac.constants`. Else `ValueError`. `input_1` enables targeting on the third `wp_flag=-1` event with `del_radius=5000`.

## Cruise5Target

Modules: `environment`, `newton`, `forces`, `intercept` only.

**Forces:** Coriolis + centrifugal − gravity + `fwd_accel`/`side_accel` as CRUISE5 `Target::forces`. `input_1` Tank leaves accels at 0.

**Intercept:** copy combus status into `targ_health` (1 alive, 0 dead, −1 hit).

**com_names:** store fields with `"com"` in outputs (Round3: at least `time`, `mach`, `lonx`, `latx`, `alt`, `dvbe`, `psivgx`, `thtvgx`, `sbeg`, `vbeg`, `sbii`).

## Cruise5Satellite

Modules: `environment`, `newton`, `forces`. `sat_thrust`, `sat_mass` (default 100). `FSPV=[sat_thrust/sat_mass, 0, 0]` as C++. No seeker.

**com_names:** same collection as Cruise5Target — store fields with `"com"` in outputs (Round3: at least `lonx`, `latx`, `alt`, `sbii`, plus `time`, `mach`, `dvbe`, `psivgx`, `thtvgx`, `sbeg`, `vbeg`). Targeting reads SATELLITE3 packets by those names.

## First case

Translate `CADAC_Simulations/CRUISE5_250115/CRUISE5/input_1.asc` + `cruise3_aero_deck.asc` + `cruise3_prop_deck.asc` to `Python/cases/cruise5/` with `family="cruise5"`. `end_time` 410. `int_step` 0.05. `plot_step` 0.2. Vehicles: `CRUISE3` UAV, `TARGET3` Tank_t1, `SATELLITE3` Sat_s1.

UAV params (C++ names): `lonx=14.7`, `latx=35.4`, `alt=7000`, `psivgx=90`, `thtvgx=0`, `dvbe=200`, `alphax=0`, `phimvx=0`, `mprop=4`, `mach_com=0.7`, `mass_init=1000`, `fuel_init=150`, `gfthm=893620`, `tfth=1`, `mseeker=0`, `acq_range=10000`, `mguidance=30`, `pronav_gain=3`, `line_gain=1`, `nl_gain_fact=0.6`, `decrement=1000`, `mcontrol=46`, `anposlimx=3`, `anneglimx=-1`, `gacp=10`, `ta=0.8`, `alpposlimx=15`, `alpneglimx=-10`, `gcp=2`, `allimx=1`, `philimx=70`, `tphi=0.5`, `altcom=7000`, `altdlim=50`, `gh=0.3`, `gv=1.0`, `psivgcx=90`, `gain_psivg=12`, `gain_thtvg=30`, `wp_lonx=14.9`, `wp_latx=35.4`, `psifgx=90`. `area` default 0.929. `wp_alt` default 0 until targeting.

Events (watch `wp_flag = -1`): (1) `wp_lonx=15.25`, `wp_latx=35.54`, `psifgx=90`, `altcom=5000`; (2) `wp_lonx=15.43`, `wp_latx=35.44`, `psifgx=180`, `altcom=2000`; (3) `mtargeting=1`, `del_radius=5000`, `mguidance=43`, `point_gain=1`, `line_gain=1`, `nl_gain_fact=0.6`, `decrement=1000`, `thtfgx=-50`, `mcontrol=44`.

Tank: `lonx=15.4`, `latx=35.3`, `alt=100`, `psivgx=45`, `dvbe=10`. Sat: `lonx=10`, `latx=30`, `alt=500000`, `psivgx=45`, `thtvgx=0`, `dvbe=7700`.

## Testing

Unit: `rtol=1e-12`, `atol=1e-14` vs CADAC formulas. TDD each task. Test files `test_cruise5_*.py` (not `test_cruise3_*`).

E2E: `tests/e2e/test_cruise5_input1.py`. Skip if `tests/e2e/goldens/cruise5/plot.csv` absent (do not create the file). Else `run_scenario` on the translated case; compare UAV (slot 0) plot columns present in both; skip sentinel `time=-1`; CSV `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`.

Regression: HYPER3 e2e, HYPER5/HYPER6/FALCON5/FALCON6 unit tests on commits that touch kernel/cli/plot/translate.

## Process

Grok implementer `cursor-grok-4.6-high` per task; Grok task reviewer; Grok whole-plan review. No Composer Fast / Kimi / Fast. Controller does not patch physics. Isolated worktree; no commit to main without user ask.

## Python style

Named state, numpy `@`, CADAC numeric order copied not “improved.” Same module **name** binds Cruise5 vs Cruise5Target vs Cruise3 by `(family, type)` / class. Local CADAC `sign` (`<0 → -1` else `+1`); not `np.sign`.
