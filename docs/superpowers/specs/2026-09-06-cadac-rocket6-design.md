# CADAC ROCKET6 Hyper (Round6 SLV) — design

Date: 2026-09-06
Status: approved (parent dispatch). Sibling of HYPER6 Round6. Depends on HYPER6 plan complete (`cadac.eom.round6`, `cadac.math.wgs84`).

Parent: `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
Sibling: `docs/superpowers/specs/2026-09-06-cadac-hyper6-design.md`

## Goal

Add JSONC three-stage SLV vehicle **Rocket6** that reuses the existing Round6 EOM and WGS84 layer. CADAC type token is `HYPER6` (input.asc `HYPER6 SLV`) and **collides** with the existing Python hypersonic `Hyper6`. Bind via family dispatch: `("rocket6", "HYPER6")`. First case: `CADAC_Simulations/ROCKET6_250122/ROCKET6/input.asc` (Vandenberg insertion, `ENDTIME 190`). Numerics regression-close to that C++ tree. Do not import `cadac.vehicles.hyper6.*`.

## Non-goals

HYPER6 hypersonic climb vehicle (stays `_VEHICLE_TYPES["HYPER6"]`). `input_ballistic.asc`. Markov / Monte Carlo sampling (`MONTE 1` ignored). NASA-extended atmosphere (`matmo=1`), weather-table atmosphere (`matmo=2`), constant wind (`mwind=1`), Dryden with live `rand()`. Proportional RCS (`rcs_type=1`). RCS side-force (`mrcs_force!=0`). TVC `mtvc` 1 and 3. Control modes other than `maut` 0 and 53. Guidance other than `mguide` 0 and 5. `maero` other than 11/12/13. `mprop` 1/2 (hyper ramjet). `mins` other than 0/1. `mgps`/`mstar` other than 0–3. Radar/Satellite/Ground0. Overwriting `_VEHICLE_TYPES["HYPER6"]`. Re-porting Round6 kinematics/euler/newton (ROCKET6 newton has no `minit`; insertion is geographic ICs already covered by Python `minit==0`).

## Layout

```
Python/src/cadac/vehicles/rocket6/
  vehicle.py      # Rocket6 type="HYPER6", family="rocket6"
  aero.py
  propulsion.py   # analytic; no PROP_DECK
  gps.py
  startrack.py
  ins.py
  guidance.py
  control.py
  rcs.py
  tvc.py
  forces.py
  intercept.py
Python/cases/rocket6/            # translated input.asc + aero_deck_SLV + weather_deck_Wallops
Python/tests/unit/test_rocket6_*.py
Python/tests/e2e/test_rocket6_insertion.py
```

Reuse: `cadac.eom.round6` (Environment, Kinematics, Euler, Newton), `cadac.math.wgs84`. Patch Round6 **environment only** for insertion `mair=012`. Do not fork kinematics/euler/newton.

## Family dispatch

`VehicleSpec.family: str | None` is the **canonical** resolved family (default `None`). `VehicleSpec.weather_deck: Path | None`.

JSONC may set `"family"` on the **scenario** and/or on a **vehicle**. `load_scenario` resolves: vehicle `family` if present, else scenario `family`, else `None`. Vehicle wins. Factory reads only `VehicleSpec.family`.

Factory (`cadac.cli._build_vehicle`):

- `family is not None`: look up **only** `_VEHICLE_FAMILIES[(family, type)]`. Miss → `ValueError` with path, family, and type. Do **not** fall back to `_VEHICLE_TYPES`. After this lookup, construct with the family-specific constructor (Rocket6: aero required, prop optional, weather into environment). Do **not** fall through to the Hyper6 both-decks constructor.
- `family is None`: look up `_VEHICLE_TYPES[type]` (existing Hyper6 stays `"HYPER6"` with no family).

`_VEHICLE_FAMILIES` is created only if missing; never replace/wipe an existing dict. Register with `setdefault(("rocket6", "HYPER6"), Rocket6)`. Never assign `_VEHICLE_TYPES["HYPER6"]`. Do not retarget unknown-type tests (`AIM5`, `NO_SUCH_TYPE` stay as they are).

`translate_scenario_asc(src, dst_dir, family=None)`. If `family` is set, write scenario-level `"family"` (vehicles keep an explicit `"family"` only if the ASC/JSON already had one). Existing callers omit `family` and stay unchanged.

JSONC for this case (scenario-level `family` from translate; vehicle-level `family` also legal and wins):

```jsonc
{ "title": "...", "family": "rocket6",
  "vehicles": [{ "type": "HYPER6", "name": "SLV",
    "aero_deck": "aero_deck_SLV.jsonc",
    "weather_deck": "weather_deck_Wallops.jsonc",
    "params": { ... }, "events": [ ... ] }] }
```

No `prop_deck`. Rocket6 constructor: aero deck required; `prop_deck` optional and unused (C++ has no `PROP_DECK`; thrust is analytic). `weather_deck` optional at construct; `mair=12` execute requires a weather Datadeck.

## Round6 EOM

ROCKET6 `kinematics.cpp` / `euler.cpp` / `newton.cpp` match the HYPER6 Round6 layer already in Python (geographic `minit=0` ICs, WGS84, US76). Do not re-port. Python `Round6Newton` `minit!=0` → `ValueError` stays; insertion does not set `minit`.

**Environment patch** (shared `Round6Environment`, `weather_deck=None` default so HYPER6 climb is unchanged):

Keep `Round6Environment.initialize`: `dvba=dvbe` as C++ `init_environment`. Do not drop it.

`mair = |matmo|mturb|mwind|` as C++ (`matmo=mair//100`, `mturb=(mair-matmo*100)//10`, `mwind=(mair-matmo*100)%10`).

Allowed:

- `(0,0,0)` / `mair=0`: existing US76, `VAED=0`, `dvba` from geographic speed. HYPER6 climb.
- `(0,1,2)` / `mair=12` (`input.asc` writes `mair 012`): US76 atmosphere; tabular wind from weather deck tables `speed` and `direction` vs `alt`; Dryden turbulence with **`gauss_value=0`** (Monte Carlo out of scope; do not call `rand()`). Wind smoother `VAEDS` as C++ (`int_step` from `ctx`). Dryden filter states as C++ `environment_dryden` with zero white-noise drive. `dvba` from `VBED-VAED`. `mair=12` with no weather Datadeck → `ValueError`.

Any other `mair` → `ValueError` (including `matmo=1/2`, `mwind=1`, turb-only `mair=10`). Skip `mfreeze` / `trcode` if absent. `mair=0` execute must still work when `ctx is None` (existing unit tests).

Define Dryden/wind fields C++ `def_environment` lists (`turb_length`, `turb_sigma`, `taux1`/`taux1d`/`taux2`/`taux2d`, `VAEDS`/`VAEDSD`, `tempc`, …). Extra fields are skip-if-exists on Hyper6.

## Stochastic input (translator)

`MONTE` lines ignored (already skipped). C++ `nmonte==0` means: `GAUSS name mean sigma` → `params[name]=mean`; `RAYL name first` → `params[name]=first`; `MARKOV name sigma bcor` → `params[name]=0`. Python always takes that deterministic path. Do not implement Markov time series.

`WEATHER_DECK file` → `weather_deck` JSONC path (`weather_deck_Wallops.jsonc`). Convert the ASC deck with existing `deck_asc_to_jsonc`.

INS `def_ins` C++ `gauss(0,σ)` instrument vectors and Cholesky `GAUSS_INIT`: initialize to **0**. Port the `mins==1` error ODEs and GPS/star updates so structure is present; insertion with zero draws is deterministic.

## Rocket6 insertion modules

CADAC MODULES order (bind Round6 vs Rocket6 by name): `kinematics`, `environment`, `propulsion`, `aerodynamics`, `gps`, `startrack`, `ins`, `guidance`, `control`, `rcs`, `tvc`, `forces`, `newton`, `euler`, `intercept`.

**Aerodynamics:** `maero` 13 (3 stages), 12 (2 stages), 11 (last stage) as C++ `Hyper::aerodynamics` + `aerodynamics_der` from `aero_deck_SLV.asc` (`ca0slvN_vs_mach`, `caaslvN_vs_mach`, `ca0bslvN_vs_mach`, `cn0slvN_vs_mach_alpha`, `clm0slvN_vs_mach_alpha`, `clmqslvN_vs_mach` for N=3,2,1). Else `ValueError`. Local CADAC sign. Tables from JSONC at runtime.

**Propulsion:** analytic. `mprop` 0 none; 3 constant-thrust rocket; 4 same thrust as 3 (LTG writes `mprop`). `thrust=spi*fuel_flow_rate*AGRAV`; fuel `fmasse` via `integrate`; `vmass=vmass0-fmasse`; `fmassr=fmass0-fmasse`; linear `IBBB` and `xcg` vs `fmasse/fmass0`; shutdown `fmassr<=0` sets `mprop=0`. When `mprop==0`, **every step** zeros `thrust`, `fmasse`, and `fmassr` (and `fmassd`) as C++ — the BECO event sets `mprop=0` only and does not reset `fmasse`; propulsion does. Vacuum thrust, no backpressure. Else `ValueError`. Skip `mfreeze` if absent. `init_propulsion` is a no-op.

**TVC:** `mtvc==0` return. `mtvc==2` second-order `tvc_scnd` as C++ (`FPB`/`FMPB` from `gtvc`, `parm-xcg`, `thrust`). Else `ValueError`. CADAC sign local. `dt=ctx.int_step`.

**RCS:** decode `rcs_type=mrcs_moment//10`, `rcs_mode=mrcs_moment%10`. Insertion: 21 (Schmitt + Euler angles), 20 (Schmitt + roll only), 22 (Schmitt + thrust-vector `UTBC`). Port `rcs_schmitt`, on-off moments. `mrcs_force==0` only (no `FARCS`). `mrcs_moment==0` no moments. Other type/mode/`mrcs_force` → `ValueError`. CADAC sign local.

**Control:** `mauty=maut//10`, `mautp=maut%10`. Modes `{0, 53}` only (`53` → yaw accel 5 + pitch accel 3). Port `control_yaw_accel` / `control_normal_accel` as C++ (online `waclp`/`paclp` from `pdynmc`). `maut==0` writes limited `delecx`/`delrcx` (zeros if controllers not called). Else `ValueError`. Limit `|del*|` as C++. Pitch uses INS `FSPCB`/`qqcx`/`dvbec`; yaw uses `FSPCB`/`rrcx` and newton `dvbe` as C++ `control.cpp`. Skip undefined C++ holes (`dalimx`, `mroll`, …) if absent; `define` the C++ `def_control` list.

**Forces:** `FAPB=[pdynmc*refa*cx, pdynmc*refa*cy, pdynmc*refa*cz]`. `FMB=[pdynmc*refa*refd*cll, pdynmc*refa*refd*clm, pdynmc*refa*refd*cln]` — use aero `refd`, **not** GHAME `refb`/`refc`. Add `FPB`/`FMPB` if `mtvc` in {1,2,3} (insertion uses 2); else if `mprop` add `thrust` to `FAPB[0]`; add `FMRCS` if `0 < mrcs_moment <= 23`. Do not write `FSPB`. Missing `FARCS`/`FMRCS`/`FPB` → zero.

**INS:** `mins==0` ideal copies as C++ `if(mins==0)` (TBI→TBIC, FSPB→FSPCB, SBII→SBIIC, VBII→VBIIC, WBIB→WBICB, geographic/Euler/flight-path computed names). `mins==1` space-stabilized error eqs as C++ (`ins_gyro`, `ins_accl`, `ins_grav` with `cadac.math.wgs84.GM`); GPS update when `mgps==3` (then `mgps=2`); star update when `mstar==3` (then `mstar=2`). Instrument errors and Cholesky draw = 0. `init` for `mins==0` is no-op; `mins==1` loads zero `ESBI`/`EVBI`/`RICI`. Else `ValueError`. Skip `mroll` if absent (treat as 0). Own module; do not import Hyper6 INS.

**GPS:** Sequential `if` fall-through in **one** `execute`, as C++ `gps.cpp` (not `elif`). `mgps==0` return. `mgps==1` init (Yuma week 787 `gps_sv_init` tables, `rsi=26560000`, `wsi=sqrt(GM/rsi**3)`, `incl=0.95986`, P/FF/PHI as C++) then set `mgps=2`; the following `if(mgps==2)` in the **same** call then runs extrapolate. `2` clock growth + covariance extrapolate; when `time-gps_epoch >=` acq/step, `mgps=3`. `3` quadriga + 8-state EKF as C++ `gps.cpp`; writes `SXH`/`VXH`. Else `ValueError`. `<4` visible SVs: set `mgps=1`, no `cout`. Keep 8×8 `PP`/`PHI`/`FF` and almanac as **instance** state (not C++ `static`, not 3×3 store packing). No console quadriga dump. Kalman `numpy` inverse of the C++ `(HH*PP*~HH+RR)` product; pin a frozen residual vs C++ formulas at `rtol=1e-12`.

**Startrack:** Sequential `if` fall-through in **one** `execute`, as C++ `startrack.cpp` (not `elif`). `mstar==0` return. `mstar==1` load 25-star catalog, `star_acq=1`, epoch, `mstar=2` if `alt>startrack_alt`; the following `if(mstar==2)` in the **same** call then runs the wait/step logic. `2` wait acq/step then `mstar=3` if high enough. `3` `star_triad` max-volume + tilt `URIC` from `TRIAD_MEAS*inv(TRIAD_TRUE)` as C++. Biases/noises from store (translator zeros Markov). Else `ValueError`. Catalog copied verbatim from `star_init`. Add `cart_from_pol` to `cadac.math.frames` (C++ `Matrix::cart_from_pol`). No console triad names.

**Guidance:** `mguide==0` zero `UTBC` and return. `mguide==5` LTG as C++ `guidance_ltg` and helpers (`_tgo`, `_igrl`, `_trate`, `_trate_rtgo`, `_pdct`, `_crct`). Needs `cad_kepler` (Morth) in `cadac.math.wgs84` from ROCKET6 `utility_functions.cpp` (`GM`, `SMALL=1e-7`). C++ `exit(1)` on LTG `x==1` → `ValueError` (no `sys.exit`). No BECO `cout`. Writes `mprop` and `beco_flag`. Else `ValueError`.

**Intercept:** `alt<=0` once: `vehicle.health=0`; if `ctx.combus` is not None, `ctx.combus[slot].status=0`. Diagnostic `modes=mguide*1000+mauty*100+mautp*10+mprop`. No `sys.exit`. No required console miss banner.

## First case

Translate `input.asc` (not `input_insertion.asc`) with `family="rocket6"` and call `deck_asc_to_jsonc` on `aero_deck_SLV.asc` and `weather_deck_Wallops.asc` into `Python/cases/rocket6/`. `end_time` 190. `int_step` 0.001. `mair` 12. Events as ASC: `time>10` (TVC/accel/RCS roll), empty `thrust=0` (event_time reset), `event_time>1` (stage 2 + `mguide=5` + `mprop=4`), `event_time>51.5` (stage 3), `beco_flag=1` (`mguide=0`, `mprop=0`).

Smoke: short run (`0.1 s`) `alt` near 100.

## Testing

Unit: `rtol=1e-12`, `atol=1e-14` vs CADAC formulas. TDD each task.

E2E: `tests/e2e/test_rocket6_insertion.py`. Skip if `tests/e2e/goldens/rocket6/plot.csv` absent. If present, that golden is a C++ plot from a **deterministic** run: `MONTE` off (`nmonte==0` path) **and** Dryden white-noise drive forced to 0 — **not** a raw `input.asc` `plot.csv` (which used `MONTE 1` and live `rand()`). CSV tolerances on plot-flagged names present in both (`alt`/`vmach` if both present); sentinel `time=-1`; `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`. Pattern HYPER6 climb e2e.

Regression: HYPER3 e2e, HYPER5 units, FALCON5/FALCON6 units, HYPER6 climb units (and e2e skip-without-golden).

## Process

Same as HYPER6 spec: Grok `cursor-grok-4.6-high` implementer + reviewer per task; whole-plan review; no Fast/Kimi; controller does not patch physics.

## Python style

Named state; CADAC numeric order; same module name binds Round6 vs Rocket6 by vehicle composition. CADAC sign `<0 → -1` else `+1` local helper; not `np.sign`. No `sys.exit`. `LARGE=1e10` module-level in gps if C++ uses it; not in locked `cadac.constants`.
