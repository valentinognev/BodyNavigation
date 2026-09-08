# CADAC AIM5 (Aim + Aircraft) — design

Date: 2026-09-06
Status: approved (chat). Depends on first-slice kernel + Round3 + Flat3 + HYPER5 + HYPER6.

Parent: `docs/superpowers/specs/2026-09-04-cadac-python-design.md`

## Goal

Run CADAC AIM5 from JSONC: types `AIM5` (class Aim) and family-scoped `AIRCRAFT3` (class Aircraft). First e2e is `input_hori.asc` (Missile + Target, 10 s). Numerics regression-close to `CADAC_Simulations/AIM5_250114/AIM5/`.

This slice **introduces family dispatch**: CADAC type tokens collide across programs (`AIRCRAFT3` later in SAM6/AGM6). Optional `family` on `VehicleSpec` selects `(family, type)` so AIM5 Aircraft does not bind a future SAM6 Aircraft, and so a family-scoped lookup cannot fall through to a global token (cruise5 `CRUISE3` must not bind HYPER3 `Cruise3`).

## Non-goals

SAM6 / AGM6 / SRAAM6 Aircraft. `input.asc` / `input_multi.asc` / `input_verti.asc` e2e (hori only). Monte Carlo. CADAC packet-slot indexing and `id.find("a")` / `"m"` / `"t"`. `cadac translate-asc` CLI wrapper (library `translate_scenario_asc` grows `family=`). Plane5 vehicle modules. Copying AIM5 `flat3_modules.cpp` (reuse `cadac.eom.flat3`). Changing HYPER5/HYPER6/FALCON/CRUISE3 cases to set `family`.

## Layout

```
Python/src/cadac/vehicles/aim5/
  vehicle.py      # Aim5 type="AIM5"; Aim5Flat3Newton alias wrapper
  aero.py
  propulsion.py
  seeker.py
  guidance.py
  control.py
  forces.py
  intercept.py
  aircraft.py     # Aim5Aircraft type="AIRCRAFT3"
Python/cases/aim5/          # translated input_hori.asc + aim5 aero/prop decks
Python/tests/unit/test_aim5_*.py test_aircraft3_aim5_*.py test_vehicle_family.py
Python/tests/e2e/test_aim5_hori.py
```

Reuse existing `Flat3Environment` / `Flat3Kinematics` / `Flat3Newton` (US76). Do not copy Plane5 vehicle modules; port AIM5 C++ `aim_modules.cpp`, `aircraft_modules.cpp` (and helpers in `aim_functions.cpp` / `aircraft_functions.cpp` only if a module needs them). AIM5 `flat3_modules.cpp` is the same US76 + Newton algebra as Python Flat3; different C++ *names* (`SAEL`/`VAEL`/`FSPA`/`TAL`/`dvae`/`sael1`) map to existing Flat3 names (`SBEL`/`VBEL`/`FSPV`/`TBL`/`dvbe`/`sbel1`).

## JSONC types

| `type` | Class | `family` | Decks |
|---|---|---|---|
| `AIM5` | `Aim5` | optional `"aim5"`; also in global `_VEHICLE_TYPES` | aero **required**, prop **required** |
| `AIRCRAFT3` | `Aim5Aircraft` | **required** `"aim5"` (not global) | none |

`AIRCRAFT3` in this slice is AIM5 Aircraft. SAM6/AGM6 Aircraft bind later as `("sam6","AIRCRAFT3")` / `("agm6","AIRCRAFT3")`.

Unknown types still `ValueError`. `CRUISE3` / `PLANE` / `PLANE6` / `HYPER5` / `HYPER6` / `TARGET3` / `SATELLITE3` stay registered globally. Do not put `AIRCRAFT3` in `_VEHICLE_TYPES`.

Unknown-type tests that used `"AIM5"` as the unregistered token retarget to `"NO_SUCH_TYPE"` (MAGSIX-independent; MAGSIX will register `ROTOR` globally).

## Kernel changes (family dispatch)

This is the **canonical family API** later plans (cruise5, SAM6, AGM6, …) must match.

`VehicleSpec` (`Python/src/cadac/io/scenario.py`) gains `family: str | None = None`. **`VehicleSpec.family` is the source of truth.** JSONC vehicle may include `"family": "aim5"`. Optional scenario-level `"family"` is copied onto vehicles that omit it at load time. Vehicle-level `"family"` wins. Do **not** invent `RunConfig.family` as the only store (`RunConfig` has no family field).

Lookup in `cli.py` `_build_vehicle`:

- If `spec.family` is set: look up **only** `_VEHICLE_FAMILIES[(family, type)]`. Missing → `ValueError` whose message includes `type` **and** `family` (e.g. `{path}: unknown vehicle type {type!r} for family {family!r}`). Do **not** fall through to `_VEHICLE_TYPES` (would bind a future cruise5 `CRUISE3` to HYPER3 `Cruise3`).
- If `spec.family` is `None`: look up `_VEHICLE_TYPES[type]` as today (HYPER3/FALCON/HYPER5/HYPER6 cases unchanged). Missing → `ValueError` whose message includes `type` (e.g. `{path}: unknown vehicle type {type!r}`).

`_VEHICLE_FAMILIES` stays empty until AIM5 registers in this plan.

AIM5 registers:

- `_VEHICLE_TYPES["AIM5"]` = `Aim5` (unique token; type-only still works)
- `_VEHICLE_FAMILIES[("aim5","AIM5")]` = `Aim5`
- `_VEHICLE_FAMILIES[("aim5","AIRCRAFT3")]` = `Aim5Aircraft` (not global)

Deck factory:

- `AIM5` always requires both `aero_deck` and `prop_deck` (unlike HYPER5 `mprop==0`).
- `("aim5","AIRCRAFT3")` is no-deck: `Aim5Aircraft(name, events)` — do **not** add `AIRCRAFT3` to global `_NO_DECK_TYPES` (TARGET3/SATELLITE3 stay the only global no-deck tokens). Use `_NO_DECK_FAMILY_TYPES = {("aim5", "AIRCRAFT3")}`.

`translate_scenario_asc(src, dst_dir, family=None)`: if `family` is given, write `"family"` on **each vehicle** (not a scenario-root key). Existing translate tests: `family is None` → no `family` key in the JSON. AIM5 hori translator call uses `family="aim5"`.

Plot slot 0 already exists (Missile is vehicle 0). Do not change CRUISE3 `PLOT_COLUMNS`. Kernel skip-missing-modules already exists (HYPER5).

## AIM5 C++ names vs Python Flat3

JSONC `params` keep CADAC ASC names from `input_hori.asc`: `sael1`, `sael2`, `sael3`, `dvae`. `Aim5Flat3Newton` (subclass of `Flat3Newton`, lives in `vehicle.py`) copies those onto Flat3 ICs before `super().initialize`: `sbel1=sael1`, `sbel2=sael2`, `sbel3=sael3`, `dvbe=dvae`. Modules read/write Flat3 names: `SBEL`, `VBEL`, `ABEL`, `FSPV` (not `FSPA`), `TBL` (not `TAL`), `TBV`, `TVL`, `dvbe`, `phiavout` (optional; missile omits it → Newton treats as 0).

`com_names` for both Aim5 and Aim5Aircraft is a **vehicle-only union**: after module `define()`, set `self.com_names` to store fields with `"com"` in outputs **plus** `{SBEL, VBEL, psivlx, thtvlx}` (AIM5 C++ Flat3 com set, mapped to Python names). Do **not** retag Flat3 `Field.outputs`. Do **not** edit `cadac.eom.flat3`. Seeker/guidance/intercept read combus **by field name**, not C++ `Packet data[i]` slots.

## Aim5 modules

CADAC MODULES (`input_hori.asc`): `environment`, `kinematics`, `aerodynamics`, `propulsion`, `seeker`, `guidance`, `control`, `forces`, `newton`, `intercept`. Vehicle module list matches that order. Protocol `define` / `initialize` / `execute` / `terminate` on `vehicle.store`.

**Aerodynamics** (`Aim::aerodynamics`): tables `cl_aim_vs_alpha_mach`; `cd_aim_on_vs_alpha_mach` if `mprop` else `cd_aim_off_vs_alpha_mach`. Aeroballistic `alpp=acos(cos(alpha)*cos(beta))`; `phip=atan2(tan(beta), sin(alpha))` with `|sin(alpha)|<SMALL` → `SMALL*sign(sin(alpha))`. `caaim=cdaim*cos(alpha)-claim*sin(alpha)`; `cnpaim=cdaim*sin(alpha)+claim*cos(alpha)`; `cnaim=fabs(cnpaim)*cos(phip)`; `cyaim=-fabs(cnpaim)*sin(phip)`. `cnalp`/`cybet` piecewise in `|alphax|`/`|betax|` degrees as C++ then `*DEG`. `gmax=(cnp_max*pdynmc*area)/(mass*grav)` with tables at `alpmax`. `SMALL=1e-7` module-level (C++ `global_constants.hpp`); not in `cadac.constants`. CADAC sign `<0 → -1` else `+1`; local helper; not `np.sign`; not `flat6._cadac_sign`.

**Propulsion:** `mprop` 0 motor off (thrust 0); 1 tables `thrust_vs_time`, `mass_vs_time` at `time`; `thrust=thrust_sl+(pres_sl-press)*aexit` (`pres_sl` default 101325). If `time>0` and `thrust_sl==0` → `mprop=0`, `thrust=0`. Else `ValueError`. No forces.

**Forces:** `FSPV[0]=(thrust-caaim*pdynmc*area)/mass`, `FSPV[1]=(cyaim*pdynmc*area)/mass`, `FSPV[2]=(-cnaim*pdynmc*area)/mass`. Diagnostics `aax=FSPV[0]/grav`, `alx=FSPV[1]/grav`, `anx=-FSPV[2]/grav`. Skip-if-exists on `FSPV`. Do not port dead `aim[40] acc_longx` (read, unused). Do not write `FSPA`.

**Control:** `init_control` sets `alp=alphax*RAD`, `bet=betax*RAD`. `execute` ports `Aim::control` pitch then yaw P-I + rate lag + incidence lag; `alphax`/`betax` clipped to `alpmax` with CADAC sign. `dt=ctx.int_step`. `integrate` stored-slope. Hori ICs: `tr=0.1`, `ta=2`, `gacp=40`, `alpmax=35`.

**Guidance:** decode `guid_manvr=mguid//10`, `guid_mode=mguid%10`. `guid_mode==1`: `APNA=skew(WOEA)@UTAA*(gnav*abs(dvta))`; `annx=-APNA[2]/grav`; `allx=APNA[1]/grav`. `guid_mode==0`: leave `annx`/`allx` at 0. `guid_manvr==1` and `tgo_aim<tgo_manvr`: add decaying spiral as C++. Circular limiter then `alcomx`/`ancomx`. `guid_mode` not in `{0,1}` or `guid_manvr` not in `{0,1}` → `ValueError`. Hori: `mguid=1`, `gnav=4` (pronav, no spiral). Last C++ write of `aim[108]` is `an_manvr` (not `amp`).

**Seeker:** `tgt_num` default **1** (`def_seeker`). `input_hori` Missile omits `tgt_num` → 1. Identify targets by `Packet.type=="AIRCRAFT3"`; `tgt_num` is **1-based index among AIRCRAFT3 packets in combus order** (not CADAC `id=="a"+n`). Always resolve the packet (writes `acft_com_slot`, `VTEL`, `psivlx_acft`, `thtvlx_acft` from `vars["VBEL"]`/`psivlx`/`thtvlx`). `mseek==0` return after that (no LOS). `mseek==1`: `STAL=STEL-SAEL` with `STEL=vars["SBEL"]`, own `SBEL`/`VBEL`/`TBL`; `dta`, `UTAA=TBL@UTAL`, polar LOS, `dvta=UTAL·VTAEL`, `tgo_aim=dta/abs(dvta)`, `WOEA=TBL@(skew(UTAL)@VTAEL)* (1/dta)`. Else `ValueError`. If `mseek==1` and no AIRCRAFT3 at that index → `ValueError`.

**Intercept:** no `sys.exit` / `cout`. If `dta<500` and `dvta>0`: aspect angles as C++ `Aim::intercept`; `vehicle.health=0`; `ctx.combus[vehicle_slot].status=0`; `ctx.combus[acft_com_slot].status=0`. No ground/`halt`/`write` latch (not in AIM5 C++).

## Aircraft3 (Aim5Aircraft)

Constructor `(name, events=None)` — no decks. `type=="AIRCRAFT3"`. Modules: `Flat3Environment`, `Flat3Kinematics`, `Aim5AircraftGuidance`, `Aim5AircraftControl`, `Aim5AircraftForces`, `Aim5Flat3Newton`. Scenario MODULES may list aero/prop/seeker/intercept — kernel skips names absent on this vehicle.

**Guidance** (`Aircraft::guidance`): `acft_option==0` `ACOML=[0,0,-grav]` (hori). `==1` `ACOML=TVL.T@[0, gturn*grav, -grav]`. `==2` escape vs **first** `Packet.type=="AIM5"` (C++ looks up `id=="t1"` but AIM5 missiles are tagged `"m1"` — Python uses type). Else `ValueError`. `initialize` / `terminate` are `pass`.

**Control** (`Aircraft::control`): bank from `atan2(ACOMV[1], -ACOMV[2])` with `EPS` zero-guard; lag if `tphi` else command; `|phiavx|>=philimx` CADAC sign clip; write `phiavout` (skip-if-exists). Load factor `ancomx=sqrt(ay^2+az^2)/grav`; lag if `tanx`; if `acft_option>0` clip by `pdynmc*clalpha*alplimx/wingloading`. Hori: `tphi`/`tanx` omitted → 0 (no lag); `acft_option=0` skips alpha limiter. ICs: `clalpha=0.0523`, `wingloading=3247`, `philimx=60`, `alplimx=12`. Shared MODULES include `control` **init**; C++ `Aircraft::init_control` is dummy — `initialize` and `terminate` are `pass`.

**Forces:** `FSPV=[acc_longx*grav, 0, -anx*grav]` (`acc_longx` default 0). Skip-if-exists `FSPV`. `initialize` / `terminate` are `pass`.

## First case

Translate `CADAC_Simulations/AIM5_250114/AIM5/input_hori.asc` + `aim5_aero_deck.asc` + `aim5_prop_deck.asc` to `Python/cases/aim5/` with `family="aim5"`. `end_time` 10. `int_step` 0.002. `plot_step` 0.02. Vehicles: `AIM5` Missile, `AIRCRAFT3` Target.

Missile params: `sael1=0`, `sael2=-9000`, `sael3=-10000`, `psivlx=45`, `thtvlx=0`, `alphax=0`, `betax=0`, `dvae=269`, `area=0.01767`, `alpmax=35`, `mprop=1`, `mass=63.8`, `aexit=0.00948`, `mseek=1`, `mguid=1`, `gnav=4`, `tr=0.1`, `ta=2`, `gacp=40`. `tgt_num` omitted (default 1).

Target params: `sael1=0`, `sael2=0`, `sael3=-10000`, `psivlx=-90`, `thtvlx=0`, `dvae=269`, `acft_option=0`, `clalpha=0.0523`, `wingloading=3247`, `philimx=60`, `alplimx=12`.

## Testing

Unit: `rtol=1e-12`, `atol=1e-14` vs CADAC formulas. TDD each task. Aircraft tests named `test_aircraft3_aim5_*.py` so they do not collide with SAM6/AGM6.

E2E: `tests/e2e/test_aim5_hori.py`. Skip if `tests/e2e/goldens/aim5/plot.csv` absent (do not create the file). Else `run_scenario` on the translated case; compare Missile (slot 0) plot columns present in both; skip sentinel `time=-1`; CSV `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`.

Regression: HYPER3 e2e and HYPER5/HYPER6/FALCON5/FALCON6 unit tests on commits that touch kernel/cli/plot/scenario/translate.

## Process

Grok implementer `cursor-grok-4.6-high` per task; Grok task reviewer; Grok whole-plan review. No Composer Fast / Kimi / Fast. Controller does not patch physics. Isolated worktree; no commit to main without user ask. Parent session coordinates; it does not write production code.

## Python style

Named state, numpy `@`, CADAC numeric order copied not “improved.” Same module **name** binds Aim5 vs Aim5Aircraft vs Plane5 by vehicle type. Skip-if-exists on name collisions.
