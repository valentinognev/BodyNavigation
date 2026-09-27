# CADAC HYPER6 Hyper (Round6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Round6 EOM and JSONC type `HYPER6`; unit-test each C++ module used by `input_climb.asc`; e2e vs CADAC CSV when a golden exists.

**Architecture:** New `cadac.eom.round6` (US76 + WGS84, not ISO 62, not Flat6). Vehicle modules port `CADAC_Simulations/HYPER6_250125/HYPER6/*.cpp`. No Plane6/Flat6 classes. Radar/Satellite/Ground0 out of scope.

**Tech Stack:** Python >= 3.11, numpy, pytest. Work in `Python/`.

**Spec:** `docs/superpowers/specs/2026-09-06-cadac-hyper6-design.md`

**Depends on:** HYPER5 plan complete (kernel skip + deck-optional factory).

## Global Constraints

- Spec + parent design spec
- Do not regress HYPER3 e2e, HYPER5 units, FALCON5/FALCON6 units
- Round6 Earth is **WGS84** (`cad_grav84`, `cad_tdi84`, `cad_tgi84`, `cad_geo84_in`, `cad_in_geo84`) from HYPER6 `utility_functions.cpp`. Do not substitute spherical `cadtei`/`cadtge`/`cadsph` or `gravity(alt)` for Round6
- WGS84 constants (`GM`, `C20`, `FLATTENING`, `SMAJOR_AXIS`, `GW_CLONG`, …) live as module-level in the new WGS84 module (C++ `global_constants.hpp`). Do not add them to locked `cadac.constants`
- Atmosphere US76 only (`mair` decoded; climb default 0). Other `mair` → `ValueError`
- JSONC type `HYPER6`. First case `input_climb.asc`
- E2E skip if `Python/tests/e2e/goldens/hyper6/plot.csv` absent
- Grok `cursor-grok-4.6-high` implementer + reviewer; TDD; no Fast/Kimi
- Skip `mfreeze` / plane `trcode` if those names are absent on the store
- Unit rtol=1e-12; CSV e2e rtol=1e-5, atol=max(1e-6, 5e-6*|g|)

## File map

- `Python/src/cadac/math/wgs84.py` — WGS84 helpers + constants
- `Python/src/cadac/eom/round6.py` — Environment, Kinematics, Euler, Newton
- `Python/src/cadac/vehicles/round6/hyper6/{aero,propulsion,actuator,control,forces,guidance,ins,vehicle}.py`
- Tests `test_wgs84.py`, `test_round6_*.py`, `test_hyper6_*.py`
- Case `Python/cases/hyper6/` from `input_climb.asc` + `ghame6_*_deck.asc`
- C++: `environment.cpp`, `kinematics.cpp`, `euler.cpp`, `newton.cpp`, `aerodynamics.cpp`, `propulsion.cpp`, `actuator.cpp`, `control.cpp`, `forces.cpp`, `guidance.cpp`, `ins.cpp`, `utility_functions.cpp`

---

### Task 1: WGS84 earth helpers

**Files:**
- Create: `Python/src/cadac/math/wgs84.py`
- Test: `Python/tests/unit/test_wgs84.py`

**Interfaces:**
- Produces: `cad_in_geo84(lon, lat, alt, time) -> (3,)`, `cad_geo84_in(sbii, time) -> (lon, lat, alt)`, `cad_tdi84(lon, lat, alt, time) -> (3,3)`, `cad_tgi84(lon, lat, alt, time) -> (3,3)`, `cad_grav84(sbii, time) -> (3,)`. Angles in **radians**. Port HYPER6 `utility_functions.cpp` including callees (`cad_geoc_in` if `cad_grav84` uses it). Constants module-level from HYPER6 `global_constants.hpp` (`GM=3.9860044e14`, `C20`, `FLATTENING`, `SMAJOR_AXIS`, `GW_CLONG=0`, …).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from cadac.math.wgs84 import cad_in_geo84, cad_geo84_in, cad_grav84, cad_tdi84

def test_roundtrip_geo84_equator():
    sbii = cad_in_geo84(0.0, 0.0, 0.0, 0.0)
    lon, lat, alt = cad_geo84_in(sbii, 0.0)
    np.testing.assert_allclose([lon, lat, alt], [0.0, 0.0, 0.0], atol=1e-6)

def test_grav84_finite_at_surface():
    sbii = cad_in_geo84(0.0, 0.0, 0.0, 0.0)
    g = cad_grav84(sbii, 0.0)
    assert g.shape == (3,)
    assert np.linalg.norm(g) > 9.0
```

Also pin one `cad_tdi84(0,0,0,0)` element against C++ `assign_loc`.

- [ ] **Step 2:** FAIL `pytest tests/unit/test_wgs84.py -v`
- [ ] **Step 3:** Port C++ (do not use scipy)
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: CADAC WGS84 earth helpers`

---

### Task 2: Round6 environment (mair=0)

**Files:**
- Create: `Python/src/cadac/eom/round6.py`
- Test: `Python/tests/unit/test_round6_environment.py`

**Interfaces:**
- Consumes: `atmosphere76`, `cad_grav84`
- Produces: `Round6Environment.name=="environment"`. `define` C++ `def_environment` mair==0 outputs (`mair`,`press`,`rho`,`vsound`,`vmach`,`pdynmc`,`tempk`,`GRAVG`,`grav`,`VAED`,`dvba`, …). Does not define `alt`/`SBII`/`VBED` (newton). `execute`: decode `matmo=mair//100`, `mturb=(mair-matmo*100)//10`, `mwind=(mair-matmo*100)%10`. Only all-zero: US76 + `GRAVG=cad_grav84(SBII,time)`, `grav=||GRAVG||`, `VAED=0`, `dvba` from geographic speed as C++ (`||VBED||` when no wind). `vmach` not `mach`. Other mair → `ValueError`. Skip hyper `trcode`/`mfreeze` if absent.

- [ ] **Step 1:** alt=10000, SBII from `cad_in_geo84(10*RAD,10*RAD,10000,0)`, `||VBED||=1000`, mair=0 → rho/press match `atmosphere76(10000)`; `vmach` finite; mair=100 raises
- [ ] **Step 2–5:** implement, pass, commit `feat: Round6 environment US76`

---

### Task 3: Round6 kinematics

**Files:** Modify `round6.py`; `Python/tests/unit/test_round6_kinematics.py`

**Interfaces:** Port `Round6::init_kinematics` / `kinematics`. Timing: `time=ctx.sim_time`, `int_step_new`/`out_step_fact` like Round3 (`ctx.int_step`/`out_fact`). `TBD=mat3tr(psibdx*RAD,thtbdx*RAD,phibdx*RAD)`; `TDI=cad_tdi84(...)`; `TBI=TBD@TDI` at init. Execute: stored-slope `integrate` of `TBID_NEW = (-skew(WBIB))@TBI`; orthonormalize as C++; Euler from TBD; `alphax`/`betax`/`alppx`/`phipx` from air-relative velocity as C++. CADAC sign `<0 → -1` else `+1` local helper. Not Flat6 quaternions. Skip `trcode` if absent.

- [ ] **Step 1:** climb ICs `thtbdx=2.5`, `lonx=latx=10`, `alt=10000`; after init TBD finite; one execute `WBIB=0` TBD stays orthonormal (`ortho_error` small)
- [ ] **Step 2–5:** implement, pass, commit `feat: Round6 kinematics DCM`

---

### Task 4: Round6 euler

**Files:** Modify `round6.py`; `Python/tests/unit/test_round6_euler.py`

**Interfaces:** Port `init_euler` / `euler`. `WBEB=[ppx,qqx,rrx]*RAD`; `WBIB=WBEB+TBI@(0,0,WEII3)`. Execute: inertia and `FMB` as C++ `Round6::euler` (read `euler.cpp`). `dt=ctx.int_step`. `IBBB` and `FMB` from store (plane/forces; not defined here).

- [ ] **Step 1:** zero FMB, identity TBI, ppx=10, dt=0.01; replica of C++ WBIB update; rates in deg/s on store
- [ ] **Step 2–5:** implement, pass, commit `feat: Round6 Euler equations`

---

### Task 5: Round6 newton init

**Files:** Modify `round6.py`; `Python/tests/unit/test_round6_newton_init.py`

**Interfaces:** Port `init_newton` **`minit==0` only**. `SBII=cad_in_geo84(lonx*RAD,latx*RAD,alt,time)`. VBEB from `alpha0x`/`beta0x`/`dvbe` as C++; `VBII` from TDI/WEII as C++. `minit!=0` → `ValueError`. Does not write `ABII` (or writes zeros as C++). `FSPB` defined here (newton-owned).

- [ ] **Step 1:** lonx=10, latx=10, alt=10000, dvbe=1000, alpha0x=2.5, beta0x=0 → `cad_geo84_in(SBII)` alt near 10000, `dvbe==1000`
- [ ] **Step 2–5:** implement, pass, commit `feat: Round6 newton initialization`

---

### Task 6: Round6 newton step

**Files:** Modify `round6.py`; `Python/tests/unit/test_round6_newton_step.py`

**Interfaces:** Port `Round6::newton`: `FSPB=FAPB/vmass`; `NEXT_ACC = TBI.T@FSPB + TGI.T@GRAVG` (C++ `~TBI*FSPB+~TGI*GRAVG`); stored-slope `integrate` VBII then SBII; `cad_geo84_in`; `VBED=TDI@(VBII-WEII@SBII)`; polar flight path. `vmass`/`FAPB` from store.

- [ ] **Step 1:** after Task 5 ICs, FAPB=0, dt=0.01; alt finite; SBII changes; replica NEXT_ACC matches ABEL/ABII
- [ ] **Step 2–5:** implement, pass, commit `feat: Round6 newton step`

---

### Task 7: Hyper6 aerodynamics (GHAME + der)

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/__init__.py`, `Python/src/cadac/vehicles/round6/hyper6/aero.py`
- Test: `Python/tests/unit/test_hyper6_aero.py`

**Interfaces:** Port `Hyper::init_aerodynamics`, `aerodynamics`, `aerodynamics_der`. `maero==1` GHAME tables from `ghame6_aero_deck.asc`. Else `ValueError`. `define` C++ `def_aerodynamics`. Tests parse ASC (JSONC in Task 16).

- [ ] **Step 1:** climb `maero=1`, alphax=2.5, vmach from 1000 m/s at 10 km; `cx`/`cz` finite; der fields finite; maero=2 raises
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER6 GHAME aerodynamics`

---

### Task 8: Hyper6 propulsion

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/propulsion.py`
- Test: `Python/tests/unit/test_hyper6_propulsion.py`

**Interfaces:** Port `Hyper::propulsion` `mprop` 0/1/2 only (climb is 2). Else `ValueError` (no 3/4). Parse `ghame6_prop_deck.asc`. Fuel/mass as C++.

- [ ] **Step 1:** mprop=2, vmach~3.3, qhold=200000; throttle in (0, thrtl_max]; mprop=0 thrust 0; mprop=4 raises
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER6 propulsion mprop 0-2`

---

### Task 9: Hyper6 actuator

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/actuator.py`
- Test: `Python/tests/unit/test_hyper6_actuator.py`

**Interfaces:** Port HYPER6 `actuator.cpp`. `mact` 0 limit-only and 2 `actuator_scnd`. Else `ValueError`. CADAC sign local helper. `dt=ctx.int_step`. Do not import `flat6._cadac_sign` or `np.sign`.

- [ ] **Step 1:** climb mact=2, delecx=1, dt=0.01; |delex|<=dlimx; mact=1 raises
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER6 second-order actuators`

---

### Task 10: control roll and rate SAS

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/control.py`
- Test: `Python/tests/unit/test_hyper6_control_roll.py`

**Interfaces:** Port HYPER6 `control_roll`, `control_roll_rate`, `control_pitch_rate`, `control_yaw_rate`. `define` full C++ `def_control`. `execute` pass until Task 12. `SMALL` module-level if C++ uses it; not in `cadac.constants`.

- [ ] **Step 1:** frozen aero/rates vs C++ rtol 1e-12
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER6 roll and rate SAS`

---

### Task 11: control_gamma

**Files:** Modify `control.py`; `Python/tests/unit/test_hyper6_control_gamma.py`

**Interfaces:** Port HYPER6 `control_gamma` (`pgam`,`wgam`,`zgam`,`thtvdcomx`) used by climb `maut=24`. `execute` still pass.

- [ ] **Step 1:** climb poles pgam=4 wgam=2 zgam=0.7, thtvdcomx=0; finite elevator vs C++
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER6 gamma controller`

---

### Task 12: maut dispatcher

**Files:** Modify `control.py`; `Python/tests/unit/test_hyper6_maut.py`

**Interfaces:** Port `Hyper::control` decode `mauty=maut//10`, `mautp=maut%10`. Modes `{0, 24}` only. Unknown including -1 → `ValueError`. Limit `|del*|` as C++. Update Task 10/11 tests that asserted execute pass.

- [ ] **Step 1:** maut=24 no error; maut=0 no write of new commands (or C++ equivalent); maut=-1 raises
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER6 maut dispatcher`

---

### Task 13: Hyper6 forces

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/forces.py`
- Test: `Python/tests/unit/test_hyper6_forces.py`

**Interfaces:** Port `Hyper::forces`. Writes `FAPB`/`FMB` only. `FSPB` newton-owned. If `FARCS`/`FMRCS` absent, treat as zero (RCS out of this plan). `FAPB=[pdynmc*refa*cx+thrust, pdynmc*refa*cy, pdynmc*refa*cz]`; moments as C++.

- [ ] **Step 1:** frozen aero/thrust vs C++ rtol 1e-12; FSPB not written
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER6 forces FAPB/FMB`

---

### Task 14: guidance stub

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/guidance.py`
- Test: `Python/tests/unit/test_hyper6_guidance_noop.py`

**Interfaces:** `define` C++ `def_guidance`. `mguide==0` return (climb default). Else `ValueError`. No LTG/line/pronav.

- [ ] **Step 1:** mguide=0 no raise; mguide=5 raises
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER6 guidance stub`

---

### Task 15: INS mins=0

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/ins.py`
- Test: `Python/tests/unit/test_hyper6_ins_ideal.py`

**Interfaces:** Port `init_ins` / `ins` `mins==0` paths: copy `TBI`→`TBIC`, `FSPB`→`FSPCB`, `SBII`→`SBIIC`, `VBII`→`VBIIC`, `WBIB`→`WBICB`, and the C++-written lon/lat/alt/Euler/flight-path computed names. `init` is no-op for mins=0. `mins!=0` → `ValueError`. Skip GPS/star names if absent.

- [ ] **Step 1:** mins=0 copies SBII to SBIIC; mins=1 raises
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER6 ideal INS`

---

### Task 16: HYPER6 vehicle + translate climb

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/vehicle.py`
- Modify: `Python/src/cadac/cli.py`
- Translate: `input_climb.asc` + `ghame6_aero_deck.asc` + `ghame6_prop_deck.asc` → `Python/cases/hyper6/`
- Test: `Python/tests/unit/test_hyper6_one_step.py`
- Retarget unknown-type tests that use `"HYPER6"` (`test_plane5_one_step.py`, `test_cruise3_one_step.py`) to `"AIM5"`

**Interfaces:** `Hyper6.type=="HYPER6"`. Modules in climb ASC order: kinematics, environment, aerodynamics, propulsion, ins, guidance, control, actuator, forces, newton, euler. Skip-if-exists on name collisions. `run_scenario` `HYPER6 -> Hyper6`; require both decks. Event `time>10` → `thtvdcomx=10`. `end_time` 60. Smoke: 0.1 s, `alt` near 10000.

- [ ] **Step 1:** smoke FAIL then PASS; JSONC type HYPER6; unknown AIM5 still raises
- [ ] **Step 2–5:** wire, pass, commit `feat: run HYPER6 from JSONC climb`

---

### Task 17: HYPER6 e2e golden (optional file)

**Files:** `Python/tests/e2e/test_hyper6_climb.py`

**Interfaces:** Skip without `tests/e2e/goldens/hyper6/plot.csv`. Else compare plot-flagged columns (`alt`/`vmach` if both present). Sentinel time=-1. Pattern FALCON6 e2e. Bump UPDATES subsubver.

- [ ] **Step 1:** skip without golden; with golden, alt at t=0
- [ ] **Step 2–5:** implement, pass, commit `test: HYPER6 e2e gate (skip without golden)`

---

## Self-review

- WGS84, Round6 env/kin/euler/newton, aero, prop, actuator, control SAS/gamma/maut 0/24, forces, guidance stub, INS mins=0, vehicle, e2e: each has a task
- Non-goals GPS/startrack/seeker/datalink/RCS/intercept/LTG/maero=2/mprop 3/4/minit=1: no tasks
- HYPER5 types unchanged; unknown token becomes AIM5
