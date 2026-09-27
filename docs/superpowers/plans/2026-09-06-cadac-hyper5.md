# CADAC HYPER5 (HYPER5 / TARGET3 / SATELLITE3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run HYPER5 Demo 4.7 from JSONC (`HYPER5` + `TARGET3`); register `SATELLITE3`; unit-test each C++ module; e2e vs CADAC CSV when a golden exists.

**Architecture:** Reuse Round3 (ISO 62). Port HYPER5 `hyper_modules.cpp` / `target_modules.cpp` / `satellite_modules.cpp`. Kernel: skip missing per-vehicle modules; decks only when the type needs them; plot slot 0. Do not import Plane5/Cruise3 vehicle classes.

**Tech Stack:** Python >= 3.11, numpy, pytest. Work in `Python/`.

**Spec:** `docs/superpowers/specs/2026-09-06-cadac-hyper5-design.md`

**Depends on:** first-slice kernel + Round3 (`docs/superpowers/plans/2026-09-04-cadac-kernel-hyper3.md`) complete.

## Global Constraints

- Spec + parent `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
- Do not regress HYPER3 e2e or FALCON5/FALCON6 unit tests
- Port `CADAC_Simulations/HYPER5_250113/HYPER5/` numerics; do not simplify autopilots
- JSONC types `HYPER5`, `TARGET3`, `SATELLITE3`; `CRUISE3`/`PLANE`/`PLANE6` stay registered
- Seeker/targeting identify packets by `Packet.type`, not CADAC `id.find("t")`
- E2E skip if `Python/tests/e2e/goldens/hyper5/plot.csv` absent (do not create it)
- Grok implementer `cursor-grok-4.6-high` + Grok reviewer; TDD; no Composer Fast / Kimi / Fast
- Protocol: `define(vehicle)` / `execute(vehicle, ctx)` on `vehicle.store`
- Unit `rtol=1e-12`, `atol=1e-14`; CSV e2e `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`

## File map

- `Python/src/cadac/math/earth.py` — add `cadine`
- `Python/src/cadac/kernel/executive.py` — skip missing modules
- `Python/src/cadac/cli.py` — deck policy, plot slot 0, type map
- `Python/src/cadac/vehicles/round3/hyper5/{aero,propulsion,forces,control,guidance,seeker,intercept,targeting,target,satellite,vehicle}.py`
- Tests `Python/tests/unit/test_cadine.py`, `test_executive.py`, `test_hyper5_*.py`, `test_target3_*.py`, `test_satellite3_*.py`
- Case `Python/cases/hyper5/` from `input.asc` + `hyper5_aero_deck.asc`
- C++: `hyper_modules.cpp`, `target_modules.cpp`, `satellite_modules.cpp`, `utility_functions.cpp` (`cadine`)

---

### Task 1: cadine

**Files:**
- Modify: `Python/src/cadac/math/earth.py`
- Test: `Python/tests/unit/test_cadine.py`

**Interfaces:**
- Consumes: `REARTH`, `WEII3` from `cadac.constants`
- Produces: `cadine(lon_rad, lat_rad, alt_m, time) -> ndarray (3,)` CADAC `cadine` in HYPER5 `utility_functions.cpp`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from cadac.constants import REARTH, WEII3
from cadac.math.earth import cadine

def test_cadine_equator_t0():
    sbii = cadine(0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(sbii, [REARTH, 0.0, 0.0], atol=1e-9)

def test_cadine_celestial_longitude():
    lon, lat, alt, time = 0.1, 0.2, 1000.0, 10.0
    rad = alt + REARTH
    cel = lon + WEII3 * time
    expected = np.array([
        rad * np.cos(lat) * np.cos(cel),
        rad * np.cos(lat) * np.sin(cel),
        rad * np.sin(lat),
    ])
    np.testing.assert_allclose(cadine(lon, lat, alt, time), expected, rtol=1e-12, atol=1e-14)
```

- [ ] **Step 2:** `cd Python && python -m pytest tests/unit/test_cadine.py -v` — FAIL
- [ ] **Step 3:** Port C++ `cadine`
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: CADAC cadine lon/lat/alt to inertial`

---

### Task 2: run_loop skip missing modules; plot slot 0

**Files:**
- Modify: `Python/src/cadac/kernel/executive.py`, `Python/src/cadac/cli.py`
- Test: `Python/tests/unit/test_executive.py`, `Python/tests/unit/test_plot_slot0.py`

**Interfaces:**
- Consumes: existing `run_loop(vehicles, modules_by_vehicle, module_order, ...)`
- Produces: if `name` not in that vehicle's modules, skip (no `KeyError`). `run_scenario` `on_step` appends plot rows only when `ctx.vehicle_slot == 0`. Single-vehicle CRUISE3/PLANE/PLANE6 plot_rows unchanged.

- [ ] **Step 1: Write the failing test**

```python
from cadac.kernel.executive import run_loop
from cadac.kernel.module import DummyModule

def test_skip_module_absent_on_vehicle():
    a = _Vehicle()
    b = _Vehicle()
    dummy = DummyModule()
    run_loop(
        vehicles=[a, b],
        modules_by_vehicle={a: [dummy], b: []},
        module_order=["dummy"],
        end_time=0.0,
        int_step=0.1,
    )
    assert a.store.get("time") == 0.0
```

Also: two dummy vehicles with the same `dummy` module; after `run_scenario` (or a small helper wrapping `on_step` logic), `len(plot_rows)` equals plot ticks, not `2 * ticks`. Prefer a unit test of the slot-0 filter without a full JSONC case. Do not change CRUISE3 `_plot_columns`.

- [ ] **Step 2:** FAIL
- [ ] **Step 3:** `named.get(name)` skip; `on_step` only slot 0
- [ ] **Step 4:** PASS including existing `test_executive.py` and HYPER3 e2e
- [ ] **Step 5: Commit** `feat: skip missing vehicle modules; plot vehicle 0`

---

### Task 3: Hyper5 aerodynamics

**Files:**
- Create: `Python/src/cadac/vehicles/round3/hyper5/__init__.py`, `Python/src/cadac/vehicles/round3/hyper5/aero.py`
- Test: `Python/tests/unit/test_hyper5_aero.py`

**Interfaces:**
- Consumes: `Datadeck.look_up` 2D; `RAD` from constants
- Produces: `Hyper5Aero(deck).name=="aerodynamics"`. `define` registers C++ `def_aerodynamics` (`cl`,`cd`,`cl_ov_cd`,`area`,`cla`,`cn`,`ca`). Does not define `mach`/`alphax`. `execute` ports `Hyper::aerodynamics`: `cn=look_up("cn_rr3x_vs_alphax_mach",alphax,mach)`, `ca=look_up("ca_rr3x_vs_alphax_mach",alphax,mach)`; `cd=cn*sin(alpha)+ca*cos(alpha)`; `cl=cn*cos(alpha)-ca*sin(alpha)`; `cla` from ±2 deg as C++. Parse `CADAC_Simulations/HYPER5_250113/HYPER5/hyper5_aero_deck.asc` in tests (JSONC case is Task 20).

- [ ] **Step 1:** mach=4, alphax=2, area=11.6986; `cl`/`cd`/`cla` match look_up then C++ formulas, rtol 1e-12
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 Roadrunner aerodynamics`

---

### Task 4: Hyper5 propulsion

**Files:**
- Create: `Python/src/cadac/vehicles/round3/hyper5/propulsion.py`
- Test: `Python/tests/unit/test_hyper5_propulsion.py`

**Interfaces:**
- Consumes: `integrate`, `AGRAV`, Datadeck. Constructor takes Datadeck or `None` (`mprop==0` must not look up).
- Produces: `Hyper5Propulsion`. `define` C++ `def_propulsion`. `initialize` sets `mass=mass0`. `execute`: `mprop==0` thrust=0, `fmassd=0`; `1` fixed `phi_const`; `2` q-hold with `tlag` integrate and phi clips; `3` keep previous `phi`. Tables `cin_vs_alphax_mach`, `spi_vs_mach_phi_alphax`. Thrust `0.0676*phi*spi*AGRAV*rho*dvbe*cin*aintake` as C++. Fuel cutoff `fmassr<=0` → `mprop=0`. Else `ValueError`. No forces.

- [ ] **Step 1:** mprop=0 no thrust; mprop=1 with parsed `hyper5_prop_deck.asc` if present else skip look_up path with a tiny fixture table; mprop=-1 raises. If the ASC exists, one-step mprop=1 thrust vs C++ formula rtol 1e-12
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 propulsion mprop 0-3`

---

### Task 5: Hyper5 forces

**Files:**
- Create: `Python/src/cadac/vehicles/round3/hyper5/forces.py`
- Test: `Python/tests/unit/test_hyper5_forces.py`

**Interfaces:**
- Produces: `Hyper5Forces.name=="forces"`. `define` `FSPV` vec out plot only. Skip-if-name-exists (Round3 newton does not define FSPV). `execute` C++ `Hyper::forces`: `fspv1=(-pdynmc*area*cd+thrust*cos(alpha))/mass`, `fspv2=sin(phimv)*(pdynmc*area*cl+thrust*sin(alpha))/mass`, `fspv3=-cos(phimv)*(pdynmc*area*cl+thrust*sin(alpha))/mass`.

- [ ] **Step 1:** pdynmc=72000, area=11.6986, cd=0.05, cl=0.2, thrust=0, mass=1352, alphax=-1.5, phimvx=0 and 90 vs C++ formulas
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 Hyper forces FSPV`

---

### Task 6: control_bank

**Files:**
- Create: `Python/src/cadac/vehicles/round3/hyper5/control.py`
- Test: `Python/tests/unit/test_hyper5_control_bank.py`

**Interfaces:**
- Produces: `Hyper5Control`. `define` bank fields from C++ `def_control` used by `control_bank` (`phimvx`, `phicx`, `philimx`, `tphi`, bank state as C++). `control_bank(vehicle, phicx, int_step)` ports `Hyper::control_bank`. `execute` is pass until Task 10 dispatcher.

- [ ] **Step 1:** Demo 4.7 `philimx=70`, `tphi=1`, int_step=0.05; one- and two-step lag vs C++
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 bank-angle control`

---

### Task 7: control_load

**Files:** Modify `control.py`; `Python/tests/unit/test_hyper5_control_load.py`

**Interfaces:** Port `Hyper::control_load`. `define` adds load fields (`anposlimx`, `anneglimx`, `gacp`, `ta`, `alphax` clips, P-I states as C++). `execute` still pass.

- [ ] **Step 1:** Demo 4.7 `gacp=10`, `ta=0.8`, `anposlimx=2`, `anneglimx=-2`; one-step vs C++ replica
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 load-factor control`

---

### Task 8: control_altitude

**Files:** Modify `control.py`; `Python/tests/unit/test_hyper5_control_altitude.py`

**Interfaces:** Port `Hyper::control_altitude` (`gh`, `gv`, `altdlim`, `altcom`). Used by mcontrol 6/16/36/46.

- [ ] **Step 1:** fly-out gains `gh=0.2`, `gv=0.3`, `altdlim=50`, `altcom=24000`; one-step vs C++
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 altitude control`

---

### Task 9: control_heading and control_flightpath

**Files:** Modify `control.py`; `Python/tests/unit/test_hyper5_control_heading.py`

**Interfaces:** Port `Hyper::control_heading` and `Hyper::control_flightpath` from `hyper_modules.cpp` (not Plane5). Two tests in one file.

- [ ] **Step 1:** heading wrap `|psivgcx|>135` opposite-sign; flight-path alphax clip vs C++
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 heading and flight-path control`

---

### Task 10: control_lateral and mcontrol dispatcher

**Files:** Modify `control.py`; `Python/tests/unit/test_hyper5_mcontrol.py`

**Interfaces:** Port `Hyper::control_lateral` and `Hyper::control` dispatcher. Modes `{0,3,4,6,16,36,40,44}` as C++ (`mcontrol 03` is int 3). Then `TBV=cadtbv(phimvx*RAD, alphax*RAD)`, `TBG=TBV@TVG`. Unknown → `ValueError`. `0` zeros `phimvx`/`alphax`. Update Tasks 6–9 tests that asserted `execute` pass.

- [ ] **Step 1:** mcontrol=44 Demo 4.7 `alcomx=0.5` finite `phimvx`/`alphax`; mcontrol=-1 raises; mcontrol=0 zeros
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 mcontrol dispatcher`

---

### Task 11: guidance_point

**Files:**
- Create: `Python/src/cadac/vehicles/round3/hyper5/guidance.py`
- Test: `Python/tests/unit/test_hyper5_guidance_point.py`

**Interfaces:**
- Consumes: `cadine`, `polar_from_cart`, `mat2tr`, Round3 `TIG`/`SBII`/`VBEG`
- Produces: `Hyper5Guidance`. `define` C++ `def_guidance` point+dispatcher fields (`mguidance`, `wp_lonx`/`wp_latx`/`wp_alt`, `point_gain`, `wp_flag`, …). `guidance_point(vehicle)` ports `Hyper::guidance_point`. `execute` pass until Task 13.

- [ ] **Step 1:** waypoint lon/lat/alt offset from vehicle; finite APGV; `wp_flag` CADAC sign as C++
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 point guidance`

---

### Task 12: guidance_pronav

**Files:** Modify `guidance.py`; `Python/tests/unit/test_hyper5_guidance_pronav.py`

**Interfaces:** Port `Hyper::guidance_pronav`: `APNB=skew(WOEB)@UTBB*(pronav_gain*closing_speed)-TBG@(GRAV_G)` with `GRAV_G=[0,0,grav+bias]`. Seeker outputs `WOEB`/`UTBB`/`closing_speed`/`range_go` registered by the test (seeker Task 14).

- [ ] **Step 1:** Demo 4.7 `pronav_gain=3.5`, `bias=5`; frozen seeker vectors; APNB vs C++ rtol 1e-12
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 pro-nav guidance`

---

### Task 13: guidance_arc and mguidance dispatcher

**Files:** Modify `guidance.py`; `Python/tests/unit/test_hyper5_mguidance.py`

**Interfaces:** Port `Hyper::guidance_arc` (returns `phicx`). `execute`: `0` zeros `alcomx`/`ancomx` and return without writing (or write zeros as C++); `44` point → `alcomx`/`ancomx`; `66` pronav; `70` arc writes `phicx`. Else `ValueError`. Clips via control limiters as C++ `Hyper::guidance`.

- [ ] **Step 1:** mguidance=66 writes finite commands; 0 no raise; 30/33/99 raise
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 mguidance dispatcher`

---

### Task 14: seeker

**Files:**
- Create: `Python/src/cadac/vehicles/round3/hyper5/seeker.py`
- Test: `Python/tests/unit/test_hyper5_seeker.py`

**Interfaces:** Port `Hyper::seeker` / `seeker_grnd_ranges`. `mseeker==0` return. `1` acquire if ground range `< acq_range`, then `mseeker=3`. `3` track: `range_go`, `STBG`, `WOEB`, `closing_speed`, `UTBB`, `targ_com_slot`. Else `ValueError`. Target packets: `ctx.combus[i].type=="TARGET3"`; kinematics from `packet.vars` names `lonx`,`latx`,`alt`,`psivgx`,`thtvgx`,`dvbe`,`VBEG`,`SBII`. Demo 4.7 `acq_range=6000`.

- [ ] **Step 1:** two packets Hyper+Target; range inside 6000 → mseeker becomes 3; mseeker=0 no write; mseeker=99 raises
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 seeker acquire and track`

---

### Task 15: Hyper intercept

**Files:**
- Create: `Python/src/cadac/vehicles/round3/hyper5/intercept.py`
- Test: `Python/tests/unit/test_hyper5_intercept.py`

**Interfaces:** Port `Hyper::intercept` without `cout`/`exit`. `halt` and `write` → `vehicle.health=0`, `ctx.combus[slot].status=0`. Ground `alt<=0` same. `mseeker==3` and `range_go<1000` closest-approach interpolation as C++ (status 0 on intercept). `write` latch. No `sys.exit`.

- [ ] **Step 1:** halt=1 write=1 → health 0 and packet status 0; halt=0 no kill
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 Hyper intercept halt/hit`

---

### Task 16: targeting stub (mtargeting 0)

**Files:**
- Create: `Python/src/cadac/vehicles/round3/hyper5/targeting.py`
- Test: `Python/tests/unit/test_hyper5_targeting_noop.py`

**Interfaces:** `define` C++ `def_targeting`. `execute`: `mtargeting==0` return; else `ValueError` until Task 19. Include module on Hyper5 vehicle later so define always runs.

- [ ] **Step 1:** mtargeting=0 no raise/no write; mtargeting=1 raises (until Task 19)
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 targeting stub`

---

### Task 17: Target3

**Files:**
- Create: `Python/src/cadac/vehicles/round3/hyper5/target.py`
- Test: `Python/tests/unit/test_target3_forces.py`, `Python/tests/unit/test_target3_intercept.py`

**Interfaces:**
- Produces: `Target3.type=="TARGET3"`. Modules: `Round3Environment`, `Round3Newton`, `Target3Forces`, `Target3Intercept`. Constructor `(name, events=None)` — no decks. `com_names` from store fields with `"com"` in outputs.
- Forces: C++ `Target::forces` Coriolis + centrifugal − gravity + `fwd_accel`/`side_accel`. Writes `FSPV`, `CORIO_V`, `CENTR_V`.
- Intercept: copy `ctx.combus[slot].status` to `targ_health`.

- [ ] **Step 1:** fwd=side=0; after newton init at lon/lat/alt; FSPV finite (apparent accel). Intercept: combus status 0 → targ_health 0
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 TARGET3 forces and intercept`

---

### Task 18: Satellite3

**Files:**
- Create: `Python/src/cadac/vehicles/round3/hyper5/satellite.py`
- Test: `Python/tests/unit/test_satellite3_forces.py`

**Interfaces:** `Satellite3.type=="SATELLITE3"`. Modules: Round3 env/newton + forces. `sat_thrust`, `sat_mass` default 100. `FSPV=[sat_thrust/sat_mass, 0, 0]`. Constructor `(name, events=None)`.

- [ ] **Step 1:** sat_thrust=0, sat_mass=100 → FSPV[0]==0; sat_thrust=100 → FSPV[0]==1
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 SATELLITE3 forces`

---

### Task 19: targeting mtargeting==1

**Files:** Modify `targeting.py`; `Python/tests/unit/test_hyper5_targeting.py`. Update Task 16 test: mtargeting=1 no longer raises.

**Interfaces:** Port `Hyper::targeting` / `targeting_satellite` / `targeting_grnd_ranges`. Visibility from Satellite packets (`type=="SATELLITE3"`). Closest Target (`type=="TARGET3"`) by ground range. Writes `wp_lonx`/`wp_latx`/`wp_alt` from that target as C++. `LARGE=1e10` module-level (C++ `global_constants.hpp`); do not add to `cadac.constants`. Else `ValueError`.

- [ ] **Step 1:** one Hyper, one Target, one Satellite on combus; mtargeting=1 sets waypoint to target lon/lat/alt; mtargeting=2 raises
- [ ] **Step 2–5:** implement, pass, commit `feat: HYPER5 satellite targeting`

---

### Task 20: HYPER5 vehicle + TARGET3 registry + translate Demo 4.7

**Files:**
- Create: `Python/src/cadac/vehicles/round3/hyper5/vehicle.py`
- Modify: `Python/src/cadac/cli.py`
- Translate: `CADAC_Simulations/HYPER5_250113/HYPER5/input.asc` + `hyper5_aero_deck.asc` → `Python/cases/hyper5/`
- Test: `Python/tests/unit/test_hyper5_one_step.py`
- Retarget unknown-type tests that still use `"HYPER6"` as unknown — keep `"HYPER6"` until HYPER6 plan; do not use `"HYPER5"` as unknown

**Interfaces:**
- `Hyper5.type=="HYPER5"`. Constructor `(name, aero_deck, prop_deck, events=None)` with `prop_deck` allowed `None`. Modules in Demo 4.7 ASC order plus targeting (define-only if MODULES omits it): environment, aerodynamics, propulsion, forces, newton, seeker, guidance, control, intercept, targeting.
- `run_scenario`: map `HYPER5`/`TARGET3`/`SATELLITE3`. Require aero for `HYPER5`; prop if `prop_deck` present. `TARGET3`/`SATELLITE3`: `Target3(name, events)` / `Satellite3(name, events)` — no decks. Existing types still require both decks.
- Translate `end_time` 25. Params include `mprop=0`, `mcontrol=44`, `mguidance=66`, `mseeker=1`, `acq_range=6000`.
- Smoke: `run_scenario` 0.05 s (`end_time` override via a tiny fixture copy or one-step by temporarily not required — prefer constructing vehicles in-test and `run_loop` 0.05 s). `hbe`/`alt` finite; both vehicles health 1.

- [ ] **Step 1:** smoke FAIL then PASS; translated JSONC loads; Target has no aero_deck
- [ ] **Step 2–5:** wire, pass, commit `feat: run HYPER5 and TARGET3 from JSONC`

---

### Task 21: HYPER5 e2e golden (optional file)

**Files:** `Python/tests/e2e/test_hyper5_pronav.py`

**Interfaces:** Skip if `tests/e2e/goldens/hyper5/plot.csv` absent. Else `run_scenario` on `Python/cases/hyper5/input.jsonc` (or the translated stem). Compare Hyper (slot 0) plot columns present in both; skip sentinel `time=-1`; CSV rtol=1e-5, atol=max(1e-6, 5e-6*|g|). Pattern `test_falcon6_gamma.py`. Bump UPDATES subsubver.

- [ ] **Step 1:** skip helper without golden; with golden, `alt` at t=0
- [ ] **Step 2–5:** implement, pass, commit `test: HYPER5 e2e gate (skip without golden)`

---

## Self-review

- cadine, kernel skip, plot slot 0, aero/prop/forces, all listed mcontrol/mguidance/mprop/mseeker, seeker, intercept, targeting 0 and 1, Target3, Satellite3, vehicle, e2e: each has a task
- Unused C++ guidance 30/33/40/43: ValueError (spec)
- No HYPER6 / Round6
- CRUISE3 plot_row default unchanged
