# CADAC FALCON5 (PLANE / Flat3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Flat3 EOM and FALCON5 `PLANE` vehicle to `cadac`; unit-test every module; e2e vs CADAC CSV when a golden exists.

**Architecture:** Same kernel as HYPER3. Flat3 uses US76 + local-level `sbel`. Vehicle modules port `FALCON5/plane_modules.cpp` and `flat3_modules.cpp` as named numpy state.

**Tech Stack:** Python >= 3.11, numpy, pytest. Work in `Python/`.

**Spec:** `docs/superpowers/specs/2026-09-04-cadac-python-design.md`

**Depends on:** `docs/superpowers/plans/2026-09-04-cadac-kernel-hyper3.md` complete (kernel, look_up, US76, gravity, integrate, executive, JSONC).

## Global Constraints

- Copy from spec + kernel plan Global Constraints
- Do not change Cruise3/HYPER3 behavior; if a test regresses, fail this task
- Port FALCON5 C++ numerics; do not “simplify” autopilots
- E2E skipped if `Python/tests/e2e/goldens/falcon5/plot.csv` is absent
- Grok implementer + Grok reviewer; TDD; no Composer Fast / Kimi 3 / Fast mode
- JSONC type token is `PLANE` (CADAC `PLANE  FALCON5`)

## File map

- `Python/src/cadac/eom/flat3.py`
- `Python/src/cadac/vehicles/plane5/{aero,propulsion,forces,control,guidance,intercept,vehicle}.py`
- Tests under `Python/tests/unit/test_flat3_*.py`, `test_plane5_*.py`
- Cases: `Python/cases/falcon5/` from `CADAC_Simulations/FALCON5_250116/FALCON5/input_turning_to_IP.asc`
- C++: `CADAC_Simulations/FALCON5_250116/FALCON5/flat3_modules.cpp`, `plane_modules.cpp`

---

### Task 1: Flat3 environment

**Files:** Create `Python/src/cadac/eom/flat3.py`, `Python/tests/unit/test_flat3_environment.py`

**Interfaces:**
- Consumes: `atmosphere76`, `gravity`
- Produces: `Flat3Environment.name=="environment"`. `alt = -sbel[2]`. Writes `grav, rho, pdynmc, mach, vsound, press` as FALCON5 `Flat3::environment`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from cadac.kernel.state import Field, StateStore
from cadac.env.us76 import atmosphere76
from cadac.env.gravity import gravity
from cadac.eom.flat3 import Flat3Environment

def test_env_from_sbel():
    s = StateStore()
    env = Flat3Environment()
    env.define(s)
    s.set("SBEL", np.array([0.0, 0.0, -3500.0]))
    s.set("dvbe", 200.0)
    env.execute(s, None)
    rho, press, tempk = atmosphere76(3500.0)
    vsound = (1.4 * 287.053 * tempk) ** 0.5
    assert abs(s.get("alt") - 3500.0) < 1e-12 or abs((-s.get("SBEL")[2]) - 3500) < 1e-12
    assert abs(s.get("grav") - gravity(3500.0)) < 1e-12
    assert abs(s.get("rho") - rho) < 1e-12
    assert abs(s.get("mach") - abs(200.0 / vsound)) < 1e-12
```

If `alt` is newton-owned in C++ (`flat3[36]`), environment must still use `-SBEL[2]` internally as C++ does; do not invent a second altitude.

- [ ] **Step 2:** `cd Python && python -m pytest tests/unit/test_flat3_environment.py -v` — FAIL
- [ ] **Step 3:** Port `Flat3::environment`
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: Flat3 US76 environment`

---

### Task 2: Flat3 kinematics (timing)

**Files:** Modify `flat3.py`; create `Python/tests/unit/test_flat3_kinematics.py`

**Interfaces:** Produces `Flat3Kinematics` writing `time` and `event_time` from `SimContext` (C++ `kinematics` is timing only).

- [ ] **Step 1:**

```python
from types import SimpleNamespace
from cadac.kernel.state import StateStore
from cadac.eom.flat3 import Flat3Kinematics

def test_time_copy():
    s = StateStore()
    k = Flat3Kinematics()
    k.define(s)
    ctx = SimpleNamespace(sim_time=1.5, event_time=0.2, int_step=0.05)
    k.execute(s, ctx)
    assert s.get("time") == 1.5
    assert s.get("event_time") == 0.2
```

- [ ] **Step 2–5:** implement, pass, commit `feat: Flat3 kinematics timing`

---

### Task 3: Flat3 newton init

**Files:** Modify `flat3.py`; create `Python/tests/unit/test_flat3_newton_init.py`

**Interfaces:** Port `Flat3::init_newton`: SBEL from sbel1/2/3, VBEL from dvbe, psivlx, thtvlx (deg→rad), TBL from TVL/TBV.

- [ ] **Step 1:** sbel=[0,0,-3500], dvbe=200, psivlx=0, thtvlx=0 → `SBEL[2]==-3500`, `dvbe==200`, `alt==3500`
- [ ] **Step 2–5:** port init, pass, commit `feat: Flat3 newton initialization`

---

### Task 4: Flat3 newton step

**Files:** Modify `flat3.py`; create `Python/tests/unit/test_flat3_newton_step.py`

**Interfaces:** Port `Flat3::newton`: `NEXT_ACC = TBL.T @ FSPV + [0,0,grav]`; `NEXT_VEL=integrate(NEXT_ACC,ABEL,VBEL,dt)`; `SBEL=integrate(NEXT_VEL,VBEL,SBEL,dt)`; polar_from_cart; mat2tr; TBV from `phiavout`.

- [ ] **Step 1:** after init, FSPV=0, dt=0.05, one step; SBEL[2] increases toward 0 (falling under gravity); replica of NEXT_ACC in the test matches store ABEL
- [ ] **Step 2–5:** port newton, pass, commit `feat: Flat3 newton step`

---

### Task 5: Plane5 aerodynamics

**Files:** Create `Python/src/cadac/vehicles/plane5/aero.py`, `Python/tests/unit/test_plane5_aero.py`

**Interfaces:** Port `Plane::aerodynamics` in `plane_modules.cpp` (table names from `Falcon5_aero_deck.asc`). Translate deck to JSONC using existing translator.

- [ ] **Step 1:** look_up-based expected cl, cd at mach=0.6, alphax from store; assert execute matches
- [ ] **Step 2–5:** implement, pass, commit `feat: FALCON5 aerodynamics`

---

### Task 6: Plane5 propulsion

**Files:** Create `propulsion.py`, `test_plane5_propulsion.py`

**Interfaces:** Port `Plane::propulsion` / `init_propulsion` (`mprop`, Mach hold `gfthm`/`tfth`, mass/fuel). Use `integrate` for any first-order lags exactly as C++.

- [ ] **Step 1:** mprop=4, mach_com=0.6, one step; thrust and mass finite; mprop=0 → thrust 0
- [ ] **Step 2–5:** implement, pass, commit `feat: FALCON5 propulsion`

---

### Task 7: Plane5 forces

**Files:** Create `forces.py`, `test_plane5_forces.py`

**Interfaces:** Same FSPV construction as FALCON5 `Plane::forces` (pdynmc, cl, cd, area, thrust, mass, alphax, phimvx).

- [ ] **Step 1:** numeric FSPV vs the three C++ formulas (same pattern as Cruise3 forces test)
- [ ] **Step 2–5:** implement, pass, commit `feat: FALCON5 forces`

---

### Task 8: control_bank

**Files:** Create `control.py` with `control_bank` only; `test_plane5_control_bank.py`

**Interfaces:** Port `Plane::control_bank(phicx, int_step)` from `plane_modules.cpp` (first-order bank lag `tphi`, limiter `philimx`).

- [ ] **Step 1:** step a command through the C++ discrete update; compare phimvx
- [ ] **Step 2–5:** implement, pass, commit `feat: FALCON5 bank-angle control`

---

### Task 9: control_load

**Files:** Modify `control.py`; `test_plane5_control_load.py`

**Interfaces:** Port `control_load(ancomx, int_step)` (P-I `gacp`, `ta`, anposlimx/anneglimx).

- [ ] **Step 1:** unit test one step vs C++ equations in the function body
- [ ] **Step 2–5:** implement, pass, commit `feat: FALCON5 load-factor control`

---

### Task 10: control_altitude

**Files:** Modify `control.py`; `test_plane5_control_altitude.py`

**Interfaces:** Port `control_altitude(altcom, phimvx)` (`gh`, `gv`, `altdlim`).

- [ ] **Step 1–5:** TDD one-step vs C++, commit `feat: FALCON5 altitude control`

---

### Task 11: control_heading and control_flightpath

**Files:** Modify `control.py`; `test_plane5_control_heading.py`

**Interfaces:** Port `control_heading(psivlcx)` and `control_flightpath(thtvgcx, phimv)` as two functions, two tests in the same file.

- [ ] **Step 1–5:** TDD both, commit `feat: FALCON5 heading and flight-path control`

---

### Task 12: control_lateral and mcontrol dispatcher

**Files:** Modify `control.py`; `test_plane5_mcontrol.py`

**Interfaces:** Port `control_lateral(alcomx)` and `Plane::control` switch on `mcontrol` (modes used in `input_turning_to_IP.asc`: 46 and 44). Read the `if (mcontrol==` chain in `plane_modules.cpp` and implement only those modes plus a `ValueError` for unknown modes.

- [ ] **Step 1:** mcontrol=46 with turning-to-IP gains produces finite ancomx/phimvx; unknown mcontrol raises
- [ ] **Step 2–5:** implement, pass, commit `feat: FALCON5 mcontrol dispatcher`

---

### Task 13: guidance_point and wp_flag

**Files:** Create `guidance.py`; `test_plane5_guidance_point.py`

**Interfaces:** Port `guidance_point` and `wp_flag` update from `Plane::guidance`.

- [ ] **Step 1:** waypoint swel=[5000,2000], vehicle at origin heading 0 → finite commands; wp_flag becomes -1 when CADAC would
- [ ] **Step 2–5:** implement, pass, commit `feat: FALCON5 point guidance`

---

### Task 14: guidance_line

**Files:** Modify `guidance.py`; `test_plane5_guidance_line.py`

**Interfaces:** Port `guidance_line` (`line_gain`, `nl_gain_fact`, `decrement`, `psiflx`, `thtflx`).

- [ ] **Step 1–5:** TDD vs C++, commit `feat: FALCON5 line guidance`

---

### Task 15: intercept stop_run

**Files:** Create `intercept.py`; `test_plane5_intercept.py`

**Interfaces:** Port `Plane::intercept`: `stop_run==1` sets health 0 / ends run the way C++ does (read the function).

- [ ] **Step 1:** stop_run=1 → packet status 0
- [ ] **Step 2–5:** implement, pass, commit `feat: FALCON5 intercept stop_run`

---

### Task 16: PLANE vehicle + translate turning case

**Files:** Create `vehicle.py`; extend `run_scenario` type map `PLANE -> Plane5`; translate `input_turning_to_IP.asc` + decks to `Python/cases/falcon5/`; `test_plane5_one_step.py`

**Interfaces:** Module order from that ASC. One-second run: alt near 3000, time=1.

- [ ] **Step 1:** failing one-second smoke test
- [ ] **Step 2–5:** wire, pass, commit `feat: run FALCON5 PLANE from JSONC`

---

### Task 17: FALCON5 e2e golden (optional file)

**Files:** `Python/tests/e2e/test_falcon5_turning.py`

**Interfaces:** If `tests/e2e/goldens/falcon5/plot.csv` missing, `pytest.skip`. Else compare plot columns with spec CSV tolerances.

- [ ] **Step 1:** test skips without golden; with golden, compare `alt` at t=0
- [ ] **Step 2–5:** implement skip/compare, commit `test: FALCON5 e2e gate (skip without golden)`

---

## Self-review

Spec Plane5 modules all have tasks. Kernel types unchanged. e2e skip is explicit.
