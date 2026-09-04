# CADAC FALCON6 (PLANE6 / Flat6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Flat6 EOM and FALCON6 `PLANE6` vehicle; unit-test each C++ module; e2e vs CADAC CSV when a golden exists.

**Architecture:** Kernel unchanged. Flat6: US76, body velocity, Euler rates, TBL kinematics. Vehicle modules port `FALCON6/*.cpp` (aerodynamics, propulsion, actuator, control, forces).

**Tech Stack:** Python >= 3.11, numpy, pytest. Work in `Python/`.

**Spec:** `docs/superpowers/specs/2026-09-04-cadac-python-design.md`

**Depends on:** kernel+HYPER3 plan complete. FALCON5 plan not required except shared `run_scenario` registry.

## Global Constraints

- Spec + kernel Global Constraints
- Do not regress HYPER3 e2e or FALCON5 unit tests
- Port FALCON6 numerics from `CADAC_Simulations/FALCON6_250201/FALCON6/`
- 3D `look_up` and `mat3tr` live in kernel modules (`tables/lookup.py`, `math/frames.py`)
- JSONC type `PLANE6`
- E2E skip if `Python/tests/e2e/goldens/falcon6/plot.csv` absent
- Grok implementer + Grok reviewer; TDD; no Composer Fast / Kimi 3 / Fast mode
- First e2e vehicle count: prefer a 1-vehicle case (`input_gamma.asc` or `input_roll.asc`) over the 11-vehicle fan

## File map

- `Python/src/cadac/eom/flat6.py`
- `Python/src/cadac/vehicles/plane6/{aero,propulsion,actuator,control,forces,guidance,vehicle}.py`
- Extend `asc_deck.py` 3DIM; `look_up` 3-arg
- Tests `test_lookup_3d.py`, `test_mat3tr.py`, `test_flat6_*.py`, `test_plane6_*.py`
- C++: `environment.cpp`, `kinematics.cpp`, `euler.cpp`, `newton.cpp`, `aerodynamics.cpp`, `propulsion.cpp`, `actuator.cpp`, `control.cpp`, `forces.cpp`

---

### Task 1: mat3tr

**Files:** Modify `Python/src/cadac/math/frames.py`; create `Python/tests/unit/test_mat3tr.py`

**Interfaces:** `mat3tr(psi, tht, phi) -> (3,3)` CADAC `mat3tr` in HYPER3/FALCON6 `utility_functions.cpp`.

- [ ] **Step 1:**

```python
import numpy as np
from cadac.math.frames import mat3tr

def test_zero_angles_identity():
    np.testing.assert_allclose(mat3tr(0.0, 0.0, 0.0), np.eye(3), atol=1e-14)
```

Also assert `mat3tr(0,0,0.1)[1,2] == np.cos(0)*np.sin(0.1)` from C++ `assign_loc(1,2,ctht*sphi)`.

- [ ] **Step 2:** FAIL `pytest tests/unit/test_mat3tr.py -v`
- [ ] **Step 3:** Port C++
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: CADAC mat3tr Euler DCM`

---

### Task 2: Parse 3DIM ASC + 3D look_up

**Files:** Modify `asc_deck.py`, `lookup.py`; tests `test_asc_deck_3d.py`, `test_lookup_3d.py`

**Interfaces:** Parse `f16_prop_deck.asc` table `ff_vs_thrust_alt_mach`. `look_up(name,x1,x2,x3)` trilinear as C++ 3D interpolate (constant upper, slope lower). Packing: match FALCON6 `read_tables`.

- [ ] **Step 1:** table shape consistent with NX1/NX2/NX3 in that file; look_up at a grid corner returns the corner value
- [ ] **Step 2–5:** implement both, pass, commit `feat: CADAC 3D table parse and look_up`

---

### Task 3: Flat6 environment

**Files:** Create `flat6.py` environment class; `test_flat6_environment.py`

**Interfaces:** Port `Flat6::environment` (US76, grav, mach as `vmach`, wind `mwind==0` path first). `mwind!=0` in a later task if the gamma case uses it (gamma fan uses default 0 — implement mwind=0 only; other values `ValueError`).

- [ ] **Step 1:** hbe=1000, dvbe=180, mwind=0 → rho/press match atmosphere76(1000)
- [ ] **Step 2–5:** implement, pass, commit `feat: Flat6 environment`

---

### Task 4: Flat6 kinematics

**Files:** Modify `flat6.py`; `test_flat6_kinematics.py`

**Interfaces:** Port `init_kinematics` / `kinematics` (`TBL` from Euler, alpha/beta from VBEB). Read `kinematics.cpp`.

- [ ] **Step 1:** alpha0x=1, beta0x=0, thtblx=1 → TBL finite, alphax ≈ 1 after init
- [ ] **Step 2–5:** port, pass, commit `feat: Flat6 kinematics`

---

### Task 5: Flat6 euler

**Files:** Modify `flat6.py`; `test_flat6_euler.py`

**Interfaces:** Port `init_euler` / `euler` (`WBEB` from ppx/qqx/rrx, integrate with moments). Read `euler.cpp`.

- [ ] **Step 1:** zero moments, nonzero ppx → rates hold or decay only as C++ writes; one-step replica in test
- [ ] **Step 2–5:** port, pass, commit `feat: Flat6 Euler equations`

---

### Task 6: Flat6 newton

**Files:** Modify `flat6.py`; `test_flat6_newton.py`

**Interfaces:** Port `init_newton` / `newton` (`VBEB` from alpha0/beta0/dvbe, `SBEL`, integrate). Read `newton.cpp`.

- [ ] **Step 1:** sbel3=-1000, dvbe=180, one step dt=0.001; hbe near 1000
- [ ] **Step 2–5:** port, pass, commit `feat: Flat6 newton`

---

### Task 7: Plane6 aerodynamics (tables)

**Files:** Create `aero.py`; `test_plane6_aero.py`

**Interfaces:** Port `Plane::aerodynamics` (not `_der` yet). Translate `f16_aero_deck.asc`.

- [ ] **Step 1:** look_up Cx/Cz (or whatever the C++ function reads) at alpha=1 deg, elev=0; execute matches
- [ ] **Step 2–5:** implement, pass, commit `feat: F-16 aerodynamics tables`

---

### Task 8: aerodynamics_der

**Files:** Modify `aero.py`; `test_plane6_aero_der.py`

**Interfaces:** Port `Plane::aerodynamics_der` exactly.

- [ ] **Step 1:** derivatives finite at the Task 7 flight condition; one C++ formula replicated
- [ ] **Step 2–5:** implement, pass, commit `feat: F-16 aero derivatives`

---

### Task 9: Plane6 propulsion

**Files:** Create `propulsion.py`; `test_plane6_propulsion.py`

**Interfaces:** Port `Plane::propulsion` (`mprop` 0/1/2, 3D fuel flow, Mach hold `gmach`).

- [ ] **Step 1:** mprop=2, vmachcom=0.6, one step; throttle in (0,1]
- [ ] **Step 2–5:** implement, pass, commit `feat: F-16 propulsion`

---

### Task 10: actuator

**Files:** Create `actuator.py`; `test_plane6_actuator.py`

**Interfaces:** Port `actuator_scnd` / `Plane::actuator` (`mact==2` second order, `dlimx`, `ddlimx`, `wnact`, `zetact`).

- [ ] **Step 1:** step command 1 deg, dt=0.001; output lags command, respects dlimx
- [ ] **Step 2–5:** implement, pass, commit `feat: F-16 second-order actuators`

---

### Task 11: control_roll and rate SAS

**Files:** Create `control.py`; `test_plane6_control_roll.py`

**Interfaces:** Port `control_roll`, `control_roll_rate`, `control_pitch_rate`, `control_yaw_rate` from `control.cpp`.

- [ ] **Step 1:** unit test each function with zero state vs C++ algebra
- [ ] **Step 2–5:** implement, pass, commit `feat: F-16 roll and rate SAS`

---

### Task 12: control_gamma (flight-path)

**Files:** Modify `control.py`; `test_plane6_control_gamma.py`

**Interfaces:** Port `control_gamma` / gamma closed-loop (`pgam`, `wgam`, `zgam`, `thtvlcomx`) used by `input_gamma.asc` (`maut=24`).

- [ ] **Step 1:** thtvlcomx=1 deg, one step, elevator command finite
- [ ] **Step 2–5:** implement, pass, commit `feat: F-16 gamma controller`

---

### Task 13: maut dispatcher

**Files:** Modify `control.py`; `test_plane6_maut.py`

**Interfaces:** Port `Plane::control` decode of `maut` (`maut=|mauty|mautp|`). Implement modes required by `input_gamma.asc` and `input_roll.asc`; unknown → `ValueError`.

- [ ] **Step 1:** maut=24 runs without error; maut=-1 raises
- [ ] **Step 2–5:** implement, pass, commit `feat: F-16 maut dispatcher`

---

### Task 14: Plane6 forces

**Files:** Create `forces.py`; `test_plane6_forces.py`

**Interfaces:** Port `Plane::forces` (body-axis aero + thrust → FAPB/FSPB as C++).

- [ ] **Step 1:** numeric FSPB vs C++ expressions at a frozen aero/thrust point
- [ ] **Step 2–5:** implement, pass, commit `feat: F-16 forces`

---

### Task 15: guidance stub

**Files:** Create `guidance.py`; `test_plane6_guidance_noop.py`

**Interfaces:** If JSONC lists `guidance` with exec, `Plane::guidance` for the gamma case is unused — port empty/safe C++ body (no new laws). Test: execute does not raise.

- [ ] **Step 1–5:** implement no-op matching C++, commit `feat: F-16 guidance module stub`

---

### Task 16: PLANE6 vehicle + translate gamma case

**Files:** Create `vehicle.py`; register `PLANE6`; translate `input_gamma.asc` (or `input.asc` first vehicle only if simpler) + decks to `Python/cases/falcon6/`; `test_plane6_one_step.py`

**Interfaces:** Run 0.1 s, int_step=0.001; `hbe` near 1000.

- [ ] **Step 1:** smoke test FAIL then PASS
- [ ] **Step 2–5:** wire, commit `feat: run FALCON6 PLANE6 from JSONC`

---

### Task 17: FALCON6 e2e golden (optional file)

**Files:** `Python/tests/e2e/test_falcon6_gamma.py`

**Interfaces:** Skip without `tests/e2e/goldens/falcon6/plot.csv`; else CSV tolerances on `hbe`/`vmach`.

- [ ] **Step 1–5:** skip/compare, commit `test: FALCON6 e2e gate (skip without golden)`

---

## Self-review

Flat6 modules and F-16 vehicle modules mapped to tasks. 3D look_up and mat3tr included. HYPER3 regression called out in constraints.
