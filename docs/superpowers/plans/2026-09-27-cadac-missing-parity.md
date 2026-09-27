# C++→Python Missing-Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port every C++ capability the inventory marks `missing`/`stubbed` (plus per-vehicle gaps the inventory hides) into the Python `cadac` package with TDD proofs.

**Architecture:** Audit first (Task 0 locks the true per-vehicle gap matrix, because `inventory.json` counts a mode `ported` if *any* vehicle implements it). Then one implementer per vehicle-area in parallel: HYPER6 vehicles, HYPER6 guidance, HYPER6 control, FALCON6, Round3/Flat3 guidance, environment/stub sweep. Ground0 is radar-owned data, not a vehicle. Finish with inventory regen + full review.

**Tech Stack:** Python >= 3.11, numpy, pytest. Work in `Python/` with `PYTHONPATH=src:tools`.

**Spec:** `docs/superpowers/specs/2026-09-06-cadac-hyper6-design.md` (HYPER6 non-goals list is the vehicle/mode source), `Python/tools/cadac_cpp/inventory.json` + `Python/tools/cadac_cpp/inventory.py` (24 `missing` + 18 `stubbed` rows), C++ sources under `CADAC_Simulations/{HYPER6_250125,ROCKET6_250122,CRUISE5_250115,HYPER5_250113,FALCON5_250116,FALCON6_250201,SAM6_250217}/`.

**Status (2026-09-27):** Implemented. Left uncommitted until the commit that records this note. `UPDATES.md` version 0.175.0.

- Task 0 gap audit, then parallel Tasks 1, 2, 4, 5, 6, 7; Task 3 after Task 2 review; probe cleanup; Task 8 inventory regen.
- Unit tests: 2267 passed, 2 skipped. HYPER6 climb e2e: 10 passed.
- FALCON5 `mguidance` 6 and 60 still raise; they are not C++ modes.
- Inventory "ported" remains global. Quality `store_get`/`store_set` counters were exempted from the frozen baseline; `baseline.json` was not rewritten.

## Global Constraints

- Grok non-fast implementers + reviewers only (e.g. `grok-4.7-high`); no Fast, Claude, GPT, Composer variants unless the user approves that run.
- TDD iron law: failing test first, watch it fail, minimal code, watch it pass. No production code without a failing test.
- Do not rename classes, CADAC type tokens (`SAT3`, `RADAR0`, `MISSILE6`, …) or family strings. New packages live under the kernel tree (`round6/hyper6/`, `flat6/falcon6/`, …). `cadac.eom.rotor` stays.
- Port C++ formulas line-for-line in behavior (row-major ijk multiply via existing helpers where C++ uses `Matrix::operator*`); unit rtol=1e-12 vs CADAC formulas; CSV e2e rtol=1e-5, atol=max(1e-6, 5e-6*|g|).
- Scenario JSON under `Python/cases/` is not edited by port tasks (except new e2e cases that copy an existing case file). Never edit `CADAC_Simulations/*.cpp|*.hpp`.
- Do not touch `Python/build/lib` (generated). Do not push. Commit steps below run only with user approval per agent-permissions.
- Each task runs `python -m pytest` on the files it touched; Task 8 runs the full unit suite.

---

### Task 0: Per-vehicle gap audit + RED baseline

**Files:**
- Create: `Python/tests/unit/test_cpp_gap_audit.py`
- Modify: none (read-only audit of `Python/src/cadac/vehicles/**`, `Python/src/cadac/eom/round6.py`, `Python/tools/cadac_cpp/inventory.json`)

**Interfaces:**
- Consumes: `inventory.json` missing/stubbed lists (24 + 18 rows), C++ branch grep.
- Produces: `GAP_MATRIX: dict[tuple[str, str, int], str]` mapping `(program, flag, value) -> owning python file` for every gap later tasks implement. Tasks 1–7 read this file to learn exact scope; the matrix is the audit's deliverable.

- [ ] **Step 1: Write the failing probe test**

```python
import pytest
from cadac.cli import _resolve_vehicle

def _vehicle(program, vtype, **kw):
    family = {"HYPER6": None, "ROCKET6": "rocket6"}.get(program, program.lower())
    cls = _resolve_vehicle(family, vtype)
    return cls("probe", **kw)

def test_audit_hyper6_guidance_branches_raise_today():
    from cadac.tables.lookup import Datadeck
    aero, prop = Datadeck.from_tables({}), Datadeck.from_tables({})
    for mguide in (3, 30, 33, 4, 5, 6, 7, 8):
        veh = _vehicle("HYPER6", "HYPER6", aero_deck=aero, prop_deck=prop, events=[])
        veh.define()
        veh.store.set("mguide", mguide)
        with pytest.raises(ValueError, match="unknown mguide"):
            _execute_module(veh, "guidance")

def test_audit_hyper6_vehicles_unregistered_today():
    with pytest.raises(ValueError, match="SAT3"):
        _resolve_vehicle(None, "SAT3")
    with pytest.raises(ValueError, match="RADAR0"):
        _resolve_vehicle(None, "RADAR0")
```

`_execute_module` is a 10-line helper in the test file that builds a `SimContext`-less stub (`types.SimpleNamespace(int_step=0.01)`) and calls the named module's `execute`. Also probe: FALCON6 `mguid` 30/33, CRUISE5/HYPER5/FALCON5 `mguidance` 3 (+6/60), HYPER6/ROCKET6 `matmo=1`, HYPER6 `maero=2`/`minit=1`, SAM6 `mins` 2/3 + `mterm=-1`, CRUISE5/HYPER5/FALCON5 `mcontrol` 1/10/11. Each probe asserts *today's* behavior (raise or resolve-fail), so the file passes now and flips to GREEN-removal as later tasks land.

- [ ] **Step 2: Run test to verify it passes as a baseline**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_cpp_gap_audit.py -v`
Expected: PASS (it locks current gaps; later tasks delete probes as they port each branch).

- [ ] **Step 3: Write the GAP_MATRIX audit**

Read each vehicle `execute` for `==`/`!=`/`in (...)` on every `MODE_FLAGS` entry and record per-vehicle support (the inventory extractor misses `!=` like ROCKET6 `mguide != 5` and counts a mode ported if *any* vehicle has it, e.g. `mguidance==33` in FALCON5 hides CRUISE5's gap). Publish the matrix as a module-level dict in the test file plus a human table in the task's review notes: program × flag × value → `ported here | raises here (file:line)`.

- [ ] **Step 4: Re-run audit test plus inventory check**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_cpp_gap_audit.py tests/unit/test_cadac_cpp_inventory.py -q`
Expected: PASS. Attach the matrix to the review package.

- [ ] **Step 5: Commit (only with user approval)**

```bash
git add Python/tests/unit/test_cpp_gap_audit.py
git commit -m "test: lock per-vehicle C++ gap baseline"
```

---

### Task 1: HYPER6 SAT3 + RADAR0 vehicles

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/satellite.py` (`Hyper6Satellite`, `type="SAT3"`), `Python/src/cadac/vehicles/round6/hyper6/radar.py` (`Hyper6Radar`, `type="RADAR0"`, Ground0 tracks owned by radar, not a vehicle)
- Modify: `Python/src/cadac/cli.py` (global `_VEHICLE_TYPES["SAT3"]`, `["RADAR0"]`; SAM6 keeps family keys so no collision), `Python/src/cadac/io/translate.py` (accept SAT3/RADAR0 blocks in HYPER6 scenarios if it currently rejects them)
- Test: `Python/tests/unit/test_hyper6_sat_radar.py`

**Interfaces:**
- Consumes: GAP_MATRIX entries `(HYPER6, vehicle, SAT3|RADAR0)`; C++ `satellite_modules.cpp` (53 lines: module list), `satellite_functions.cpp` (orbit math), `radar_modules.cpp`, `radar_functions.cpp`, `ground0_modules.cpp` (radar-owned).
- Produces: `_resolve_vehicle(None, "SAT3") -> Hyper6Satellite`, `_resolve_vehicle(None, "RADAR0") -> Hyper6Radar`, both with working `define/initialize/execute` and `com_names`.

- [ ] **Step 1: Write the failing test**

```python
from cadac.cli import _resolve_vehicle

def test_sat3_radar_registered():
    assert _resolve_vehicle(None, "SAT3").__name__ == "Hyper6Satellite"
    assert _resolve_vehicle(None, "RADAR0").__name__ == "Hyper6Radar"

def test_sat3_one_step_advances_orbit():
    from cadac.tables.lookup import Datadeck
    cls = _resolve_vehicle(None, "SAT3")
    veh = cls("sat", events=[])
    veh.define()
    _run_one_step(veh)
    assert veh.store.get("time") > 0

def test_radar_holds_ground_tracks_not_a_vehicle():
    cls = _resolve_vehicle(None, "RADAR0")
    veh = cls("rdr", events=[])
    veh.define()
    assert "NGROUND0" in veh.store or "ground" in " ".join(veh.store.names()).lower()
    with pytest.raises(ValueError, match="Ground0"):
        _resolve_vehicle(None, "GROUND0")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_hyper6_sat_radar.py -v`
Expected: FAIL with `unknown vehicle type 'SAT3'`.

- [ ] **Step 3: Write minimal implementation**

Port `satellite_modules.cpp` module list + `satellite_functions.cpp` math into `satellite.py`; port `radar_modules.cpp` + `radar_functions.cpp` + `ground0_modules.cpp` track handling into `radar.py` (tracks as radar module state, matching C++ `NGROUND0` array). Register both globals in `cli.py`. Keep `type` strings exactly `SAT3`/`RADAR0`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_hyper6_sat_radar.py tests/unit/test_cadac_cpp_inventory.py -q`
Expected: PASS. Then delete the two `test_audit_*sat_radar*` probes from Task 0's file in this task's final edit.

- [ ] **Step 5: Commit (only with user approval)**

```bash
git add Python/src/cadac/vehicles/round6/hyper6/satellite.py Python/src/cadac/vehicles/round6/hyper6/radar.py Python/src/cadac/cli.py Python/src/cadac/io/translate.py Python/tests/unit/test_hyper6_sat_radar.py Python/tests/unit/test_cpp_gap_audit.py
git commit -m "feat: add HYPER6 SAT3 and RADAR0 vehicles"
```

---

### Task 2: HYPER6 line/arc guidance (mguide 30/3/33/4)

**Files:**
- Modify: `Python/src/cadac/vehicles/round6/hyper6/guidance.py`
- Test: `Python/tests/unit/test_hyper6_guidance_lines.py`

**Interfaces:**
- Consumes: C++ `guidance.cpp:201-221` (`guidance_line`, `guidance_arc` callees); existing store fields (`wp_lonx/wp_latx/wp_alt`, `psifdx/thtfdx`, `grav`, `phicomx/alcomx/ancomx`).
- Produces: `mguide in (30, 3, 33, 4)` executes without `ValueError`; `mguide==5..8` still raise (Task 3 owns them).

- [ ] **Step 1: Write the failing test**

```python
def test_hyper6_line_guidance_branches():
    veh = _hyper6_vehicle(mguide=30, wp=(0.1, 0.05, 15000.0))
    _execute_module(veh, "guidance")
    assert veh.store.get("alcomx") != 0.0
    veh30 = veh
    veh3 = _hyper6_vehicle(mguide=3, wp=(0.1, 0.05, 15000.0))
    _execute_module(veh3, "guidance")
    assert veh3.store.get("ancomx") != 0.0
    veh33 = _hyper6_vehicle(mguide=33, wp=(0.1, 0.05, 15000.0))
    _execute_module(veh33, "guidance")
    assert veh33.store.get("alcomx") != 0.0 and veh33.store.get("ancomx") != 0.0

def test_hyper6_arc_guidance_sets_bank():
    veh = _hyper6_vehicle(mguide=4, wp=(0.1, 0.05, 15000.0))
    _execute_module(veh, "guidance")
    assert veh.store.get("phicomx") != 0.0
```

(`_hyper6_vehicle` builds `Hyper6` with empty decks, calls `define()`, sets `mguide` + waypoint fields.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_hyper6_guidance_lines.py -v`
Expected: FAIL with `unknown mguide 30`.

- [ ] **Step 3: Write minimal implementation**

Port the four C++ `if(mguide==…)` blocks verbatim in behavior (`guidance_line` shared by 30/3/33 with the same per-branch output assignments: 30→lateral only, 3→pitch only with `alcomx=0`, 33→both; 4→`phicomx=guidance_arc(…)`). Reuse existing `guidance_line`/`guidance_arc` helpers if present, else port them from C++ in this task.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_hyper6_guidance_lines.py -q`
Expected: PASS. Remove the matching Task 0 probes.

- [ ] **Step 5: Commit (only with user approval)**

```bash
git add Python/src/cadac/vehicles/round6/hyper6/guidance.py Python/tests/unit/test_hyper6_guidance_lines.py Python/tests/unit/test_cpp_gap_audit.py
git commit -m "feat: add HYPER6 line and arc guidance"
```

---

### Task 3: HYPER6 terminal guidance (mguide 5/6/7/8)

**Files:**
- Modify: `Python/src/cadac/vehicles/round6/hyper6/guidance.py` (extend Task 2), `Python/src/cadac/vehicles/round6/hyper6/seeker.py` or datalink wiring only if C++ `guidance_pronav/AGL/glideslope` reads seeker/datalink state (else no new files)
- Test: `Python/tests/unit/test_hyper6_guidance_terminal.py`

**Interfaces:**
- Consumes: C++ `guidance.cpp:222-260` (`guidance_ltg`, `guidance_pronav`, `guidance_AGL`, `guidance_glideslope`, `gs_flag`/`ltg_count`/`time_ltg`/`UTBC` plumbing); ROCKET6's `guidance_ltg` as a pattern reference only (hyper buffers differ: `UTBBC/STBIK/VTBIK/TBIC`).
- Produces: `mguide in (5, 6, 7, 8)` execute; `mguide==5` advances `time_ltg`/`ltg_count` and writes `UTBC`; `6/7` write `aycomx/azcomx`; `8` writes `UTBC` from glideslope.

- [ ] **Step 1: Write the failing test**

```python
def test_hyper6_ltg_advances_clock():
    veh = _hyper6_vehicle(mguide=5)
    _execute_module(veh, "guidance")
    _execute_module(veh, "guidance")
    assert veh.store.get("ltg_count") == 2

def test_hyper6_pronav_and_agl_issue_accel():
    for mguide in (6, 7):
        veh = _hyper6_vehicle(mguide=mguide)
        _execute_module(veh, "guidance")
        assert veh.store.get("aycomx") != 0.0 or veh.store.get("azcomx") != 0.0

def test_hyper6_glideslope_writes_utbc():
    veh = _hyper6_vehicle(mguide=8)
    _execute_module(veh, "guidance")
    assert np.linalg.norm(np.asarray(veh.store.get("UTBC"))) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_hyper6_guidance_terminal.py -v`
Expected: FAIL with `unknown mguide 5`.

- [ ] **Step 3: Write minimal implementation**

Port the four blocks including clock/save plumbing (`init_flag`, `time_ltg`, `ltg_count`, `gs_flag`, `mprop` feedback, `UTBC/aycomx/azcomx` outputs). Port callees `guidance_ltg/pronav/AGL/glideslope` from HYPER6 sources (do not copy ROCKET6's; buffer names differ).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_hyper6_guidance_terminal.py tests/unit/test_hyper6_guidance_lines.py -q`
Expected: PASS. Remove matching Task 0 probes.

- [ ] **Step 5: Commit (only with user approval)**

```bash
git add Python/src/cadac/vehicles/round6/hyper6/guidance.py Python/tests/unit/test_hyper6_guidance_terminal.py Python/tests/unit/test_cpp_gap_audit.py
git commit -m "feat: add HYPER6 terminal guidance"
```

---

### Task 4: HYPER6 accel/heading control (mauty 3/4, mautp 3/4/5)

**Files:**
- Modify: `Python/src/cadac/vehicles/round6/hyper6/control.py`
- Test: `Python/tests/unit/test_hyper6_control_accel.py`

**Interfaces:**
- Consumes: C++ `control.cpp:176-206` (`control_lateral_accel`, `control_normal_accel`, `control_heading`, `control_altitude`, `gmax/gminx` limiting, `mauty=maut/10`, `mautp=maut%10` decode); existing `control_yaw_rate/pitch_rate/gamma/roll` stay.
- Produces: `maut` values exercising lateral-accel (mauty 3), normal-accel (mautp 3), gamma (mautp 4, already), heading (mauty 4), altitude (mautp 5) produce bounded `delacx/delecx/delrcx`.

- [ ] **Step 1: Write the failing test**

```python
def test_hyper6_lateral_accel_control():
    veh = _hyper6_vehicle(maut=30, alcomx=0.5)
    _execute_module(veh, "control")
    assert veh.store.get("delrcx") != 0.0

def test_hyper6_heading_control():
    veh = _hyper6_vehicle(maut=40, psivdcomx=0.2)
    _execute_module(veh, "control")
    assert veh.store.get("phicomx") != 0.0
```

(plus `mautp` 3/5 cases asserting `delecx` saturates at `delimx` for huge `ancomx`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_hyper6_control_accel.py -v`
Expected: FAIL with `unknown maut 30`.

- [ ] **Step 3: Write minimal implementation**

Port the C++ `mauty==3 / mautp==3 / mautp==4 / mauty==4 / mautp==5` blocks with `gmax/gminx` limiting and existing command saturation (`dalimx/delimx/drlimx`). Port the four callee controllers.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_hyper6_control_accel.py -q`
Expected: PASS. Remove matching Task 0 probes.

- [ ] **Step 5: Commit (only with user approval)**

```bash
git add Python/src/cadac/vehicles/round6/hyper6/control.py Python/tests/unit/test_hyper6_control_accel.py Python/tests/unit/test_cpp_gap_audit.py
git commit -m "feat: add HYPER6 accel and heading control"
```

---

### Task 5: FALCON6 line guidance + accel control (mguid 30/33, mauty 3/4)

**Files:**
- Modify: `Python/src/cadac/vehicles/flat6/falcon6/guidance.py`, `Python/src/cadac/vehicles/flat6/falcon6/control.py`
- Test: `Python/tests/unit/test_falcon6_guidance_control.py`

**Interfaces:**
- Consumes: C++ `FALCON6/guidance.cpp` (line-guidance 30/33), `FALCON6/control.cpp` (`mauty==3/4` accel/heading branches); GAP_MATRIX FALCON6 rows.
- Produces: `mguid` 30/33 guide; `mauty` 3/4 control; today both raise `ValueError`.

- [ ] **Step 1: Write the failing test**

```python
def test_falcon6_line_guidance():
    veh = _falcon6_vehicle(mguid=30)
    _execute_module(veh, "guidance")
    assert veh.store.get("dwbh") != 0.0 or veh.store.get("nl_gain") != 0.0

def test_falcon6_accel_control():
    veh = _falcon6_vehicle(maut=30)
    _execute_module(veh, "control")
    assert True  # executes without ValueError; command assertions per C++ branch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_falcon6_guidance_control.py -v`
Expected: FAIL with `unknown mguid 30`.

- [ ] **Step 3: Write minimal implementation**

Port FALCON6 (not HYPER6) guidance/control branches; keep Flat6 state names. Do not touch HYPER6 files (Task 4 owns those) — this is why the tasks run in parallel safely.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_falcon6_guidance_control.py -q`
Expected: PASS. Remove matching Task 0 probes.

- [ ] **Step 5: Commit (only with user approval)**

```bash
git add Python/src/cadac/vehicles/flat6/falcon6/guidance.py Python/src/cadac/vehicles/flat6/falcon6/control.py Python/tests/unit/test_falcon6_guidance_control.py Python/tests/unit/test_cpp_gap_audit.py
git commit -m "feat: add FALCON6 line guidance and accel control"
```

---

### Task 6: Round3/Flat3 guidance drop-outs + control stubs

**Files:**
- Modify: `Python/src/cadac/vehicles/round3/cruise5/guidance.py`, `Python/src/cadac/vehicles/round3/hyper5/guidance.py`, `Python/src/cadac/vehicles/flat3/falcon5/guidance.py`, plus `control.py` in the same three packages only for `mcontrol` 1/10/11 if the Task 0 audit confirms they raise.
- Test: `Python/tests/unit/test_round3_flat3_guidance_gap.py`

**Interfaces:**
- Consumes: GAP_MATRIX rows for `(CRUISE5|HYPER5, mguidance, 3|6|60)`, `(FALCON5, mguidance, 3)`; C++ `cruise_modules.cpp:981-1009`, `hyper_modules.cpp:1006-1034`, `plane_modules.cpp:937-959`; existing `guidance_line/point/pronav/arc` helpers.
- Produces: audit-confirmed missing branches execute (floor: pitch-line 3 via `guidance_line`; 6/60 per C++ bodies); `mcontrol` 1/10/11 either execute or stay raising only if C++ shows them unreachable (audit decides, reviewer verifies).

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("program", ["CRUISE5", "HYPER5"])
def test_pitch_line_guidance(program):
    veh = _round3_vehicle(program, mguidance=3)
    _execute_module(veh, "guidance")
    assert veh.store.get("ancomx") != 0.0

def test_falcon5_pitch_line_guidance():
    veh = _falcon5_vehicle(mguidance=3)
    _execute_module(veh, "guidance")
    assert veh.store.get("ancomx") != 0.0
```

(plus 6/60 cases asserting the C++-specified outputs; if C++ `6` needs seeker state, plant it in the test like existing seeker tests do.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_round3_flat3_guidance_gap.py -v`
Expected: FAIL with `unknown mguidance 3`.

- [ ] **Step 3: Write minimal implementation**

Port each program's own C++ branch bodies (do not cross-copy CRUISE5↔HYPER5: waypoint frames differ — spherical vs same-kernel-but-own-module). Keep `anposlimx/anneglimx/allimx` limiting.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_round3_flat3_guidance_gap.py -q`
Expected: PASS. Remove matching Task 0 probes.

- [ ] **Step 5: Commit (only with user approval)**

```bash
git add Python/src/cadac/vehicles/round3/cruise5/guidance.py Python/src/cadac/vehicles/round3/hyper5/guidance.py Python/src/cadac/vehicles/flat3/falcon5/guidance.py Python/tests/unit/test_round3_flat3_guidance_gap.py Python/tests/unit/test_cpp_gap_audit.py
git commit -m "feat: add Round3 Flat3 pitch-line guidance"
```

---

### Task 7: Environment, aero, INS and intercept stubs

**Files:**
- Modify: `Python/src/cadac/eom/round6.py` (`matmo==1` path shared by HYPER6+ROCKET6), `Python/src/cadac/vehicles/round6/hyper6/aero.py` (`maero==2` transfer vehicle), `Python/src/cadac/vehicles/round6/hyper6/vehicle.py` or newton (`minit==1` satellite-relative ICs), `Python/src/cadac/vehicles/flat6/sam6/ins.py` (`mins` 2/3), `Python/src/cadac/vehicles/flat6/sam6/intercept.py` (`mterm==-1`)
- Test: `Python/tests/unit/test_env_stub_gap.py`

**Interfaces:**
- Consumes: C++ `HYPER6/environment.cpp:183+` + `ROCKET6/environment.cpp:193+` (`matmo` decode), `HYPER6/aerodynamics.cpp` (`maero==2`), `HYPER6/newton.cpp` (`minit==1`), `SAM6/ins.cpp` + `intercept.cpp`; ROCKET6 `mguide==5` extractor artifact (test-only: prove LTG already works, do not reimplement).
- Produces: `matmo==1` atmospheres, `maero==2` tables path, `minit==1` ICs, SAM6 `mins` 2/3 + `mterm` -1 execute; ROCKET6 LTG covered by a regression test.

- [ ] **Step 1: Write the failing test**

```python
def test_round6_matmo1_atmosphere():
    veh = _hyper6_vehicle(mair=100)
    _execute_module(veh, "environment")
    assert veh.store.get("rho") > 0

def test_rocket6_ltg_regression():
    veh = _rocket6_vehicle(mguide=5)
    _execute_module(veh, "guidance")  # passes today; locks the extractor artifact
    assert veh.store.get("ltg_count") >= 1

def test_sam6_mterm_minus1():
    veh = _sam6_missile(mterm=-1)
    _execute_module(veh, "intercept")
    assert True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && PYTHONPATH=src python -m pytest tests/unit/test_env_stub_gap.py -v`
Expected: FAIL on `matmo`/`maero`/`minit`/`mins`/`mterm` (the LTG case passes — it documents the inventory false-positive).

- [ ] **Step 3: Write minimal implementation**

Port `matmo==1` (NASA-extended atmosphere selection; shared `eom/round6.py` so both vehicles gain it), `maero==2`, `minit==1`, SAM6 `mins` 2/3 + `mterm` -1 from their C++ files. Fix `extract_python.py` `_EQ`/`_IN_TUPLE` to also match `!=` guards (so ROCKET6 `mguide != 5` counts implemented) — 5-line change, covered by `test_cadac_cpp_extract_python.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_env_stub_gap.py tests/unit/test_cadac_cpp_extract_python.py -q`
Expected: PASS. Remove matching Task 0 probes.

- [ ] **Step 5: Commit (only with user approval)**

```bash
git add Python/src/cadac/eom/round6.py Python/src/cadac/vehicles/round6/hyper6/aero.py Python/src/cadac/vehicles/round6/hyper6/vehicle.py Python/src/cadac/vehicles/flat6/sam6/ins.py Python/src/cadac/vehicles/flat6/sam6/intercept.py Python/tools/cadac_cpp/extract_python.py Python/tests/unit/test_env_stub_gap.py Python/tests/unit/test_cpp_gap_audit.py
git commit -m "feat: fill environment aero INS intercept stubs"
```

---

### Task 8: Whole-branch parity review + inventory regen

**Files:**
- Modify: `Python/tools/cadac_cpp/inventory.json` (regen only), `UPDATES.md` (one new top entry), `README.md` (only if the architecture line changed)
- Test: full `tests/unit`, e2e HYPER6 intercept/head-on if goldens exist

**Interfaces:**
- Consumes: all Tasks 0–7 deliverables.
- Produces: zero `missing` vehicle rows; remaining `missing`/`stubbed` mode rows each either ported or justified in review notes; `import cadac.cli` green; unit suite green.

- [ ] **Step 1: Regenerate inventory and write the delta test**

```bash
cd Python && PYTHONPATH=src:tools python -m cadac_cpp.inventory
```

Add `Python/tests/unit/test_parity_delta.py`:

```python
import json

def test_no_missing_vehicles():
    rows = json.load(open("Python/tools/cadac_cpp/inventory.json"))
    missing = [r for r in rows if r["kind"] == "vehicle" and r["status"] == "missing"]
    assert missing == []
```

- [ ] **Step 2: Run to verify**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit -q`
Expected: PASS (2205+ new tests). E2E `test_hyper6_climb.py` still passes; intercept/head-on run if goldens present, else skip like other goldens.

- [ ] **Step 3: Dispatch whole-branch reviewer** (spec compliance + cross-task conflicts: duplicate helper names, registry collisions on `SAT3`/`RADAR0`, shared `eom/round6.py` regressions). One fix round max, then adjudicate residuals in review notes.

- [ ] **Step 4: Update docs**

`UPDATES.md` new top entry (bump per project-docs: feature → subver). `README.md` only if the vehicle/kernel table changed.

- [ ] **Step 5: Commit (only with user approval)**

```bash
git add Python/tools/cadac_cpp/inventory.json UPDATES.md README.md Python/tests/unit/test_parity_delta.py
git commit -m "feat: close C++ parity gaps, regen inventory"
```

## Self-Review

- Spec coverage: HYPER6 vehicles (Task 1) + all 11 HYPER6 missing modes (Tasks 2–4, 7) + 2 HYPER6 vehicles + ROCKET6 matmo/mguide artifact (Task 7) + FALCON6 slice (Task 5) + CRUISE5/HYPER5/FALCON5 drop-outs (Task 6) + all 18 stubbed rows (Tasks 4–7 verify-or-port; Task 0 probes force the decision per row). No inventory row without an owner.
- No placeholders: every step names files, C++ lines, test code, run commands, commit content.
- Type consistency: `_hyper6_vehicle/_rocket6_vehicle/_falcon6_vehicle/_round3_vehicle/_falcon5_vehicle/_sam6_missile/_execute_module/_run_one_step` are test-local helpers defined in each test file (no cross-task imports); production names (`Hyper6Satellite`, `Hyper6Radar`, `guidance_ltg/pronav/AGL/glideslope/line/arc`, `control_lateral_accel/normal_accel/heading/altitude`) match C++ callee names.

**Plan complete and saved to `docs/superpowers/plans/2026-09-27-cadac-missing-parity.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch Task 0 first (it defines scope for the rest), then Tasks 1–7 in parallel where files allow (1, 4, 5, 6 touch disjoint packages; 2→3 sequential on `guidance.py`; 7 touches shared `eom/round6.py` so it runs alone or last-but-one), reviewer per task, Task 8 final review.

**2. Inline Execution** — execute Tasks 0–8 in order in this session with checkpoints for review.

**Which approach?**
