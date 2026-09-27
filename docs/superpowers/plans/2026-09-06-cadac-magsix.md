# CADAC MAGSIX Rotor (DNU Magnus) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add unique Rotor EOM and JSONC type `ROTOR` (`family="magsix"`); unit-test each C++ module used by MAGSIX `input.asc` (attitude RECT.MR1) and `input_trajectoryMR1.asc`; e2e vs CADAC CSV when goldens exist.

**Architecture:** New `cadac.eom.rotor` (US76 + spherical `gravity(hbe)`, DNU Magnus trajectory + attitude). Vehicle `cadac.vehicles.planar.magsix.vehicle.Rotor` composes environment/trajectory/attitude. No Flat6/Flat3/Round6 classes. No aero/prop decks.

**Tech Stack:** Python >= 3.11, numpy, pytest. Work in `Python/`.

**Spec:** `docs/superpowers/specs/2026-09-06-cadac-magsix-design.md`

**Depends on:** kernel skip-if-missing module + deck-optional factory. Family API is added in Task 1 if absent.

## Global Constraints

- Spec + parent design spec (Rotor is a reserved EOM layer, not a Flat6 flag)
- Do not regress HYPER3 e2e, HYPER5/HYPER6 units, FALCON5/FALCON6 units
- Do not reuse `Flat6*` / `Flat3*` / `Round6*` / `Round3*` classes
- Atmosphere US76 (`atmosphere76(hbe)`); gravity `gravity(hbe)`. Not ISO 62, not WGS84
- JSONC type `ROTOR`. Register `_VEHICLE_TYPES["ROTOR"]` and `_VEHICLE_FAMILIES[("magsix","ROTOR")]`
- `family` set → only family table; `family is None` → type table
- Constructor `(name, events=None)`; no decks; `"ROTOR"` in `_NO_DECK_TYPES`
- First case `input.asc` RECT.MR1 attitude: `hbe=1000`, `dvbe=16.6`, `omega_rpm=850`, `thtvlx=-77`, `nonlinear=0`, `ENDTIME 0.35`, `int_step 0.0001`
- Executive `sim_time` / `int_step` / `end_time` are **DNT**; store `time = tau * sim_time`
- `RPM=9.5493`, `RHO_SL=1.225` module-level in `eom/rotor.py`; do not edit `cadac.constants`
- `nonlinear in {0,1}`; other → `ValueError`. `mwind==0` only; other → `ValueError`
- Ground impact: `health=0` + combus status 0; **no** `sys.exit`
- E2E skip if goldens absent. Align on `sim_time` (plot-flagged). Compare **all** shared plot-flagged columns. Lock `plot_step` 0.005 (attitude) / 0.01 (trajectory)
- Unit rtol=1e-12; CSV e2e rtol=1e-5, atol=max(1e-6, 5e-6*|g|)
- Grok `cursor-grok-4.6-high` implementer + reviewer; TDD; no Fast/Kimi
- Unknown-type sentinel is `"NO_SUCH_TYPE"`: always retarget tests that use `"AIM5"` or `"ROTOR"` as the unknown token. Do not leave `"AIM5"`
- Family API is the AIM5 loader (vehicle key else scenario `"family"` else `None`; vehicle wins). Do not overwrite `test_vehicle_family.py`. `ValueError` includes type and family when set
- Do not port `eng_ang_mom=rotor[213]`

## File map

- `Python/src/cadac/io/scenario.py` — `VehicleSpec.family`
- `Python/src/cadac/io/translate.py` — `translate_scenario_asc(..., family=None)`
- `Python/src/cadac/cli.py` — `_VEHICLE_FAMILIES`, ROTOR factory, `_NO_DECK_TYPES`
- `Python/src/cadac/eom/rotor.py` — Environment, Trajectory, Attitude
- `Python/src/cadac/vehicles/planar/magsix/{__init__,vehicle}.py`
- Tests `test_rotor_family.py`, `test_rotor_*.py`, `test_magsix_*.py` (do not overwrite `test_vehicle_family.py`)
- Cases `Python/cases/magsix/`
- C++: `environment.cpp`, `trajectory.cpp`, `attitude.cpp`, `global_constants.hpp`, `class_hierarchy.hpp`

---

### Task 1: Family API (idempotent AIM5 loader)

**Files:**
- Modify only if missing: `Python/src/cadac/io/scenario.py`, `Python/src/cadac/io/translate.py`, `Python/src/cadac/cli.py`
- Test: `Python/tests/unit/test_rotor_family.py`
- Do **not** create or overwrite `Python/tests/unit/test_vehicle_family.py` (AIM5 owns that file)

**Interfaces:**
- Consumes: existing `VehicleSpec`, `load_scenario`, `translate_scenario_asc`, `_VEHICLE_TYPES`, `_build_vehicle`
- Produces (add only if absent; keep AIM5 semantics if present):
  - `VehicleSpec.family: str | None = None`
  - `load_scenario`: vehicle `"family"` else scenario-level `"family"` else `None`; vehicle key wins
  - `translate_scenario_asc(src, dst_dir, family=None)`: stamp `"family"` on each vehicle when given; omit the key when `None`
  - `_VEHICLE_FAMILIES: dict[tuple[str, str], type]` (empty until Task 7)
  - `_build_vehicle`: `family` set → **only** `_VEHICLE_FAMILIES[(family, type)]`; `family is None` → `_VEHICLE_TYPES[type]`. No type-table fallthrough
  - Unknown → `ValueError` that includes the type token **and**, when set, the family token (e.g. `unknown vehicle type 'CRUISE3' family 'magsix'`)
- Do not register `ROTOR` yet

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc
from cadac.cli import _VEHICLE_FAMILIES, _VEHICLE_TYPES


def test_scenario_family_applies_when_vehicle_omits_it(tmp_path: Path):
    path = tmp_path / "scen.jsonc"
    path.write_text(
        '{ "title": "t", "family": "magsix", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.01}, "end_time": 0, '
        '"vehicles": [ { "type": "CRUISE3", "name": "v", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    assert load_scenario(path).vehicles[0].family == "magsix"


def test_vehicle_family_overrides_scenario_family(tmp_path: Path):
    path = tmp_path / "ov.jsonc"
    path.write_text(
        '{ "title": "t", "family": "aim5", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.01}, "end_time": 0, '
        '"vehicles": [ { "family": "magsix", "type": "CRUISE3", "name": "v", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    assert load_scenario(path).vehicles[0].family == "magsix"


def test_translate_family_magsix_stamps_vehicles(tmp_path: Path):
    src = tmp_path / "in.asc"
    src.write_text(
        "TITLE t\nOPTIONS n_scrn\nMODULES\n\tenvironment\tdef,exec\nEND\n"
        "TIMING\n\tint_step 0.01\nEND\nVEHICLES 1\n\tROTOR RECT.MR1\n"
        "\thbe 1000\n\tEND\nEND\nENDTIME 0.35\nSTOP\n",
        encoding="utf-8",
        newline="\n",
    )
    translate_scenario_asc(src, tmp_path, family="magsix")
    data = json.loads((tmp_path / "in.jsonc").read_text(encoding="utf-8"))
    assert data["vehicles"][0]["family"] == "magsix"
    assert data["vehicles"][0]["type"] == "ROTOR"


def test_family_set_error_includes_type_and_family(tmp_path: Path):
    path = tmp_path / "nope.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.01}, "end_time": 0, '
        '"vehicles": [ { "type": "CRUISE3", "name": "v", '
        '"family": "magsix", "params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    assert "CRUISE3" in _VEHICLE_TYPES
    assert ("magsix", "CRUISE3") not in _VEHICLE_FAMILIES
    with pytest.raises(ValueError, match=r"CRUISE3.*magsix|magsix.*CRUISE3"):
        run_scenario(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_rotor_family.py -v`

Expected: FAIL if family API or family-in-ValueError is missing. If AIM5 already shipped the loader, only the family-in-message assert may fail.

- [ ] **Step 3: Write minimal implementation**

If `VehicleSpec.family` / scenario default / vehicle-key-wins already exist, keep them. Else:

- `VehicleSpec` add `family: str | None = None`
- `_vehicle(parent, raw, scenario_family=None)` sets `family=raw.get("family", scenario_family)`
- `load_scenario` passes `data.get("family")` into `_vehicle`
- `translate_scenario_asc(src, dst_dir, family=None)` stamps vehicles when `family` is a string
- `_VEHICLE_FAMILIES = {}` if missing
- Resolve class as in Interfaces

Always ensure unknown lookup raises with **type and family when set**:

```python
if cls is None:
    if spec.family is not None:
        raise ValueError(
            f"{path}: unknown vehicle type {spec.type!r} family {spec.family!r}"
        )
    raise ValueError(f"{path}: unknown vehicle type {spec.type!r}")
```

Do not register `ROTOR`. Do not rewrite `test_vehicle_family.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_rotor_family.py tests/unit/test_scenario.py tests/translate/test_asc_scenario.py tests/unit/test_cruise3_one_step.py tests/unit/test_plane5_one_step.py tests/unit/test_hyper5_one_step.py tests/unit/test_hyper6_one_step.py -q`

If `tests/unit/test_vehicle_family.py` exists, include it. Then `pytest -q`.

Expected: PASS. AIM5 family tests still pass. Existing JSONC without `family` still uses `_VEHICLE_TYPES`.

- [ ] **Step 5: Commit**

```bash
git add Python/src/cadac/io/scenario.py Python/src/cadac/io/translate.py Python/src/cadac/cli.py Python/tests/unit/test_rotor_family.py
git commit -m "feat: MAGSIX family lookup keeps AIM5 scenario default"
```

---

### Task 2: Rotor environment (mwind=0)

**Files:**
- Create: `Python/src/cadac/eom/rotor.py`
- Test: `Python/tests/unit/test_rotor_environment.py`

**Interfaces:**
- Consumes: `atmosphere76`, `gravity`, `cadac.constants.R`
- Produces: `RotorEnvironment.name=="environment"`. Module-level `RPM=9.5493`, `RHO_SL=1.225`. `define` C++ `def_environment` names (`mwind`,`press`,`rho`,`vsound`,`grav`,`vmach`,`pdynmc`,`tempk`,`dvae`,`dvael`,`waltl`,`dvaeh`,`walth`,`vaed3`,`psiwdx`,`twind`,`VAELS`,`VAELSD`,`VAEL`,`dvba`,`VBAL`). Does not define `hbe`/`VBEL`. `execute`: `mwind==0` only — US76 at `hbe`, `grav=gravity(hbe)`, `VAEL=0`, `VBAL=VBEL-VAEL`, `dvba=||VBAL||`, `vsound=sqrt(1.4*R*tempk)`, `vmach=abs(dvba/vsound)`, `pdynmc=0.5*rho*dvba*dvba`. Other `mwind` → `ValueError`. `initialize`/`terminate` no-op. Do not subclass `Flat6Environment`.

- [ ] **Step 1: Write the failing test**

```python
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import R
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.eom.rotor import RotorEnvironment
from cadac.kernel.state import Field, StateStore


def _vehicle(hbe=1000.0, vbel=(16.6, 0.0, 0.0), mwind=0):
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    env = RotorEnvironment()
    env.define(vehicle)
    s.define(Field("hbe", hbe, "real", "out", "trajectory"))
    s.define(Field("VBEL", vbel, "vec", "diag", "trajectory"))
    s.set("mwind", mwind)
    return vehicle, env


def test_name_is_environment():
    assert RotorEnvironment().name == "environment"


def test_environment_does_not_define_trajectory_fields():
    s = StateStore()
    RotorEnvironment().define(SimpleNamespace(store=s))
    for name in ("hbe", "VBEL", "dvbe", "alt", "SBII"):
        assert name not in s.names()
    assert s.get("mwind") == 0


def test_mwind0_us76_at_hbe_1000():
    vehicle, env = _vehicle()
    env.execute(vehicle, None)
    s = vehicle.store
    rho, press, tempk = atmosphere76(1000.0)
    vsound = (1.4 * R * tempk) ** 0.5
    dvba = 16.6
    np.testing.assert_allclose(s.get("rho"), rho, rtol=1e-12)
    np.testing.assert_allclose(s.get("press"), press, rtol=1e-12)
    np.testing.assert_allclose(s.get("tempk"), tempk, rtol=1e-12)
    np.testing.assert_allclose(s.get("vsound"), vsound, rtol=1e-12)
    np.testing.assert_allclose(s.get("dvba"), dvba, rtol=1e-12)
    np.testing.assert_allclose(s.get("vmach"), abs(dvba / vsound), rtol=1e-12)
    np.testing.assert_allclose(s.get("pdynmc"), 0.5 * rho * dvba**2, rtol=1e-12)
    np.testing.assert_allclose(s.get("grav"), gravity(1000.0), rtol=1e-12)
    np.testing.assert_allclose(s.get("VAEL"), np.zeros(3), rtol=1e-12)
    np.testing.assert_allclose(s.get("VBAL"), np.array([16.6, 0.0, 0.0]), rtol=1e-12)


def test_mwind_1_raises():
    vehicle, env = _vehicle(mwind=1)
    with pytest.raises(ValueError, match="mwind"):
        env.execute(vehicle, None)
```

- [ ] **Step 2:** FAIL `pytest tests/unit/test_rotor_environment.py -v`
- [ ] **Step 3:** Port C++ `environment.cpp` `mwind==0` path into `RotorEnvironment`. Put `RPM` and `RHO_SL` at module top.
- [ ] **Step 4:** PASS `pytest tests/unit/test_rotor_environment.py -v`
- [ ] **Step 5: Commit** `feat: MAGSIX Rotor environment US76`

---

### Task 3: Rotor trajectory init (DNU ICs)

**Files:** Modify `Python/src/cadac/eom/rotor.py`; Test: `Python/tests/unit/test_rotor_trajectory_init.py`

**Interfaces:**
- Consumes: `RotorEnvironment` constants `RPM`, `RHO_SL`; `atmosphere76`; `mat2tr`; `AGRAV`, `RAD`, `DEG`
- Produces: `RotorTrajectory.name=="trajectory"`. `define` C++ `def_trajectory` names including `time`,`sim_time` (**both plot-flagged**),`cd`,`cmdw`,`clw`,`cma`,`mass`,`ref_area`,`ref_length`,`velocity_ss`,`gamma_ss` (**scalar rad**),`omega_ss`,`dvbe`,`psivlx`,`thtvlx`,`hbg`,`hbe`,`omega`,`sbel1`,`sbel2`,`sbel3`,`velocityx`,`velocityxd`,`gamma`,`gammaxd`,`omegax`,`omegaxd`,`moi_spin`,`moi_spinx`,`SBEL`,`SBELD`,`VBEL`,`omega_rpm`,`tau`,`mu`,`tpsp_ratio`. `initialize` ports `init_trajectory` exactly (ignore `sbel3`; `SBEL=(sbel1,sbel2,-hbe)`; `velocity_ss` uses `RHO_SL`; launch `tau` uses `atmosphere76(hbe)`). No stdout.

RECT.MR1 numbers for asserts (recompute in the test from the C++ formulas, rtol 1e-12): `hbe=1000`, `dvbe=16.6`, `psivlx=0`, `thtvlx=-77`, `omega_rpm=850`, `mass=1.5`, `ref_area=0.0468`, `ref_length=0.0625`, `cd=1.31`, `cmdw=-0.45`, `clw=2.51`, `cma=0.508`, `moi_spin=0.004`.

- [ ] **Step 1: Write the failing test**

```python
import math
from types import SimpleNamespace

import numpy as np

from cadac.constants import AGRAV, DEG, RAD
from cadac.env.us76 import atmosphere76
from cadac.eom.rotor import RPM, RHO_SL, RotorTrajectory
from cadac.kernel.state import StateStore
from cadac.math.frames import mat2tr


def test_name_is_trajectory():
    assert RotorTrajectory().name == "trajectory"


def test_rect_mr1_init_sbel_velocityx_gamma_omegax():
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    traj = RotorTrajectory()
    traj.define(vehicle)
    params = {
        "cd": 1.31,
        "cmdw": -0.45,
        "clw": 2.51,
        "cma": 0.508,
        "mass": 1.5,
        "ref_area": 0.0468,
        "ref_length": 0.0625,
        "dvbe": 16.6,
        "psivlx": 0.0,
        "thtvlx": -77.0,
        "hbe": 1000.0,
        "sbel1": 0.0,
        "sbel2": 0.0,
        "sbel3": 99.0,
        "omega_rpm": 850.0,
        "moi_spin": 0.004,
    }
    for name, value in params.items():
        s.set(name, value)
    traj.initialize(vehicle, None)

    gamma_ss = math.atan(1.31 * (-0.45) / (2.51 * 0.508))
    velocity_ss = math.sqrt(
        2 * AGRAV * 1.5 * abs(math.sin(gamma_ss)) / (RHO_SL * 0.0468 * 1.31)
    )
    omega_ss = -velocity_ss * 0.508 / (0.0625 * (-0.45))
    rho, _, _ = atmosphere76(1000.0)
    tau = 2 * 1.5 / (rho * 0.0468 * velocity_ss)
    np.testing.assert_allclose(s.get("gamma_ss"), gamma_ss, rtol=1e-12)
    np.testing.assert_allclose(s.get("velocity_ss"), velocity_ss, rtol=1e-12)
    np.testing.assert_allclose(s.get("omega_ss"), omega_ss, rtol=1e-12)
    np.testing.assert_allclose(s.get("SBEL"), np.array([0.0, 0.0, -1000.0]), rtol=1e-12)
    np.testing.assert_allclose(s.get("velocityx"), 16.6 / velocity_ss, rtol=1e-12)
    np.testing.assert_allclose(s.get("gamma"), -77.0 * RAD, rtol=1e-12)
    np.testing.assert_allclose(s.get("omegax"), (850.0 / RPM) * tau, rtol=1e-12)
    np.testing.assert_allclose(s.get("tau"), tau, rtol=1e-12)
    tvl = mat2tr(0.0, -77.0 * RAD)
    vbel = tvl.T @ np.array([16.6, 0.0, 0.0])
    np.testing.assert_allclose(s.get("VBEL"), vbel, rtol=1e-12)
    assert s.get("time") == 0.0
    assert "plot" in s.field("sim_time").outputs
    assert "plot" in s.field("time").outputs
```

- [ ] **Step 2:** FAIL `pytest tests/unit/test_rotor_trajectory_init.py -v`
- [ ] **Step 3:** Port `init_trajectory` (and `def_trajectory` fields). `gamma_ss` is a scalar Field.
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: MAGSIX trajectory DNU initialization`

---

### Task 4: Rotor trajectory step (DNU EOM + ground impact)

**Files:** Modify `rotor.py`; Test: `Python/tests/unit/test_rotor_trajectory_step.py`

**Interfaces:**
- Consumes: Task 3 `RotorTrajectory.initialize`; `RotorEnvironment.execute`; `integrate`; `SimContext`; `Packet`
- Produces: `execute` ports `Rotor::trajectory` (DNU eqs, then metric, then `SBEL` with `dt=int_step*tau`). Writes `moi_spinx` DNU from input `moi_spin` kg·m². `hbe=-SBEL[2]`. `time=tau*ctx.sim_time`, `sim_time=ctx.sim_time`. If `hbe < hbg`: `vehicle.health=0` and `ctx.combus[ctx.vehicle_slot].status=0`. No `sys.exit`. `terminate` no-op.

- [ ] **Step 1: Write the failing test**

```python
from types import SimpleNamespace

import numpy as np

from cadac.constants import DEG, RAD
from cadac.eom.rotor import RotorEnvironment, RotorTrajectory
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import StateStore
from cadac.math.frames import mat2tr


def _rect_ready(hbg=0.0):
    s = StateStore()
    vehicle = SimpleNamespace(store=s, health=1, name="RECT.MR1")
    env = RotorEnvironment()
    traj = RotorTrajectory()
    env.define(vehicle)
    traj.define(vehicle)
    for name, value in {
        "cd": 1.31,
        "cmdw": -0.45,
        "clw": 2.51,
        "cma": 0.508,
        "mass": 1.5,
        "ref_area": 0.0468,
        "ref_length": 0.0625,
        "dvbe": 16.6,
        "psivlx": 0.0,
        "thtvlx": -77.0,
        "hbe": 1000.0,
        "hbg": hbg,
        "sbel1": 0.0,
        "sbel2": 0.0,
        "omega_rpm": 850.0,
        "moi_spin": 0.004,
    }.items():
        s.set(name, value)
    traj.initialize(vehicle, None)
    return vehicle, env, traj


def test_one_dnt_step_matches_cpp_velocityxd():
    vehicle, env, traj = _rect_ready()
    dt = 0.0001
    ctx = SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=[Packet(name="RECT.MR1", type="ROTOR", status=1, vars={})],
        vehicle_slot=0,
    )
    env.execute(vehicle, ctx)
    s = vehicle.store
    velocityx = s.get("velocityx")
    gamma = s.get("gamma")
    omegax = s.get("omegax")
    cd, clw, cma, cmdw = 1.31, 2.51, 0.508, -0.45
    mass, ref_area, ref_length = 1.5, 0.0468, 0.0625
    velocity_ss = s.get("velocity_ss")
    rho = s.get("rho")
    grav = s.get("grav")
    tau = 2 * mass / (rho * ref_area * velocity_ss)
    mu = 2 * mass / (rho * ref_area * ref_length)
    moi_spinx = 0.004 / (ref_length**2 * mu**2 * mass)
    velocityxd_new = -cd * velocityx * velocityx - tau * grav * np.sin(gamma) / velocity_ss
    vx = integrate(velocityxd_new, 0.0, velocityx, dt)
    traj.execute(vehicle, ctx)
    np.testing.assert_allclose(s.get("velocityx"), vx, rtol=1e-12)
    np.testing.assert_allclose(s.get("moi_spinx"), moi_spinx, rtol=1e-12)
    np.testing.assert_allclose(s.get("time"), tau * 0.0, rtol=1e-12)
    assert s.get("hbe") > 990.0
    assert vehicle.health == 1


def test_ground_impact_sets_health_no_sys_exit():
    vehicle, env, traj = _rect_ready(hbg=10000.0)
    ctx = SimContext(
        sim_time=0.0,
        int_step=0.0001,
        event_time=0.0,
        out_fact=0.0,
        combus=[Packet(name="RECT.MR1", type="ROTOR", status=1, vars={})],
        vehicle_slot=0,
    )
    env.execute(vehicle, ctx)
    traj.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[0].status == 0


def test_sbel_dt_is_int_step_times_tau_and_time_at_nonzero_sim_time():
    vehicle, env, traj = _rect_ready()
    dt = 0.0001
    sim_time = 0.01
    ctx = SimContext(
        sim_time=sim_time,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=[Packet(name="RECT.MR1", type="ROTOR", status=1, vars={})],
        vehicle_slot=0,
    )
    env.execute(vehicle, ctx)
    s = vehicle.store
    sbel0 = np.array(s.get("SBEL"), dtype=float)
    sbeld0 = np.array(s.get("SBELD"), dtype=float)
    velocityx = s.get("velocityx")
    velocityxd = s.get("velocityxd")
    gamma = s.get("gamma")
    gammaxd = s.get("gammaxd")
    omegax = s.get("omegax")
    omegaxd = s.get("omegaxd")
    cd, clw, cma, cmdw = 1.31, 2.51, 0.508, -0.45
    mass, ref_area, ref_length = 1.5, 0.0468, 0.0625
    velocity_ss = s.get("velocity_ss")
    rho, grav, psivlx = s.get("rho"), s.get("grav"), s.get("psivlx")
    tau = 2 * mass / (rho * ref_area * velocity_ss)
    mu = 2 * mass / (rho * ref_area * ref_length)
    moi_spinx = 0.004 / (ref_length**2 * mu**2 * mass)
    velocityxd_new = -cd * velocityx**2 - tau * grav * np.sin(gamma) / velocity_ss
    vx = integrate(velocityxd_new, velocityxd, velocityx, dt)
    gammaxd_new = clw * omegax / mu - tau * grav * np.cos(gamma) / (velocity_ss * vx)
    gamma_new = integrate(gammaxd_new, gammaxd, gamma, dt)
    omegaxd_new = (
        cma * vx**2 / (mu * moi_spinx)
        + cmdw * vx * omegax / (mu**2 * moi_spinx)
    )
    dvbe = vx * velocity_ss
    thtvlx = gamma_new * DEG
    tvl = mat2tr(psivlx * RAD, thtvlx * RAD)
    vbel = tvl.T @ np.array([dvbe, 0.0, 0.0])
    sbel_want = integrate(vbel, sbeld0, sbel0, dt * tau)
    sbel_wrong_dt = integrate(vbel, sbeld0, sbel0, dt)
    assert not np.allclose(sbel_want, sbel_wrong_dt, rtol=1e-9)
    traj.execute(vehicle, ctx)
    np.testing.assert_allclose(s.get("SBEL"), sbel_want, rtol=1e-12)
    np.testing.assert_allclose(s.get("time"), tau * sim_time, rtol=1e-12)
    np.testing.assert_allclose(s.get("sim_time"), sim_time, rtol=1e-12)
```

- [ ] **Step 2:** FAIL `pytest tests/unit/test_rotor_trajectory_step.py -v`
- [ ] **Step 3:** Port `trajectory()` including SBEL `dt=int_step*tau` and impact. Use `mat2tr` then `TVL.T @ VBEV`.
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: MAGSIX DNU trajectory step`

---

### Task 5: Rotor attitude init

**Files:** Modify `rotor.py`; Test: `Python/tests/unit/test_rotor_attitude_init.py`

**Interfaces:**
- Consumes: `tau` from trajectory init
- Produces: `RotorAttitude.name=="attitude"`. `define` C++ `def_attitude` names (`nonlinear`,`moi_trans`,`cyb`,`cyb3`,`clwb`,`clp`,`clwb3`,`clp3`,`clwb2p`,`clwbp2`,`cnb`,`cnr`,`cnb3`,`cnr3`,`cnb2r`,`cnbr2`,`beta`,`betad`,`phi`,`phid`,`phidd`,`psi`,`psid`,`psidd`,`betax`,`phix`,`ppx`,`psix`,`rrx`). `initialize`: `beta=betax*RAD`, `phi=phix*RAD`, `phid=ppx*RAD*tau`, `psi=psix*RAD`, `psid=rrx*RAD*tau`. RECT.MR1: `betax=0`, `phix=3`, `psix=0`, `rrx=-40`, `ppx` default 0.

- [ ] **Step 1: Write the failing test**

```python
from types import SimpleNamespace

import numpy as np

from cadac.constants import RAD
from cadac.eom.rotor import RotorAttitude, RotorTrajectory
from cadac.kernel.state import StateStore


def test_name_is_attitude():
    assert RotorAttitude().name == "attitude"


def test_rect_mr1_attitude_init_rates_in_dnu():
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    traj = RotorTrajectory()
    att = RotorAttitude()
    traj.define(vehicle)
    att.define(vehicle)
    for name, value in {
        "cd": 1.31,
        "cmdw": -0.45,
        "clw": 2.51,
        "cma": 0.508,
        "mass": 1.5,
        "ref_area": 0.0468,
        "ref_length": 0.0625,
        "dvbe": 16.6,
        "psivlx": 0.0,
        "thtvlx": -77.0,
        "hbe": 1000.0,
        "sbel1": 0.0,
        "sbel2": 0.0,
        "omega_rpm": 850.0,
        "moi_spin": 0.004,
        "betax": 0.0,
        "phix": 3.0,
        "ppx": 0.0,
        "psix": 0.0,
        "rrx": -40.0,
        "moi_trans": 0.0268,
        "nonlinear": 0,
    }.items():
        s.set(name, value)
    traj.initialize(vehicle, None)
    att.initialize(vehicle, None)
    tau = s.get("tau")
    np.testing.assert_allclose(s.get("beta"), 0.0, rtol=1e-12)
    np.testing.assert_allclose(s.get("phi"), 3.0 * RAD, rtol=1e-12)
    np.testing.assert_allclose(s.get("phid"), 0.0, rtol=1e-12)
    np.testing.assert_allclose(s.get("psi"), 0.0, rtol=1e-12)
    np.testing.assert_allclose(s.get("psid"), -40.0 * RAD * tau, rtol=1e-12)
```

- [ ] **Step 2:** FAIL `pytest tests/unit/test_rotor_attitude_init.py -v`
- [ ] **Step 3:** Port `def_attitude` / `init_attitude`
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: MAGSIX attitude DNU initialization`

---

### Task 6: Rotor attitude step (nonlinear 0 and 1)

**Files:** Modify `rotor.py`; Test: `Python/tests/unit/test_rotor_attitude_step.py`

**Interfaces:**
- Consumes: trajectory execute (writes `mu`,`moi_spinx`,`velocityx`,`velocityxd`,`gamma`,`omegax`,`tau`); `grav`,`mass`,`ref_length`,`velocity_ss`
- Produces: `execute` ports `Rotor::attitude` Table 12.1 eqs. 4–8 in C++ order: integrate `beta` first; `phidd_new` uses the **new** `beta`; equal-slope `phi`; `psidd_new` uses the **post-update** `phid`; then `psi`. `nonlinear in {0, 1}` else `ValueError`. Cubic terms as C++ (`/6` factors). Outputs deg / deg/s via `/tau`. Do not read a fake `eng_ang_mom`.

- [ ] **Step 1: Write the failing test**

```python
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG
from cadac.eom.rotor import RotorAttitude, RotorEnvironment, RotorTrajectory
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import StateStore


def _attitude_ready(nonlinear=0):
    s = StateStore()
    vehicle = SimpleNamespace(store=s, health=1, name="RECT.MR1")
    env, traj, att = RotorEnvironment(), RotorTrajectory(), RotorAttitude()
    env.define(vehicle)
    traj.define(vehicle)
    att.define(vehicle)
    for name, value in {
        "cd": 1.31,
        "cmdw": -0.45,
        "clw": 2.51,
        "cma": 0.508,
        "mass": 1.5,
        "ref_area": 0.0468,
        "ref_length": 0.0625,
        "dvbe": 16.6,
        "psivlx": 0.0,
        "thtvlx": -77.0,
        "hbe": 1000.0,
        "hbg": 0.0,
        "sbel1": 0.0,
        "sbel2": 0.0,
        "omega_rpm": 850.0,
        "moi_spin": 0.004,
        "betax": 0.0,
        "phix": 3.0,
        "ppx": 0.0,
        "psix": 0.0,
        "rrx": -40.0,
        "moi_trans": 0.0268,
        "nonlinear": nonlinear,
        "cyb": -3.82,
        "clwb": -0.357,
        "clp": -5.82,
        "cnb": -0.737,
        "cnr": -13.8,
        "cyb3": 0.0,
        "clwb3": 0.0,
        "clp3": 0.0,
        "clwb2p": 0.0,
        "clwbp2": 0.0,
        "cnb3": 0.0,
        "cnr3": 0.0,
        "cnb2r": 0.0,
        "cnbr2": 0.0,
    }.items():
        s.set(name, value)
    traj.initialize(vehicle, None)
    att.initialize(vehicle, None)
    return vehicle, env, traj, att


def test_nonlinear0_one_step_beta_matches_cpp():
    vehicle, env, traj, att = _attitude_ready(0)
    dt = 0.0001
    ctx = SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=[Packet(name="RECT.MR1", type="ROTOR", status=1, vars={})],
        vehicle_slot=0,
    )
    env.execute(vehicle, ctx)
    traj.execute(vehicle, ctx)
    s = vehicle.store
    beta, betad = s.get("beta"), s.get("betad")
    phi, psi = s.get("phi"), s.get("psi")
    psid = s.get("psid")
    velocityx, velocityxd = s.get("velocityx"), s.get("velocityxd")
    gamma, tau = s.get("gamma"), s.get("tau")
    grav, velocity_ss = s.get("grav"), s.get("velocity_ss")
    cyb = -3.82
    betad_new = (
        (-velocityxd / velocityx + velocityx * cyb) * beta
        - psid
        + tau * grav * np.cos(gamma) / (velocityx * velocity_ss) * phi
        + tau * grav * np.sin(gamma) / (velocityx * velocity_ss) * psi
    )
    beta_want = integrate(betad_new, betad, beta, dt)
    att.execute(vehicle, ctx)
    np.testing.assert_allclose(s.get("beta"), beta_want, rtol=1e-12)
    assert np.isfinite(s.get("phix"))
    assert s.get("nonlinear") == 0


def test_nonlinear1_adds_cyb3_term():
    vehicle, env, traj, att = _attitude_ready(1)
    vehicle.store.set("cyb3", 1.0)
    vehicle.store.set("beta", 0.1)
    dt = 0.0001
    ctx = SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=[Packet(name="RECT.MR1", type="ROTOR", status=1, vars={})],
        vehicle_slot=0,
    )
    env.execute(vehicle, ctx)
    traj.execute(vehicle, ctx)
    s = vehicle.store
    beta, betad = s.get("beta"), s.get("betad")
    phi, psi, psid = s.get("phi"), s.get("psi"), s.get("psid")
    velocityx, velocityxd = s.get("velocityx"), s.get("velocityxd")
    gamma, tau = s.get("gamma"), s.get("tau")
    grav, velocity_ss = s.get("grav"), s.get("velocity_ss")
    betad_new = (
        (-velocityxd / velocityx + velocityx * (-3.82)) * beta
        - psid
        + tau * grav * np.cos(gamma) / (velocityx * velocity_ss) * phi
        + tau * grav * np.sin(gamma) / (velocityx * velocity_ss) * psi
        + 1 * (velocityx * 1.0 * beta**3 / 6.0)
    )
    beta_want = integrate(betad_new, betad, beta, dt)
    att.execute(vehicle, ctx)
    np.testing.assert_allclose(s.get("beta"), beta_want, rtol=1e-12)


def test_nonlinear_2_raises():
    vehicle, env, traj, att = _attitude_ready(2)
    ctx = SimContext(
        sim_time=0.0,
        int_step=0.0001,
        event_time=0.0,
        out_fact=0.0,
        combus=[Packet(name="RECT.MR1", type="ROTOR", status=1, vars={})],
        vehicle_slot=0,
    )
    env.execute(vehicle, ctx)
    traj.execute(vehicle, ctx)
    with pytest.raises(ValueError, match="nonlinear"):
        att.execute(vehicle, ctx)


def test_phi_uses_new_beta_then_psidd_uses_new_phid():
    vehicle, env, traj, att = _attitude_ready(0)
    dt = 0.0001
    ctx = SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=[Packet(name="RECT.MR1", type="ROTOR", status=1, vars={})],
        vehicle_slot=0,
    )
    env.execute(vehicle, ctx)
    traj.execute(vehicle, ctx)
    s = vehicle.store
    beta, betad = s.get("beta"), s.get("betad")
    phi, phid, phidd = s.get("phi"), s.get("phid"), s.get("phidd")
    psi, psid, psidd = s.get("psi"), s.get("psid"), s.get("psidd")
    velocityx, velocityxd = s.get("velocityx"), s.get("velocityxd")
    gamma, tau = s.get("gamma"), s.get("tau")
    grav, velocity_ss = s.get("grav"), s.get("velocity_ss")
    omegax = s.get("omegax")
    mu, moi_spinx = s.get("mu"), s.get("moi_spinx")
    moi_transx = 0.0268 / (0.0625**2 * mu**2 * 1.5)
    betad_new = (
        (-velocityxd / velocityx + velocityx * (-3.82)) * beta
        - psid
        + tau * grav * np.cos(gamma) / (velocityx * velocity_ss) * phi
        + tau * grav * np.sin(gamma) / (velocityx * velocity_ss) * psi
    )
    beta_new = integrate(betad_new, betad, beta, dt)
    phidd_new = (
        velocityx * omegax * (-0.357) / (mu * mu * moi_transx) * beta_new
        + velocityx * (-5.82) / (mu * mu * moi_transx) * phid
        + moi_spinx * omegax / moi_transx * psid
    )
    phid_new = integrate(phidd_new, phidd, phid, dt)
    phi_want = integrate(phid_new, phid_new, phi, dt)
    psidd_new = (
        velocityx * velocityx * (-0.737) / (mu * moi_transx) * beta_new
        - moi_spinx * omegax / moi_transx * phid_new
        + velocityx * (-13.8) / (mu * mu * moi_transx) * psid
    )
    psid_new = integrate(psidd_new, psidd, psid, dt)
    psi_want = integrate(psid_new, psid_new, psi, dt)
    att.execute(vehicle, ctx)
    np.testing.assert_allclose(s.get("beta"), beta_new, rtol=1e-12)
    np.testing.assert_allclose(s.get("phid"), phid_new, rtol=1e-12)
    np.testing.assert_allclose(s.get("phi"), phi_want, rtol=1e-12)
    np.testing.assert_allclose(s.get("psid"), psid_new, rtol=1e-12)
    np.testing.assert_allclose(s.get("psi"), psi_want, rtol=1e-12)
    np.testing.assert_allclose(s.get("ppx"), phid_new * DEG / tau, rtol=1e-12)
```

- [ ] **Step 2:** FAIL `pytest tests/unit/test_rotor_attitude_step.py -v`
- [ ] **Step 3:** Port `attitude.cpp` including cubic `nonlinear*` terms and equal-slope angle integrates
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: MAGSIX Magnus attitude DNU step`

---

### Task 7: ROTOR vehicle + translate attitude `input.asc`

**Files:**
- Create: `Python/src/cadac/vehicles/planar/magsix/__init__.py`, `Python/src/cadac/vehicles/planar/magsix/vehicle.py`
- Modify: `Python/src/cadac/cli.py`
- Translate: `CADAC_Simulations/MAGSIX_231111/MAGSIX/input.asc` → `Python/cases/magsix/input.jsonc` with `family="magsix"`
- Test: `Python/tests/unit/test_rotor_one_step.py`
- Modify unknown-type tests that still use `"AIM5"` or `"ROTOR"` as the unregistered token: always retarget to `"NO_SUCH_TYPE"` (including AIM5's `test_family_none_unknown_rotor_raises` if present). Do not leave `"AIM5"` as sentinel.

**Interfaces:**
- Consumes: `RotorEnvironment`, `RotorTrajectory`, `RotorAttitude`; Task 1 family lookup
- Produces: `Rotor.type=="ROTOR"`. `Rotor(name, events=None)`. `modules` order: environment, trajectory, attitude (always composed and defined). `define` skip-if-exists on name collisions (Hyper6 pattern). `run_scenario` `ROTOR -> Rotor` via **both** `_VEHICLE_TYPES["ROTOR"]` and `_VEHICLE_FAMILIES[("magsix","ROTOR")]`. `"ROTOR"` in `_NO_DECK_TYPES`. Smoke: `end_time` 0.01 DNT, `hbe` near 1000, `phix` finite. Case `end_time==0.35`, `int_step==0.0001`, `plot_step==0.005`, params match the spec table, no `aero_deck`/`prop_deck`.

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.io.jsonc import loads
from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc
from cadac.kernel.state import Field

ROOT = Path(__file__).resolve().parents[3]
MAGSIX_ASC = ROOT / "CADAC_Simulations/MAGSIX_231111/MAGSIX"
CASES = Path(__file__).resolve().parents[2] / "cases" / "magsix"
MODULE_ORDER = ["environment", "trajectory", "attitude"]


def _attitude_smoke(tmp_path: Path) -> Path:
    translate_scenario_asc(MAGSIX_ASC / "input.asc", tmp_path, family="magsix")
    path = tmp_path / "input.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.01
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def test_rotor_type_health_and_module_order():
    from cadac.vehicles.planar.magsix.vehicle import Rotor

    vehicle = Rotor("RECT.MR1")
    assert vehicle.type == "ROTOR"
    assert vehicle.health == 1
    assert [module.name for module in vehicle.modules] == MODULE_ORDER


def test_rotor_define_skips_existing_field():
    from cadac.vehicles.planar.magsix.vehicle import Rotor

    vehicle = Rotor("RECT.MR1")
    vehicle.store.define(Field("hbe", 42.0, "real", "out", "pre", ("plot",)))
    vehicle.define()
    assert vehicle.store.get("hbe") == 42.0
    assert vehicle.store.field("hbe").module == "pre"


def test_attitude_smoke_hbe_and_phix(tmp_path: Path):
    result = run_scenario(_attitude_smoke(tmp_path))
    assert result.plot_rows
    row = result.plot_rows[-1]
    assert 990.0 < row["hbe"] < 1010.0
    assert abs(row["phix"]) < 180.0
    for value in row.values():
        assert value == value  # finite


def test_committed_attitude_case_is_rect_mr1():
    cfg = load_scenario(CASES / "input.jsonc")
    assert cfg.end_time == 0.35
    assert cfg.timing["int_step"] == 0.0001
    assert cfg.timing["plot_step"] == 0.005
    v = cfg.vehicles[0]
    assert v.type == "ROTOR"
    assert v.family == "magsix"
    assert v.name == "RECT.MR1"
    assert v.aero_deck is None
    assert v.prop_deck is None
    assert v.params["hbe"] == 1000
    assert v.params["dvbe"] == 16.6
    assert v.params["omega_rpm"] == 850
    assert v.params["thtvlx"] == -77
    assert v.params["nonlinear"] == 0
    assert v.params["phix"] == 3
    assert v.params["rrx"] == -40
    assert v.params["moi_spin"] == 0.004
    assert v.params["moi_trans"] == 0.0268


def test_rotor_does_not_require_decks(tmp_path: Path):
    path = tmp_path / "rotor.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.0001}, "end_time": 0, '
        '"vehicles": [ { "type": "ROTOR", "name": "r", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    run_scenario(path)


def test_family_magsix_rotor_runs(tmp_path: Path):
    result = run_scenario(_attitude_smoke(tmp_path))
    assert result.plot_rows[0]["hbe"] == pytest.approx(1000.0, rel=1e-3)


def test_unknown_type_no_such_type_still_raises(tmp_path: Path):
    path = tmp_path / "unknown.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.0001}, "end_time": 0, '
        '"vehicles": [ { "type": "NO_SUCH_TYPE", "name": "x", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="NO_SUCH_TYPE"):
        run_scenario(path)
```

- [ ] **Step 2:** FAIL (Rotor missing / unknown type ROTOR)
- [ ] **Step 3:** Implement `Rotor` vehicle (skip-if-exists define). Register both tables. `_NO_DECK_TYPES` includes `"ROTOR"`. Translate `input.asc` with `family="magsix"` into `Python/cases/magsix/`. Grep tests for unknown-type `"AIM5"` or `"ROTOR"` and retarget **all** of them to `"NO_SUCH_TYPE"` (`match="NO_SUCH_TYPE"`). Do not leave `"AIM5"` as sentinel.
- [ ] **Step 4:** PASS `pytest tests/unit/test_rotor_one_step.py tests/unit/test_rotor_family.py tests/unit/test_cruise3_one_step.py tests/unit/test_plane5_one_step.py tests/unit/test_hyper5_one_step.py tests/unit/test_hyper6_one_step.py -q`
- [ ] **Step 5: Commit** `feat: run ROTOR from JSONC attitude RECT.MR1`

---

### Task 8: Translate trajectory RECT.MR1 + smoke

**Files:**
- Translate: `CADAC_Simulations/MAGSIX_231111/MAGSIX/input_trajectoryMR1.asc` → `Python/cases/magsix/input_trajectoryMR1.jsonc` with `family="magsix"`
- Test: `Python/tests/unit/test_rotor_trajectory_case.py`

**Interfaces:**
- Consumes: Task 7 `Rotor` + factory. Scenario MODULES are `environment`,`trajectory` only — `run_loop` skips `attitude` exec (vehicle still defines attitude at zeros).
- Produces: committed case `end_time==50`, `int_step==0.001`, `plot_step==0.01`, no attitude params (`nonlinear` / `moi_trans` / `cyb` absent). Smoke `end_time` 0.1 DNT (before ground impact); `hbe` near 1000; plot rows have `hbe`/`dvbe`/`omega_rpm`. Do **not** assert `phix` absent.

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

from cadac import run_scenario
from cadac.io.jsonc import loads
from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc

ROOT = Path(__file__).resolve().parents[3]
MAGSIX_ASC = ROOT / "CADAC_Simulations/MAGSIX_231111/MAGSIX"
CASES = Path(__file__).resolve().parents[2] / "cases" / "magsix"


def test_committed_trajectory_case():
    cfg = load_scenario(CASES / "input_trajectoryMR1.jsonc")
    assert cfg.end_time == 50
    assert cfg.timing["int_step"] == 0.001
    assert cfg.timing["plot_step"] == 0.01
    assert [m.name for m in cfg.modules] == ["environment", "trajectory"]
    v = cfg.vehicles[0]
    assert v.type == "ROTOR"
    assert v.family == "magsix"
    assert v.params["hbe"] == 1000
    assert v.params["dvbe"] == 16.6
    assert v.params["omega_rpm"] == 850
    assert "nonlinear" not in v.params
    assert "moi_trans" not in v.params
    assert v.aero_deck is None


def test_trajectory_smoke_hbe_dvbe(tmp_path: Path):
    translate_scenario_asc(
        MAGSIX_ASC / "input_trajectoryMR1.asc", tmp_path, family="magsix"
    )
    path = tmp_path / "input_trajectoryMR1.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    result = run_scenario(path)
    row = result.plot_rows[-1]
    assert 990.0 < row["hbe"] < 1010.0
    assert "dvbe" in row
    assert "omega_rpm" in row
    assert "sim_time" in row
```

- [ ] **Step 2:** FAIL (missing case file)
- [ ] **Step 3:** Translate with `family="magsix"`. No new physics.
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: MAGSIX trajectory RECT.MR1 JSONC case`

---

### Task 9: MAGSIX attitude e2e golden (optional file)

**Files:** `Python/tests/e2e/test_magsix_attitude.py`

**Interfaces:** Skip without `tests/e2e/goldens/magsix/plot.csv`. Else compare **all** shared plot-flagged columns (not only `hbe`/`dvbe`). Align rows on `sim_time` if both sides have it, else `time`. Sentinel `time=-1`. Require `hbe`, `dvbe`, and `sim_time` in the shared set (no `alt`). Case `Python/cases/magsix/input.jsonc` (`plot_step` 0.005). `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import numpy as np
import pytest

from cadac import run_scenario
from cadac.io.plot import write_plot_csv

CASE = Path(__file__).resolve().parents[2] / "cases" / "magsix" / "input.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "magsix" / "plot.csv"
RTOL = 1e-5


def _csv_atol(golden):
    return max(1e-6, 5e-6 * abs(golden))


def load_cadac_plot_csv(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    columns = [col for col in lines[2].split(",") if col]
    rows = []
    for line in lines[3:]:
        if not line.strip():
            continue
        values = [float(item) for item in line.split(",") if item != ""]
        row = dict(zip(columns, values))
        if row["time"] == -1:
            continue
        rows.append(row)
    return columns, rows


def _align_key(row):
    return "sim_time" if "sim_time" in row else "time"


def _row_at(rows, key, value):
    for row in rows:
        if abs(row[key] - value) < 1e-9:
            return row
    raise KeyError((key, value))


def require_golden(path: Path):
    if not path.is_file():
        pytest.skip("MAGSIX attitude golden plot.csv is absent")


def test_require_golden_skips_when_missing(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="MAGSIX attitude golden"):
        require_golden(tmp_path / "plot.csv")


def test_hbe_t0_matches_within_csv_tolerances(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(path, "magsix", ["time", "sim_time", "hbe"], [[0.0, 0.0, 1000.0]])
    require_golden(path)
    _, golden_rows = load_cadac_plot_csv(path)
    key = _align_key(golden_rows[0])
    got = _row_at([{"time": 0.0, "sim_time": 0.0, "hbe": 1000.0}], key, 0.0)["hbe"]
    want = _row_at(golden_rows, key, 0.0)["hbe"]
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=_csv_atol(want))


def test_hbe_matches_golden_at_t0():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    _, golden_rows = load_cadac_plot_csv(GOLDEN)
    key = _align_key(golden_rows[0]) if "sim_time" in result.plot_rows[0] else "time"
    got = _row_at(result.plot_rows, key, 0.0)["hbe"]
    want = _row_at(golden_rows, key, 0.0)["hbe"]
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=_csv_atol(want))


def test_all_shared_plot_columns_match_golden_at_shared_times():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    columns, golden_rows = load_cadac_plot_csv(GOLDEN)
    python_keys = result.plot_rows[0].keys()
    shared = [column for column in columns if column in python_keys]
    assert "hbe" in shared
    assert "dvbe" in shared
    assert "sim_time" in shared
    key = "sim_time" if "sim_time" in shared else "time"
    compare = [column for column in shared if column != key]
    assert compare
    for golden in golden_rows:
        python = _row_at(result.plot_rows, key, golden[key])
        for column in compare:
            want = golden[column]
            np.testing.assert_allclose(
                python[column],
                want,
                rtol=RTOL,
                atol=_csv_atol(want),
                err_msg=f"{column} at {key}={golden[key]}",
            )
```

- [ ] **Step 2:** `pytest tests/e2e/test_magsix_attitude.py -v` — skip path PASS without golden; t0 helper PASS
- [ ] **Step 3:** Keep helpers local (do not import HYPER6 e2e module)
- [ ] **Step 4:** PASS (skip or compare)
- [ ] **Step 5: Commit** `test: MAGSIX attitude e2e gate (skip without golden)`

---

### Task 10: MAGSIX trajectory e2e golden (optional file)

**Files:** `Python/tests/e2e/test_magsix_trajectory.py`

**Interfaces:** Skip without `tests/e2e/goldens/magsix/trajectory/plot.csv`. Case `Python/cases/magsix/input_trajectoryMR1.jsonc` (`plot_step` 0.01). Compare **all** shared plot-flagged columns. Align on `sim_time` when present. Require `hbe`, `dvbe`, and `sim_time`. Ground impact is allowed (C++ marks dead; executive skips further exec).

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import numpy as np
import pytest

from cadac import run_scenario
from cadac.io.plot import write_plot_csv

CASE = (
    Path(__file__).resolve().parents[2] / "cases" / "magsix" / "input_trajectoryMR1.jsonc"
)
GOLDEN = (
    Path(__file__).resolve().parent / "goldens" / "magsix" / "trajectory" / "plot.csv"
)
RTOL = 1e-5


def _csv_atol(golden):
    return max(1e-6, 5e-6 * abs(golden))


def load_cadac_plot_csv(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    columns = [col for col in lines[2].split(",") if col]
    rows = []
    for line in lines[3:]:
        if not line.strip():
            continue
        values = [float(item) for item in line.split(",") if item != ""]
        row = dict(zip(columns, values))
        if row["time"] == -1:
            continue
        rows.append(row)
    return columns, rows


def _row_at(rows, key, value):
    for row in rows:
        if abs(row[key] - value) < 1e-9:
            return row
    raise KeyError((key, value))


def require_golden(path: Path):
    if not path.is_file():
        pytest.skip("MAGSIX trajectory golden plot.csv is absent")


def test_require_golden_skips_when_missing(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="MAGSIX trajectory golden"):
        require_golden(tmp_path / "plot.csv")


def test_hbe_matches_golden_at_t0():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    _, golden_rows = load_cadac_plot_csv(GOLDEN)
    key = "sim_time" if "sim_time" in result.plot_rows[0] and "sim_time" in golden_rows[0] else "time"
    got = _row_at(result.plot_rows, key, 0.0)["hbe"]
    want = _row_at(golden_rows, key, 0.0)["hbe"]
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=_csv_atol(want))


def test_all_shared_plot_columns_match_golden_at_shared_times():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    columns, golden_rows = load_cadac_plot_csv(GOLDEN)
    python_keys = result.plot_rows[0].keys()
    shared = [column for column in columns if column in python_keys]
    assert "hbe" in shared
    assert "dvbe" in shared
    assert "sim_time" in shared
    key = "sim_time" if "sim_time" in shared else "time"
    compare = [column for column in shared if column != key]
    assert compare
    for golden in golden_rows:
        python = _row_at(result.plot_rows, key, golden[key])
        for column in compare:
            want = golden[column]
            np.testing.assert_allclose(
                python[column],
                want,
                rtol=RTOL,
                atol=_csv_atol(want),
                err_msg=f"{column} at {key}={golden[key]}",
            )
```

- [ ] **Step 2:** `pytest tests/e2e/test_magsix_trajectory.py -v` — skip path PASS without golden
- [ ] **Step 3:** Implement the e2e module as in Step 1 (helpers live in this file)
- [ ] **Step 4:** PASS `pytest tests/e2e/test_magsix_trajectory.py tests/e2e/test_magsix_attitude.py tests/unit/test_rotor_environment.py tests/unit/test_rotor_trajectory_init.py tests/unit/test_rotor_trajectory_step.py tests/unit/test_rotor_attitude_init.py tests/unit/test_rotor_attitude_step.py tests/unit/test_rotor_one_step.py tests/unit/test_rotor_trajectory_case.py -q`
- [ ] **Step 5: Commit** `test: MAGSIX trajectory e2e gate (skip without golden)`

---

## Self-review

- Spec coverage: family API (T1, AIM5 loader kept, no clobber of `test_vehicle_family.py`), environment (T2), trajectory init/step/impact/SBEL `dt*tau`/nonzero `time` (T3–T4), attitude init/step with C++ beta→phi(new beta)→psi(new phid) (T5–T6), vehicle+attitude case+`NO_SUCH_TYPE` (T7), trajectory case without `phix` absent (T8), e2e all shared columns (T9–T10). `sim_time` plot-flagged. `plot_step` 0.005 / 0.01 locked.
- Non-goals: mwind≠0, multi-run, `eng_ang_mom`, sys.exit, Flat6 reuse, constants.py RPM — no tasks.
- Placeholder scan: no TBD; Task 6 phi replica is inside the Step 1 fence.
- Types: `Rotor(name, events=None)`, `translate_scenario_asc(src, dst_dir, family=None)`, `VehicleSpec.family`, module names `environment`/`trajectory`/`attitude`.
