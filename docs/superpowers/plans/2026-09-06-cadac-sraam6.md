# CADAC SRAAM6 (MISSILE6 / TARGET3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run SRAAM6 `input_1v1.asc` from JSONC (`family="sraam6"`, types `MISSILE6` + `TARGET3`); unit-test each C++ module used by that case; e2e vs CADAC CSV when a golden exists.

**Architecture:** Family registry for ambiguous `TARGET3`. Missile: port SRAAM6 vehicle modules; reuse `cadac.eom.flat6.Flat6Newton` only. Target: Flat3 aircraft names (`SAEL`/`dvae`), not HYPER5 Round3 Target3. No Plane6/SAM6/AGM6 imports.

**Tech Stack:** Python >= 3.11, numpy, pytest. Work in `Python/`.

**Spec:** `docs/superpowers/specs/2026-09-06-cadac-sraam6-design.md`

**Depends on:** HYPER5 kernel skip + deck-optional factory (shipped). Family API is added here if missing.

## Global Constraints

- Spec + parent `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
- Do not regress HYPER3 e2e, HYPER5/HYPER6 units, FALCON5/FALCON6 units
- Port `CADAC_Simulations/SRAAM6_250130/SRAAM6/` numerics; do not import Plane6 or SAM6/AGM6
- Reuse `Flat6Newton` only. Do not reuse `Flat6Euler` / `Flat6Kinematics` / `Flat6Environment`
- AIM5 family API: `VehicleSpec.family` is the source of truth; scenario `"family"` is a default; vehicle key wins; **no** `RunConfig.family` as primary. `family` set → **only** `_VEHICLE_FAMILIES`. `family` omitted → **only** `_VEHICLE_TYPES` (HYPER5 `TARGET3` stays global). `_build_vehicle(path, spec)`. Never put SRAAM6 `TARGET3` in `_VEHICLE_TYPES`. Do not use `AIM5` as an unknown-type sentinel (it is/will be global)
- Missile: both decks required. Target: `(name, events=None)` no decks. Do not add SRAAM6 `TARGET3` to `_NO_DECK_TYPES`
- Packets by `type` `MISSILE6`/`TARGET3` + field names. `tgt_num` 1-based among TARGET3 packets
- Unused C++ mode ints → `ValueError`. No `sys.exit`. Monte Carlo out of scope
- `SMALL=1e-7` module-level; CADAC `sign` local (`<0 → -1` else `+1`); not `np.sign`; not `cadac.constants`
- Unit `rtol=1e-12`, `atol=1e-14`; CSV e2e `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`
- E2E skip if `Python/tests/e2e/goldens/sraam6/plot.csv` absent (do not create it)
- Grok `cursor-grok-4.6-high` implementer + reviewer; TDD; no Fast/Kimi
- Protocol: `define(vehicle)` / `initialize` / `execute(vehicle, ctx)` on `vehicle.store`
- Exact 1v1 ICs from `input_1v1.asc` (listed in the spec)

## File map

- `Python/src/cadac/io/scenario.py` — `VehicleSpec.family` (AIM5 canonical; no `RunConfig.family`)
- `Python/src/cadac/cli.py` — `_VEHICLE_FAMILIES`, `_build_vehicle(path, spec)`
- `Python/src/cadac/io/translate.py` — `family=` writes on each vehicle; GAUSS/MARKOV store means (do not skip if parser exists)
- `Python/src/cadac/eom/flat3.py` — `Flat3AircraftEnvironment`, `Flat3AircraftNewton`
- `Python/src/cadac/vehicles/sraam6/{__init__,vehicle,target,environment,kinematics,euler,aero,propulsion,seeker,guidance,control,actuator,forces,tvc,intercept}.py`
- Tests `Python/tests/unit/test_sraam6_*.py`
- Case `Python/cases/sraam6/` from `input_1v1.asc` + `sraam6_*_deck.asc`
- E2E `Python/tests/e2e/test_sraam6_1v1.py`
- C++: `environment.cpp`, `kinematics.cpp`, `euler.cpp`, `newton.cpp`, `aerodynamics.cpp`, `propulsion.cpp`, `seeker.cpp`, `guidance.cpp`, `control.cpp`, `actuator.cpp`, `forces.cpp`, `tvc.cpp`, `intercept.cpp`, `target_modules.cpp`, `flat3_modules.cpp`

---

### Task 1: Family registry (AIM5 API, idempotent)

**Files:**
- Modify: `Python/src/cadac/io/scenario.py`, `Python/src/cadac/cli.py` (only if AIM5 API is missing)
- Test: `Python/tests/unit/test_sraam6_family.py`

**Interfaces:**
- Consumes: existing `_VEHICLE_TYPES`, `VehicleSpec`, `load_scenario`, `_build_vehicle`
- Produces: AIM5 canonical API. `VehicleSpec.family: str | None = None`. `load_scenario`: vehicle `"family"` else scenario-level `"family"` else `None`; vehicle key wins. **No** `RunConfig.family`. `_VEHICLE_FAMILIES: dict[tuple[str,str], type]` (empty until Task 25 if not already present). `_build_vehicle(path, spec)` — two args. If `spec.family is not None`: **only** `_VEHICLE_FAMILIES[(spec.family, spec.type)]`; miss → `ValueError` with type and family; **no** `_VEHICLE_TYPES` fallthrough. If `spec.family is None`: `_VEHICLE_TYPES[spec.type]` as today. Do not put SRAAM6 TARGET3 in `_VEHICLE_TYPES`. Do not change `_NO_DECK_TYPES`. Do not add `test_unknown_aim5_without_family_still_raises` (`AIM5` is/will be global). If this API already exists (AIM5), keep it and add the tests below.

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

import pytest

from cadac.cli import _build_vehicle, run_scenario
from cadac.io.scenario import load_scenario
from cadac.vehicles.hyper5.target import Target3


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _minimal(type_name: str, *, scenario_family=None, vehicle_family=None) -> dict:
    vehicle = {"type": type_name, "name": "v", "params": {}, "events": []}
    if vehicle_family is not None:
        vehicle["family"] = vehicle_family
    body = {
        "title": "family",
        "options": {},
        "modules": [{"name": "environment", "phases": ["def", "exec"]}],
        "timing": {"int_step": 0.1},
        "end_time": 0.0,
        "vehicles": [vehicle],
    }
    if scenario_family is not None:
        body["family"] = scenario_family
    return body


def test_omitted_family_target3_is_hyper5(tmp_path: Path):
    path = _write(tmp_path / "g.jsonc", _minimal("TARGET3"))
    cfg = load_scenario(path)
    assert cfg.vehicles[0].family is None
    vehicle = _build_vehicle(path, cfg.vehicles[0])
    assert type(vehicle) is Target3


def test_scenario_family_applies_when_vehicle_omits_it(tmp_path: Path):
    path = _write(tmp_path / "s.jsonc", _minimal("TARGET3", scenario_family="sraam6"))
    assert load_scenario(path).vehicles[0].family == "sraam6"


def test_vehicle_family_overrides_scenario_family(tmp_path: Path):
    path = _write(
        tmp_path / "s.jsonc",
        _minimal("TARGET3", scenario_family="sraam6", vehicle_family="hyper5"),
    )
    assert load_scenario(path).vehicles[0].family == "hyper5"


def test_family_sraam6_does_not_fall_through_to_global_target3(tmp_path: Path):
    path = _write(tmp_path / "s.jsonc", _minimal("TARGET3", vehicle_family="sraam6"))
    cfg = load_scenario(path)
    assert cfg.vehicles[0].family == "sraam6"
    with pytest.raises(ValueError, match="TARGET3"):
        run_scenario(path)
```

Task 25 retargets `test_family_sraam6_does_not_fall_through_to_global_target3` to `("sraam6","CRUISE3")` after SRAAM6 TARGET3 is registered. Do **not** keep a no-family `AIM5` raises test.

- [ ] **Step 2:** `cd Python && python -m pytest tests/unit/test_sraam6_family.py -v` — FAIL if `VehicleSpec.family` or no-fallthrough lookup is missing
- [ ] **Step 3:** If missing: add `VehicleSpec.family`; `_vehicle(..., scenario_family=None)` sets `family=raw.get("family", scenario_family)`; `load_scenario` passes `data.get("family")`; `_VEHICLE_FAMILIES = {}`; `_build_vehicle(path, spec)` as specified. Do not add `RunConfig.family`.
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: CADAC VehicleSpec family dispatch`

---

### Task 2: Translate family on each vehicle + GAUSS means

**Files:**
- Modify: `Python/src/cadac/io/translate.py` (only if AIM5 `family=` or ROCKET6/AGM6 GAUSS means are missing)
- Test: `Python/tests/unit/test_sraam6_translate.py`

**Interfaces:**
- Consumes: `translate_scenario_asc(src, dst_dir, family=None)`
- Produces: If `family` is not `None`, **each vehicle** gets `"family": family` (not a scenario-root key). `family is None` → no `family` key on vehicles. GAUSS/MARKOV/RAYL: **do not skip** if ROCKET6/AGM6 already store means. `GAUSS name mean sigma` → `params[name]=mean`; `MARKOV` → `0` (or the stored mean if that parser already does). If that parser is still absent, add store-means — do not rip a shared translator. `MONTE` stays ignored at scenario level. Existing callers without `family=` unchanged.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from cadac.io.jsonc import loads
from cadac.io.translate import translate_scenario_asc

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "CADAC_Simulations/SRAAM6_250130/SRAAM6/input_1v1.asc"


def test_translate_1v1_family_on_each_vehicle(tmp_path: Path):
    translate_scenario_asc(SRC, tmp_path, family="sraam6")
    data = loads((tmp_path / "input_1v1.jsonc").read_text(encoding="utf-8"))
    assert "family" not in data or data.get("family") is None
    assert [v["family"] for v in data["vehicles"]] == ["sraam6", "sraam6"]
    assert data["end_time"] == 12
    assert data["vehicles"][0]["type"] == "MISSILE6"
    assert data["vehicles"][1]["type"] == "TARGET3"
    params = data["vehicles"][0]["params"]
    assert "GAUSS" not in params
    assert params.get("biast", 0) == 0
    assert params.get("biasp", 0) == 0
    assert params.get("biaseh", 0) == 0
    assert params["tgt_num"] == 1
    assert params["mseek"] == 2
    assert params["ms1dyn"] == 1
    assert params["mprop"] == 1
    assert params["mact"] == 2
    assert params["maut"] == 2
    assert params["sbel3"] == -5000
    assert params["dvbe"] == 250
    tgt = data["vehicles"][1]["params"]
    assert tgt["tgt_option"] == 1
    assert tgt["sael1"] == 10000
    assert tgt["dvae"] == 250
    assert "aero_deck" not in data["vehicles"][1]
    assert data["vehicles"][0]["events"][0]["set"] == {"maut": 3, "mguid": 3}


def test_translate_without_family_omits_vehicle_key(tmp_path: Path):
    translate_scenario_asc(SRC, tmp_path)
    data = loads((tmp_path / "input_1v1.jsonc").read_text(encoding="utf-8"))
    assert all("family" not in v for v in data["vehicles"])
```

1v1 GAUSS means are 0 — assert stored values are 0, not that the keys are absent.

- [ ] **Step 2:** FAIL if vehicles lack `family` or GAUSS crashes the parser
- [ ] **Step 3:** Write `family` on each vehicle when set. If GAUSS parser missing, add store-means. Do not skip GAUSS/MARKOV when a sibling already stores means.
- [ ] **Step 4:** PASS; existing `tests/translate/` still pass
- [ ] **Step 5: Commit** `feat: translate ASC family on vehicles`

---

### Task 3: SRAAM6 Flat6 environment

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/__init__.py`, `Python/src/cadac/vehicles/sraam6/environment.py`
- Test: `Python/tests/unit/test_sraam6_environment.py`

**Interfaces:**
- Consumes: `atmosphere76`, `gravity` (or C++ `G*EARTH_MASS/(REARTH+hbe)**2` — same helper)
- Produces: `Sraam6Environment.name=="environment"`. `define` C++ `def_environment` (`press`,`rho`,`vsound`,`grav`,`vmach`,`pdynmc`,`tempk`,`mfreeze_environ`,`pdynmcf`,`vmachf`). Does not define `hbe`/`dvbe`/`trcond`. `execute`: US76 at `hbe`; `vmach=abs(dvbe/vsound)`; `pdynmc=0.5*rho*dvbe**2`. If `mguid==6`: `vmach<=trmach` → `trcond=2`; `pdynmc<=trdynm` → `trcond=3`. `mfreeze` latch as C++; skip `mfreeze` if absent.

- [ ] **Step 1:**

```python
import math

import numpy as np
import pytest

from cadac.constants import R
from cadac.env.us76 import atmosphere76
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sraam6.environment import Sraam6Environment

RTOL = 1e-12
ATOL = 1e-14


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ready(*, hbe=5000.0, dvbe=250.0, mguid=0, mfreeze=0, trmach=0.5, trdynm=1e4, trcond=0):
    vehicle = _Vehicle()
    env = Sraam6Environment()
    env.define(vehicle)
    for name, value, ftype in (
        ("hbe", hbe, "real"),
        ("dvbe", dvbe, "real"),
        ("mguid", mguid, "int"),
        ("mfreeze", mfreeze, "int"),
        ("trmach", trmach, "real"),
        ("trdynm", trdynm, "real"),
        ("trcond", trcond, "int"),
    ):
        vehicle.store.define(Field(name, value, ftype, "data", "ext"))
    env.execute(vehicle, SimContext(0.0, 0.001, 0.0, 0.0, None, 0))
    return vehicle.store


def test_us76_mach_at_1v1_launch():
    store = _ready()
    rho, press, tempk = atmosphere76(5000.0)
    vsound = math.sqrt(1.4 * R * tempk)
    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("vmach"), abs(250.0 / vsound), rtol=RTOL, atol=ATOL)


def test_mguid6_low_mach_sets_trcond_2():
    store = _ready(dvbe=1.0, mguid=6, trmach=0.5)
    assert store.get("trcond") == 2
```

- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 Flat6 environment US76`

---

### Task 4: SRAAM6 Flat6 kinematics

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/kinematics.py`
- Test: `Python/tests/unit/test_sraam6_kinematics.py`

**Interfaces:**
- Consumes: `mat3tr`, `integrate`, `DEG`,`RAD`,`PI`,`EPS`. Module-level `SMALL=1e-7`. Local CADAC `sign`.
- Produces: `Sraam6Kinematics.name=="kinematics"`. Port `def_kinematics` / `init_kinematics` / `kinematics` from SRAAM6 `kinematics.cpp`. Quaternion from Euler; rates **`pp`,`qq`,`rr`** (define them if absent, default 0). Incidence from **`VBEB`**, not `VBAL`. `phip`: `fabs(vbeb2)<EPS and fabs(vbeb3)<EPS` → `0`; else if `fabs(vbeb2)<SMALL` → `atan2(SMALL, vbeb3)`; else `atan2(vbeb2, vbeb3)`. `EPS` from constants; `SMALL=1e-7` module-level. `time=ctx.sim_time`; write `ctx.int_step` from `int_step_new` and `ctx.out_fact` from `out_step_fact` as C++. `trcond` vs `trcvel` (ortho) and `tralp`. Do not import `Flat6Kinematics`.

- [ ] **Step 1:** 1v1 ICs `psiblx=thtblx=phiblx=0`; after init `TBL` near identity, `q0` finite. One execute `pp=qq=rr=0`, `VBEB=(250,0,0)`: `TBL` stays orthonormal (`etbl` small), `alphax≈0`, `phip==0` (both `vbeb2`/`vbeb3` `<EPS`). `VBEB=(250, 1e-8, 1.0)` (`|vbeb2|<SMALL`, `|vbeb3|>=EPS`) → `phip==atan2(SMALL, 1.0)`. `fabs(ortho_error)>trcvel` with a broken quaternion sets `trcond=1`.
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 Flat6 quaternion kinematics`

---

### Task 5: SRAAM6 Flat6 euler

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/euler.py`
- Test: `Python/tests/unit/test_sraam6_euler.py`

**Interfaces:**
- Consumes: `integrate`
- Produces: `Sraam6Euler.name=="euler"`. Port `def_euler` / `euler` (`ppd`,`pp`,`qqd`,`qq`,`rrd`,`rr`,`ppx`,`qqx`,`rrx`,`WBEB`). `ppd_new=FMB[0]/ai11`; pitch/yaw as C++ with `ai33`. Degrees on `ppx`/`qqx`/`rrx`. No `IBBB`. No `Flat6Euler`.

- [ ] **Step 1:**

```python
import numpy as np

from cadac.constants import DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sraam6.euler import Sraam6Euler

RTOL = 1e-12
ATOL = 1e-14


def test_zero_moment_holds_roll_rate():
    vehicle = type("V", (), {"store": StateStore()})()
    euler = Sraam6Euler()
    euler.define(vehicle)
    store = vehicle.store
    store.define(Field("FMB", (0.0, 0.0, 0.0), "vec", "out", "forces"))
    store.define(Field("ai11", 0.308, "real", "out", "propulsion"))
    store.define(Field("ai33", 59.80, "real", "out", "propulsion"))
    store.set("pp", 0.1)
    euler.execute(vehicle, SimContext(0.0, 0.001, 0.0, 0.0, None, 0))
    np.testing.assert_allclose(store.get("pp"), 0.1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ppx"), 0.1 * DEG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WBEB"), [0.1, 0.0, 0.0], rtol=RTOL, atol=ATOL)


def test_roll_moment_integrates_pp():
    vehicle = type("V", (), {"store": StateStore()})()
    euler = Sraam6Euler()
    euler.define(vehicle)
    store = vehicle.store
    store.define(Field("FMB", (0.308, 0.0, 0.0), "vec", "out", "forces"))
    store.define(Field("ai11", 0.308, "real", "out", "propulsion"))
    store.define(Field("ai33", 59.80, "real", "out", "propulsion"))
    ppd_new = 0.308 / 0.308
    expected = integrate(ppd_new, 0.0, 0.0, 0.001)
    euler.execute(vehicle, SimContext(0.0, 0.001, 0.0, 0.0, None, 0))
    np.testing.assert_allclose(store.get("pp"), expected, rtol=RTOL, atol=ATOL)
```

- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 Euler ai11/ai33`

---

### Task 6: SRAAM6 aerodynamics tables

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/aero.py`
- Test: `Python/tests/unit/test_sraam6_aero.py`

**Interfaces:**
- Consumes: `Datadeck.look_up`, `AGRAV`
- Produces: `Sraam6Aero(deck).name=="aerodynamics"`. `define` C++ `def_aerodynamics` (including `alplimx`,`trcvel`,`trmach`,`trdynm`,`trload`,`tralp`,`trtht`,`trthtd`,`trphid`,`trate`,`trcond`,`gmax`,`gavail`, coefficients). `initialize` sets `refl=0.1524`, `refa=0.01824`, `trcvel=10e-4`, `trmach=0.5`, `trdynm=10e+3`, `trload=3`, `tralp=1`, `trcond=0`. `execute` ports `Missile::aerodynamics` (not `_der` yet). Parse `sraam6_aero_deck.asc` in tests. `gmax<trload` → `trcond=4`.

- [ ] **Step 1:** 1v1-like: `vmach` from 250 m/s at 5 km, `alppx=0`, `phip=0`, `ppx=qqx=rrx=0`, `mprop=1`, `vmass=92`, `xcgref=xcg=1.536`, `alimit=50`, `dpx=dqx=drx=0`, `alplimx=46`. `ca`/`cn`/`cy` match look_up + C++ formulas, rtol 1e-12. `initialize` overwrites `trcvel` to `1e-3`.
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 aerodynamics tables`

---

### Task 7: aerodynamics_der

**Files:** Modify `aero.py`; Test: `Python/tests/unit/test_sraam6_aero_der.py`

**Interfaces:** Port `Missile::aerodynamics_der` exactly. Call from `aerodynamics()` after loading coefficients. Bypass if `alppx>=alplimx-3` (keep saved `dna`…). Roots `wnq`/`zetq`/`realq*` as C++.

- [ ] **Step 1:** same flight point as Task 6 plus `ai11=0.308`, `ai33=59.80`; `dna`/`dma` finite vs C++ finite-difference formulas rtol 1e-12; `alppx=44`, `alplimx=46` does not change a pre-set `dna`
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 aero derivatives`

---

### Task 8: SRAAM6 propulsion

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/propulsion.py`
- Test: `Python/tests/unit/test_sraam6_propulsion.py`

**Interfaces:** Port `Missile::propulsion`. Constructor takes Datadeck. `mprop==0` thrust 0. `mprop==1` tables `thrust_vs_time`,`mass_vs_time`,`cg_vs_time`,`moipitch_vs_time`,`moiroll_vs_time`; `thrust=tsl+(101325-press)*aexit`; `time>2.69` → `mprop=0`. Else `ValueError`. `mfreeze` latch as C++. Parse `sraam6_prop_deck.asc`. Defaults `vmass=92`, `xcg=1.536`, `ai11=0.308`, `ai33=59.80`.

- [ ] **Step 1:** `mprop=1`, `time=0`, `aexit=0.0125`, `press` from US76 at 5000 m → thrust vs C++ formula rtol 1e-12; `mprop=0` thrust 0; `mprop=2` raises; `time=2.70`, `mprop=1` → `mprop==0` and thrust 0
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 rocket propulsion`

---

### Task 9: SRAAM6 actuator

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/actuator.py`
- Test: `Python/tests/unit/test_sraam6_actuator.py`

**Interfaces:** Port `actuator.cpp`. `mact==0` limit-only four fins; `mact==2` `actuator_scnd`. Else `ValueError`. Mix: `delcx1=-dpcx+dqcx-drcx` etc. Recover `dpx,dqx,drx` as C++. Local CADAC `sign`. `dt=ctx.int_step`. 1v1: `mact=2`, `dlimx=28`, `ddlimx=600`, `wnact=100`, `zetact=0.7`.

- [ ] **Step 1:** `mact=0`, `dpcx=0`, `dqcx=40`, `drcx=0`, `dlimx=28` → `|dqx|<=28`. `mact=2`, `dqcx=1`, `dt=0.001` → `|dqx|<=dlimx` and output lags command. `mact=1` uses the `mact<2` limit path (C++). `mact=3` raises.
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 four-fin actuators`

---

### Task 10: control_roll

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/control.py`
- Test: `Python/tests/unit/test_sraam6_control_roll.py`

**Interfaces:** Port `def_control` (all C++ names) and `control_roll`. `gkp=(2*zrcl*wrcl+dlp)/dld`; `gkphi=wrcl*wrcl/dld`; `dpcx` as C++. `execute` pass until Task 13. 1v1: `phicomx=0`, `wrcl=20`, `zrcl=0.9`.

- [ ] **Step 1:** `phiblx=0`, `pp=0`, `dlp=-2`, `dld=1` → `dpcx` vs C++ rtol 1e-12; `phiblx=10` nonzero command
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 roll controller`

---

### Task 11: control_rate

**Files:** Modify `control.py`; Test: `Python/tests/unit/test_sraam6_control_rate.py`

**Interfaces:** Port `control_rate`. `zetlagr=0.6` (1v1). `SMALL` clamp on `dmd`. `execute` still pass.

- [ ] **Step 1:** frozen `dna,dnd,dma,dmq,dmd,dvbe=250,qq,rr` replica of C++ `grate`/`dqcx`/`drcx` rtol 1e-12
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 rate controller`

---

### Task 12: control_accel

**Files:** Modify `control.py`; Test: `Python/tests/unit/test_sraam6_control_accel.py`

**Interfaces:** Port `control_accel`. On-line `wacl=(0.013*sqrt(pdynmc)+7.1)*(factwacl+1)` etc, `pacl=14`. Circular `alimit` limiter. Integrate `yy`/`zz`. `execute` still pass.

- [ ] **Step 1:** 1v1 `alimit=50`, `ancomx=1`, `alcomx=0`, `pdynmc` at 5 km / 250 m/s, `dt=0.001`; `dqcx` finite vs C++ one-step; `|ancomx|`,`|alcomx|` after limiter ≤ `alimit`
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 acceleration controller`

---

### Task 13: maut dispatcher

**Files:** Modify `control.py`; Test: `Python/tests/unit/test_sraam6_maut.py`

**Interfaces:** Port `Missile::control`. `maut==0` return (no new `dpcx`). `{1,2,3}` as spec. Else `ValueError`. Wire roll/rate/accel.

- [ ] **Step 1:** `maut=0` leaves `dqcx` unchanged; `maut=2` writes `dqcx`; `maut=3` writes `dqcx`; `maut=1` writes `dpcx`; `maut=4` and `maut=-1` raise
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 maut dispatcher`

---

### Task 14: SRAAM6 forces

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/forces.py`
- Test: `Python/tests/unit/test_sraam6_forces.py`

**Interfaces:** Port `Missile::forces`. Writes `FAPB`/`FMB` only. If `mtvc` absent or `==0`, `FAPB[0]+=thrust`. Else add `FPB`/`FMPB`. Do not write `FSPB`.

- [ ] **Step 1:** `pdynmc=1e4`, `refa=0.01824`, `refl=0.1524`, `ca=0.5`, `cy=cn=cll=clm=cln=0`, `thrust=100`, no `mtvc` → `FAPB[0]==-pdynmc*refa*ca+100` rtol 1e-12; with `mtvc=2` and `FPB=(1,2,3)` → `FAPB` includes `FPB` not raw thrust
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 forces FAPB/FMB`

---

### Task 15: SRAAM6 TVC

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/tvc.py`
- Test: `Python/tests/unit/test_sraam6_tvc.py`

**Interfaces:** Port `tvc.cpp` / `tvc_scnd`. `mtvc==0` return. `{1,2,3}` as spec. Else `ValueError`. Local CADAC `sign`. Include on the missile vehicle in Task 25 so `define` runs.

- [ ] **Step 1:** `mtvc=0` leaves `FPB` at define zeros; `mtvc=1`, `gtvc=1`, `dqcx=1`, `thrust=100` → `FPB` finite vs `sin/cos` formulas; `mtvc=4` raises
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 thrust vector control`

---

### Task 16: seeker download, kinematic, modes 0/2/3/4

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/seeker.py`
- Test: `Python/tests/unit/test_sraam6_seeker.py`

**Interfaces:**
- Consumes: `Packet`, `polar_from_cart`, `mat2tr`, `_skew` local
- Produces: `Sraam6Seeker`. `define` C++ `def_seeker`. `tgt_num` 1-based among `type=="TARGET3"`; copy `SAEL`→`STEL`, `VAEL`→`VTEL`. `sht_num` same among TARGET3 (`0` → no shooter). `mseek==0` geometry only. `==2` if `dbt<racq` → 3. `==3`/`==4`: `ms1dyn==0` uses `seeker_kin`; `ms1dyn==1` calls `seeker_dyn` (this task: method may set `ehy=ehz=0` and pointing from `seeker_kin` so FOV/`dtimac` still work; Task 17 replaces the body with C++ `seeker_dyn`). `timeac>dtimac` → 4. `==4` sets `mguid=6`. Else `ValueError`. `ms1dyn` not in `{0,1}` → `ValueError`. `TTL` identity. Helpers `seeker_kin`, `seeker_uthpb`, `seeker_thb` as C++ (uthpb C++ `&&` precedence).

- [ ] **Step 1:** Fake combus one `MISSILE6` + one `TARGET3` with 1v1 `SBEL=(0,0,-5000)`, `SAEL=(10000,500,-2000)`, `VAEL` from `dvae=250` heading 180. `tgt_num=1`, `mseek=2`, `racq=7000` → `mseek==3` (`dbt>7000` would stay 2 — use `racq=20000` to enter 3 in the same test, and a second test `racq=1` stays 2). `mseek=0` does not set 3. `mseek=1` raises. `seeker_kin` `sigdy`/`thtpb` finite. `tgt_num=2` with one target leaves `STEL` at zeros.
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 kinematic seeker and tgt_num`

---

### Task 17: seeker_dyn

**Files:** Modify `seeker.py`; Test: `Python/tests/unit/test_sraam6_seeker_dyn.py`

**Interfaces:** Port `seeker_dyn` / `seeker_aimp`. Kalman 2nd-order lags, gimbal, break-lock `trcond` 6–9 → `mseek=2`,`mguid=3`. `dbt<dblind` → `mseek=5`. `ms1dyn==1` path used in 1v1. Noise fields default 0.

- [ ] **Step 1:** `mseek=4`, `ms1dyn=1`, `dblind=3`, `dbt` via `SBTL` with `||SBTL||=100`, `dt=0.001`; `sigdpy`/`sigdpz` finite; `||SBTL||=1` (`<dblind`) → `mseek==5`; `ththb` beyond `trtht` → `mseek==2` and `trcond==6`
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 dynamic seeker`

---

### Task 18: guidance midcourse + mnav

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/guidance.py`
- Test: `Python/tests/unit/test_sraam6_guidance_mid.py`

**Interfaces:** Port `def_guidance`, `guidance` extrapolation, `guidance_mid`. `mnav==3` store `STEL`/`VTEL` then `mnav=0`; `mnav==0` keep; else `ValueError`. `mguid==0` skip laws. `mguid==3` mid. `mguid==6` calls `guidance_term` (this task: method `pass`; Task 19 fills C++ `guidance_term`). Else `ValueError`. 1v1: `gnav=3.75`, `mnav=3`.

- [ ] **Step 1:** `mguid=3`, `mnav=3`, 1v1 `SBEL`/`STEL`/`VTEL`/`VBEL`/`TBL=I`; `ancomx`/`alcomx` vs C++ `AAPNB` rtol 1e-12; `mnav` becomes 0; `mguid=0` does not require LOS; `mguid=1` raises; `mnav=1` raises
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 midcourse pronav`

---

### Task 19: guidance terminal

**Files:** Modify `guidance.py`; Test: `Python/tests/unit/test_sraam6_guidance_term.py`

**Interfaces:** Port `guidance_term` (closing speed, gravity bias, circular `gmax` limiter). `time>3` and `dcvel<trcvel` → `trcond=1`.

- [ ] **Step 1:** frozen `thtpb,psipb,sigdpy,sigdpz,FSPB,gmax,TBL,STEL,SBEL,VBEL,VTEL,gnav=3.75`; `alcomx`/`ancomx` vs C++ rtol 1e-12; `aa=sqrt(all**2+ann**2)` clipped to `gmax`
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 terminal pronav`

---

### Task 20: missile intercept

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/intercept.py`
- Test: `Python/tests/unit/test_sraam6_intercept.py`

**Interfaces:** Port `intercept.cpp` without `cout`/`exit`. `mterm` in `{0,1,2}` else `ValueError`. 1v1 `mterm=1`. C++ fires on `closing_speed>0 && write` where `closing_speed=UTBL·VTBEL` is LOS range-rate (positive = opening after CPA). Halt / ground / `trcond and stop` kill missile. Hit kills missile and TARGET3 at `tgt_com_slot`. `write` default 1.

- [ ] **Step 1:** `halt=1` → health 0 and combus status 0, no exception. `SBEL[2]=1` (alt≤0) → dead. `mterm=3` raises. Two-step CPA matching C++: `mseek=4`, `dbt<100`; first execute with range-rate `<=0` (still closing) leaves `write==1` and both alive; second execute with range-rate `>0` (opening after CPA) → both statuses 0 and `miss` finite
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 intercept without sys.exit`

---

### Task 21: Flat3 aircraft EOM (SAEL/dvae)

**Files:**
- Modify: `Python/src/cadac/eom/flat3.py`
- Test: `Python/tests/unit/test_sraam6_target_eom.py`

**Interfaces:**
- Consumes: existing `atmosphere76`, `gravity`, `mat2tr`, `polar_from_cart`, `integrate`, `Flat3Kinematics`
- Produces: `Flat3AircraftEnvironment` — US76 from `alt=-SAEL[2]`, Mach from `dvae` (not `dvbe`/`SBEL`). `Flat3AircraftNewton` — C++ `def_newton`/`init_newton`/`newton` names: `TAL`,`TAV`,`TVL`,`dvae`,`SAEL`,`VAEL`,`AAEL`,`psialx`,`thtalx`,`sael1`,`sael2`,`sael3`,`psial`,`thtal`. Reads `FSPA`,`phiavout`. Do not change `Flat3Environment`/`Flat3Newton` used by PLANE.

- [ ] **Step 1:** 1v1 Target ICs `sael1=10000`,`sael2=500`,`sael3=-2000`,`psialx=180`,`thtalx=0`,`dvae=250`; after init `SAEL==(10000,500,-2000)`, `dvae==250`, `alt=2000`. One step `FSPA=0`, `dt=0.001`, `phiavout=0` → `SAEL` changes, Mach matches `atmosphere76(2000)` and `dvae`. Existing `tests/unit/test_flat3_*.py` / plane5 newton tests still pass
- [ ] **Step 2–5:** implement, pass, commit `feat: Flat3 aircraft SAEL/dvae EOM`

---

### Task 22: SRAAM6 target guidance

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/target.py` (guidance class here or `target_guidance.py` — prefer methods/classes in `target.py` until the file needs a split; if split, `target_guidance.py`)
- Test: `Python/tests/unit/test_sraam6_target_guidance.py`

**Interfaces:** Port `Target::def_guidance` / `guidance`. `tgt_option` in `{0,1,2}` else `ValueError`. Option 1: 1v1 `gturn=1`. Option 2: 1-based MISSILE6 packet; `mseek%10==4` escape as C++ using `SBEL`/`VBEL`/`SAEL`/`VAEL` field names.

- [ ] **Step 1:** `tgt_option=1`, `gturn=1`, `grav=9.8`, `TVL=I` → `ACOML==(0,gturn*grav,-grav)` rtol 1e-12. `tgt_option=0` → `ACOML==(0,0,-grav)`. `tgt_option=3` raises. Option 2 with a missile packet `mseek=4` yields finite `ACOML`
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 target guidance g-turn`

---

### Task 23: SRAAM6 target control

**Files:** Modify `target.py` (or `target_control.py`); Test: `Python/tests/unit/test_sraam6_target_control.py`

**Interfaces:** Port `def_control` / `control`. Defaults `tphi=0`,`tanx=0`,`philimx=120`,`alplimx=40`,`clalpha=0.0523`,`wingloading=3247`. Writes `phiavout`. CADAC `sign`. `tgt_option>0` alpha limiter.

- [ ] **Step 1:** `tphi=tanx=0`, `tgt_option=1`, `ACOML` from Task 22, `TVL=I` → `phiavx`/`anx` finite; `|phiavx|<=120`. `tphi=0.1`, `dt=0.001` → bank lags command
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 target bank and load-factor control`

---

### Task 24: SRAAM6 target forces

**Files:** Modify `target.py` (or `target_forces.py`); Test: `Python/tests/unit/test_sraam6_target_forces.py`

**Interfaces:** Port `Target::forces`. `FSPA=(acc_longx*grav, 0, -anx*grav)`. `acc_longx` default 0.

- [ ] **Step 1:** `acc_longx=0`, `anx=1`, `grav=9.80675445` → `FSPA==(0,0,-grav)` rtol 1e-12
- [ ] **Step 2–5:** implement, pass, commit `feat: SRAAM6 target FSPA`

---

### Task 25: Vehicles, registry, 1v1 case, smoke

**Files:**
- Create: `Python/src/cadac/vehicles/sraam6/vehicle.py`; finish `target.py` `Sraam6Target`
- Modify: `Python/src/cadac/cli.py` — register family pairs; missile both decks; Target no decks
- Translate + decks → `Python/cases/sraam6/`
- Test: `Python/tests/unit/test_sraam6_one_step.py`
- Update Task 1: `family="sraam6"` + `TARGET3` now constructs `Sraam6Target`. Retarget no-fallthrough to unregistered `("sraam6","CRUISE3")` (not TARGET3). Do not add a no-family AIM5-raises test.

**Interfaces:**
- `Sraam6Missile.type=="MISSILE6"`. Constructor `(name, aero_deck, prop_deck, events=None)`. Modules in 1v1 MODULES order plus tvc for define: environment, kinematics, aerodynamics, propulsion, seeker, guidance, control, actuator, forces, euler, newton (`Flat6Newton`), intercept, tvc. Skip-if-exists on define collisions. After `define`, `com_names` = fields with `"com"` in `outputs` (includes `time`, `vmach`, `SBEL`, `VBEL`, `mseek`).
- `Sraam6Target.type=="TARGET3"`. Constructor `(name, events=None)`. Modules: `Flat3AircraftEnvironment`, `Flat3Kinematics`, `Flat3AircraftNewton`, target guidance, control, forces. `com_names` from `"com"` outputs (includes `dvae`, `SAEL`, `VAEL`, `psial`, `thtal`, …).
- `_VEHICLE_FAMILIES[("sraam6","MISSILE6")]=Sraam6Missile`; `[("sraam6","TARGET3")]=Sraam6Target`. Factory: `spec.family` set + TARGET3 pair → no decks; MISSILE6 → both decks required. `_build_vehicle(path, spec)`.
- Case: `translate_scenario_asc(..., family="sraam6")` + both `deck_asc_to_jsonc`. Each vehicle `"family": "sraam6"`. `end_time` 12. Smoke: copy JSONC, `end_time=0.1`; missile `hbe` near 5000; **target `SAEL` finite** (read from the Target vehicle store after `run_scenario` / `run_loop`, not from plot slot 0); **both `health==1`**.

- [ ] **Step 1:**

```python
import json
import math
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.io.jsonc import loads
from cadac.io.translate import deck_asc_to_jsonc, translate_scenario_asc
from cadac.vehicles.sraam6.target import Sraam6Target
from cadac.vehicles.sraam6.vehicle import Sraam6Missile

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "CADAC_Simulations/SRAAM6_250130/SRAAM6"
CASES = Path(__file__).resolve().parents[2] / "cases" / "sraam6"


def test_missile_module_order_includes_tvc():
    vehicle = Sraam6Missile("Missile", None, None)
    assert vehicle.type == "MISSILE6"
    names = [m.name for m in vehicle.modules]
    assert names[:4] == ["environment", "kinematics", "aerodynamics", "propulsion"]
    assert "tvc" in names
    assert "newton" in names


def test_target_constructor_has_no_decks():
    vehicle = Sraam6Target("Target aircraft")
    assert vehicle.type == "TARGET3"


def test_com_names_from_com_outputs():
    missile = Sraam6Missile("Missile", None, None)
    missile.define()
    for name in ("time", "vmach", "SBEL", "VBEL", "mseek"):
        assert name in missile.com_names
        assert "com" in missile.store.field(name).outputs
    target = Sraam6Target("Target aircraft")
    target.define()
    for name in ("dvae", "SAEL", "VAEL", "psial", "thtal"):
        assert name in target.com_names
        assert "com" in target.store.field(name).outputs


def test_1v1_tenth_second_hbe(tmp_path: Path):
    translate_scenario_asc(SRC / "input_1v1.asc", tmp_path, family="sraam6")
    deck_asc_to_jsonc(SRC / "sraam6_aero_deck.asc", tmp_path / "sraam6_aero_deck.jsonc")
    deck_asc_to_jsonc(SRC / "sraam6_prop_deck.asc", tmp_path / "sraam6_prop_deck.jsonc")
    path = tmp_path / "input_1v1.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    result = run_scenario(path)
    row = next(r for r in result.plot_rows if r["time"] >= 0.1)
    assert math.isfinite(row["hbe"])
    assert 4500.0 < row["hbe"] < 5500.0


def test_1v1_tenth_second_target_sael_and_health(tmp_path: Path):
    from cadac.cli import _build_vehicle
    from cadac.io.scenario import load_scenario
    from cadac.kernel.executive import SimContext, run_loop

    translate_scenario_asc(SRC / "input_1v1.asc", tmp_path, family="sraam6")
    deck_asc_to_jsonc(SRC / "sraam6_aero_deck.asc", tmp_path / "sraam6_aero_deck.jsonc")
    deck_asc_to_jsonc(SRC / "sraam6_prop_deck.asc", tmp_path / "sraam6_prop_deck.jsonc")
    path = tmp_path / "input_1v1.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    cfg = load_scenario(path)
    vehicles = []
    for spec in cfg.vehicles:
        vehicle = _build_vehicle(path, spec)
        vehicle.define()
        for name, value in spec.params.items():
            vehicle.store.set(name, value)
        ctx = SimContext(0.0, 0.001, 0.0, 0.0, None, len(vehicles))
        for module in vehicle.modules:
            module.initialize(vehicle, ctx)
        vehicles.append(vehicle)
    run_loop(
        vehicles,
        {v: v.modules for v in vehicles},
        [m["name"] for m in data["modules"]],
        0.1,
        0.001,
    )
    assert vehicles[0].health == 1
    assert vehicles[1].health == 1
    sael = vehicles[1].store.get("SAEL")
    assert all(math.isfinite(float(x)) for x in sael)


def test_family_sraam6_cruise3_does_not_fall_through(tmp_path: Path):
    path = tmp_path / "c.jsonc"
    path.write_text(
        json.dumps(
            {
                "title": "nf",
                "options": {},
                "modules": [],
                "timing": {"int_step": 0.1},
                "end_time": 0.0,
                "vehicles": [
                    {
                        "family": "sraam6",
                        "type": "CRUISE3",
                        "name": "c",
                        "params": {},
                        "events": [],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CRUISE3"):
        run_scenario(path)
```

`run_scenario` on a family-sraam6 TARGET3-only tiny JSONC (no decks) does not raise missing deck; MISSILE6 without decks raises. Translated JSONC: each vehicle `"family": "sraam6"`. Retarget Task 1 `test_family_sraam6_does_not_fall_through_to_global_target3` to the CRUISE3 test above (TARGET3 now resolves).

- [ ] **Step 2–5:** wire, copy case files into `Python/cases/sraam6/`, pass, commit `feat: run SRAAM6 1v1 from JSONC`

---

### Task 26: SRAAM6 e2e golden (optional file)

**Files:** `Python/tests/e2e/test_sraam6_1v1.py`

**Interfaces:** Skip if `tests/e2e/goldens/sraam6/plot.csv` absent. Else `run_scenario` on `Python/cases/sraam6/input_1v1.jsonc`. Compare missile plot columns present in both (`hbe`/`vmach` if both present). Sentinel `time=-1`. Pattern `test_falcon6_gamma.py`. CSV rtol=1e-5, atol=max(1e-6, 5e-6*|g|).

- [ ] **Step 1:** skip helper without golden; with a tiny fixture CSV, `hbe` at t=0; full golden test skipped until the file exists
- [ ] **Step 2–5:** implement, pass, commit `test: SRAAM6 1v1 e2e gate (skip without golden)`

---

## Self-review

**Spec coverage:** AIM5 family API (Task 1–2, 25). Missile env/kin (`phip` EPS+SMALL)/euler/aero/der/prop/actuator/control/forces/tvc/seeker/guidance/intercept CPA `closing_speed>0` (Tasks 3–20). Flat6Newton reused in Task 25. Target EOM/guidance/control/forces (21–24). 1v1 case + smoke `SAEL`+both health (25). E2E skip (26). Unused modes ValueError on each dispatcher task. No sys.exit (20). GAUSS means stored (2). Packets by type+names and tgt_num (16, 22, 20). HYPER5 TARGET3 global preserved (1). No-fallthrough after registry is `("sraam6","CRUISE3")` (25).

**Placeholder scan:** no TBD/TODO; each task has files, interfaces, and a failing-test step.

**Type consistency:** `Sraam6Missile` / `Sraam6Target`; `_VEHICLE_FAMILIES` tuples; `VehicleSpec.family` (not `RunConfig.family`); `_build_vehicle(path, spec)`; Target constructor `(name, events=None)`; Missile both decks; store names `SAEL`/`dvae`/`FSPA` vs missile `SBEL`/`dvbe`/`FAPB`.
