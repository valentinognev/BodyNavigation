# CADAC CRUISE5 (CRUISE3 / TARGET3 / SATELLITE3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run CRUISE5 `input_1.asc` from JSONC (family `cruise5`: UAV `CRUISE3` + Tank `TARGET3` + Sat `SATELLITE3`); unit-test each C++ module used by that case; e2e vs CADAC CSV when a golden exists.

**Architecture:** Reuse Round3 (ISO 62) and `cadine`. Port CRUISE5 `cruise_modules.cpp` / `target_modules.cpp` / `satellite_modules.cpp`. Bind colliding CADAC type tokens via `_VEHICLE_FAMILIES[(family, type)]` only — never overwrite HYPER3 `CRUISE3` or HYPER5 `TARGET3`/`SATELLITE3`. Do not import `Cruise3`, `Hyper5`, `Target3`, or `Satellite3`.

**Tech Stack:** Python >= 3.11, numpy, pytest. Work in `Python/`.

**Spec:** `docs/superpowers/specs/2026-09-06-cadac-cruise5-design.md`

**Depends on:** kernel + Round3 + HYPER5 (`cadine`, skip-missing-modules, plot slot 0).

## Global Constraints

- Spec + parent `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
- Do not regress HYPER3 e2e or HYPER5/HYPER6/FALCON5/FALCON6 unit tests
- Port `CADAC_Simulations/CRUISE5_250115/CRUISE5/` numerics; do not simplify autopilots
- Family `cruise5` map only: `("cruise5","CRUISE3")=Cruise5`, `("cruise5","TARGET3")=Cruise5Target`, `("cruise5","SATELLITE3")=Cruise5Satellite`
- Canonical family API (AIM5): `VehicleSpec.family` is source of truth; scenario-level `"family"` defaults vehicles that omit it; vehicle key wins; family set → only `_VEHICLE_FAMILIES`; none → only `_VEHICLE_TYPES`. `ValueError` includes type and family when set
- Do not reassign `_VEHICLE_FAMILIES = {}` (would wipe AIM5). Register cruise5 pairs only in Task 17
- Seeker/targeting identify packets by `Packet.type` and field names, not CADAC `id.find("t")`
- E2E skip if `Python/tests/e2e/goldens/cruise5/plot.csv` absent (do not create it)
- Tests named `test_cruise5_*.py` (not `test_cruise3_*`)
- Do not retarget unknown-type tests (AIM5 plan owns AIM5→ROTOR)
- Grok implementer `cursor-grok-4.6-high` + Grok reviewer; TDD; no Composer Fast / Kimi / Fast
- Protocol: `define(vehicle)` / `execute(vehicle, ctx)` on `vehicle.store`
- Unit `rtol=1e-12`, `atol=1e-14`; CSV e2e `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`
- No `sys.exit`. Unused modes `ValueError`

## File map

- `Python/src/cadac/io/scenario.py` — `VehicleSpec.family: str | None = None`
- `Python/src/cadac/io/translate.py` — `translate_scenario_asc(src, dst_dir, family=None)`
- `Python/src/cadac/cli.py` — `_VEHICLE_FAMILIES`, resolve-by-family, plot columns by class
- `Python/src/cadac/vehicles/round3/cruise5/{aero,propulsion,forces,control,guidance,seeker,intercept,targeting,target,satellite,vehicle}.py`
- Tests `Python/tests/unit/test_cruise5_*.py`, `Python/tests/e2e/test_cruise5_input1.py`
- Case `Python/cases/cruise5/` from `input_1.asc` + `cruise3_aero_deck.asc` + `cruise3_prop_deck.asc`
- C++: `cruise_modules.cpp`, `target_modules.cpp`, `satellite_modules.cpp`, `utility_functions.cpp` (`cadine`, `cadtbv`, `sign`, `angle`)

## input_1.asc ICs (later tasks)

UAV `CRUISE3`: `lonx=14.7`, `latx=35.4`, `alt=7000`, `psivgx=90`, `thtvgx=0`, `dvbe=200`, `alphax=0`, `phimvx=0`, `mprop=4`, `mach_com=0.7`, `mass_init=1000`, `fuel_init=150`, `gfthm=893620`, `tfth=1`, `mseeker=0`, `acq_range=10000`, `mguidance=30`, `pronav_gain=3`, `line_gain=1`, `nl_gain_fact=0.6`, `decrement=1000`, `mcontrol=46`, `anposlimx=3`, `anneglimx=-1`, `gacp=10`, `ta=0.8`, `alpposlimx=15`, `alpneglimx=-10`, `gcp=2`, `allimx=1`, `philimx=70`, `tphi=0.5`, `altcom=7000`, `altdlim=50`, `gh=0.3`, `gv=1.0`, `psivgcx=90`, `gain_psivg=12`, `gain_thtvg=30`, `wp_lonx=14.9`, `wp_latx=35.4`, `psifgx=90`. Default `area=0.929`. `int_step=0.05`.

Tank `TARGET3` Tank_t1: `lonx=15.4`, `latx=35.3`, `alt=100`, `psivgx=45`, `dvbe=10`.

Sat `SATELLITE3` Sat_s1: `lonx=10`, `latx=30`, `alt=500000`, `psivgx=45`, `thtvgx=0`, `dvbe=7700`.

---

### Task 1: family dispatch (idempotent)

**Files:**
- Modify: `Python/src/cadac/io/scenario.py`, `Python/src/cadac/io/translate.py`, `Python/src/cadac/cli.py`
- Test: `Python/tests/unit/test_cruise5_family.py`

**Interfaces:**
- Consumes: existing `VehicleSpec`, `load_scenario`, `translate_scenario_asc(src, dst_dir)`, `_VEHICLE_TYPES`, `_plot_columns`
- Produces (idempotent — **if AIM5 already added this API, keep it**; do **not** write `_VEHICLE_FAMILIES = {}`):
  - `VehicleSpec.family: str | None = None` is the source of truth
  - `load_scenario`: resolved family = vehicle `"family"` if present else scenario-level `"family"` else `None`; vehicle key wins. `RunConfig` has no family field
  - `translate_scenario_asc(src, dst_dir, family=None) -> None` — when `family` is a str, **each** vehicle dict gets `"family": family`; when `None`, omit the key
  - `_VEHICLE_FAMILIES`: create the dict **only if the name is absent**. Never replace an existing map
  - Resolve: if `spec.family` is set: **only** `_VEHICLE_FAMILIES.get((spec.family, spec.type))`; missing → `ValueError` whose message includes **type and family**. If `spec.family` is `None`: **only** `_VEHICLE_TYPES`. No fallthrough either way
  - `_plot_columns(vehicle)`: `if type(vehicle) is Cruise3: return list(PLOT_COLUMNS)` else `flagged_plot_columns(vehicle.store)` (add only if missing)
  - Do not overwrite `_VEHICLE_TYPES["CRUISE3"]` / `TARGET3` / `SATELLITE3`. Do not register cruise5 pairs (Task 17)

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.cli import _plot_columns
from cadac.io.plot import PLOT_COLUMNS, flagged_plot_columns
from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc
from cadac.kernel.state import Field, StateStore

ROOT = Path(__file__).resolve().parents[3]
HYPER3 = ROOT / "CADAC_Simulations/HYPER3_250114/HYPER3"
CRUISE5 = ROOT / "CADAC_Simulations/CRUISE5_250115/CRUISE5"


def test_load_scenario_family_defaults_none(tmp_path):
    src = tmp_path / "n.jsonc"
    src.write_text(
        '{ "title": "t", "options": {}, "modules": [], "timing": {"int_step": 0.05}, '
        '"end_time": 1, "vehicles": [ { "type": "CRUISE3", "name": "v", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    cfg = load_scenario(src)
    assert cfg.vehicles[0].family is None
    assert cfg.vehicles[0].type == "CRUISE3"


def test_load_scenario_reads_family_key(tmp_path):
    src = tmp_path / "f.jsonc"
    src.write_text(
        '{ "title": "t", "options": {}, "modules": [], "timing": {"int_step": 0.05}, '
        '"end_time": 1, "vehicles": [ { "family": "cruise5", "type": "CRUISE3", '
        '"name": "UAV", "params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    cfg = load_scenario(src)
    assert cfg.vehicles[0].family == "cruise5"
    assert cfg.vehicles[0].type == "CRUISE3"
    assert cfg.vehicles[0].name == "UAV"


def test_scenario_family_applies_when_vehicle_omits_it(tmp_path):
    src = tmp_path / "s.jsonc"
    src.write_text(
        '{ "title": "t", "family": "cruise5", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.05}, "end_time": 1, '
        '"vehicles": [ { "type": "CRUISE3", "name": "UAV", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    cfg = load_scenario(src)
    assert cfg.vehicles[0].family == "cruise5"


def test_vehicle_family_overrides_scenario_family(tmp_path):
    src = tmp_path / "o.jsonc"
    src.write_text(
        '{ "title": "t", "family": "aim5", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.05}, "end_time": 1, '
        '"vehicles": [ { "family": "cruise5", "type": "CRUISE3", "name": "UAV", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    cfg = load_scenario(src)
    assert cfg.vehicles[0].family == "cruise5"


def test_translate_without_family_omits_key(tmp_path):
    translate_scenario_asc(HYPER3 / "input_climb.asc", tmp_path)
    raw = json.loads((tmp_path / "input_climb.jsonc").read_text(encoding="utf-8"))
    assert "family" not in raw["vehicles"][0]
    cfg = load_scenario(tmp_path / "input_climb.jsonc")
    assert cfg.vehicles[0].family is None


def test_translate_family_cruise5_writes_on_all_vehicles(tmp_path):
    translate_scenario_asc(CRUISE5 / "input_1.asc", tmp_path, family="cruise5")
    raw = json.loads((tmp_path / "input_1.jsonc").read_text(encoding="utf-8"))
    assert len(raw["vehicles"]) == 3
    for vehicle in raw["vehicles"]:
        assert vehicle["family"] == "cruise5"
    types = [v["type"] for v in raw["vehicles"]]
    assert types == ["CRUISE3", "TARGET3", "SATELLITE3"]
    cfg = load_scenario(tmp_path / "input_1.jsonc")
    assert [v.family for v in cfg.vehicles] == ["cruise5", "cruise5", "cruise5"]
    assert cfg.vehicles[0].params["mprop"] == 4
    assert cfg.vehicles[0].params["mcontrol"] == 46
    assert cfg.vehicles[0].params["mguidance"] == 30
    assert cfg.end_time == 410


def test_family_set_does_not_fall_through_to_global_cruise3(tmp_path):
    path = tmp_path / "uav.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], "timing": {"int_step": 0.05}, '
        '"end_time": 0, "vehicles": [ { "family": "cruise5", "type": "CRUISE3", '
        '"name": "UAV", "aero_deck": "a.jsonc", "prop_deck": "p.jsonc", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / "a.jsonc").write_text(
        '{ "title": "a", "tables": [] }\n', encoding="utf-8", newline="\n"
    )
    (tmp_path / "p.jsonc").write_text(
        '{ "title": "p", "tables": [] }\n', encoding="utf-8", newline="\n"
    )
    with pytest.raises(ValueError, match="CRUISE3") as excinfo:
        run_scenario(path)
    assert "cruise5" in str(excinfo.value)


def test_family_none_hyper5_target3_still_works(tmp_path):
    path = tmp_path / "t.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [ '
        '{ "name": "environment", "phases": ["def", "init", "exec"] }, '
        '{ "name": "newton", "phases": ["def", "init", "exec"] }, '
        '{ "name": "forces", "phases": ["def", "exec"] }, '
        '{ "name": "intercept", "phases": ["def", "exec"] } ], '
        '"timing": {"int_step": 0.05}, "end_time": 0, '
        '"vehicles": [ { "type": "TARGET3", "name": "Truck_t1", '
        '"params": { "lonx": 0, "latx": 0, "alt": 100, "dvbe": 1 }, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    result = run_scenario(path)
    assert result.plot_rows is not None


def test_plot_columns_non_cruise3_class_with_type_cruise3_uses_flagged():
    class Other:
        type = "CRUISE3"

        def __init__(self):
            self.store = StateStore()
            self.store.define(
                Field("alt", 1.0, "real", "out", "newton", ("plot",))
            )

    other = Other()
    cols = _plot_columns(other)
    assert cols == flagged_plot_columns(other.store)
    assert cols != list(PLOT_COLUMNS)
```

- [ ] **Step 2:** `cd Python && python -m pytest tests/unit/test_cruise5_family.py -v` — FAIL if API missing (`family` / no-fallthrough / plot-by-class). If AIM5 already shipped the API, these tests should already be close to green except plot-by-class and cruise5 translate
- [ ] **Step 3:** If missing: add `VehicleSpec.family`; `_vehicle(..., scenario_family=None)` with `family=raw["family"] if "family" in raw else scenario_family`; `load_scenario` passes `data.get("family")`; `translate_scenario_asc(..., family=None)` stamps each vehicle; create `_VEHICLE_FAMILIES` **only if absent**; resolve family-first with `ValueError` including type and family when set; `_plot_columns` by `type(vehicle) is Cruise3`. Never `_VEHICLE_FAMILIES = {}` over an existing map
- [ ] **Step 4:** PASS `test_cruise5_family.py` plus `tests/unit/test_scenario.py`. If `test_vehicle_family.py` exists (AIM5), it must still pass. Do not retarget unknown-type tests
- [ ] **Step 5: Commit** `feat: VehicleSpec family dispatch without overwriting CRUISE3`

---

### Task 2: Cruise5 aerodynamics

**Files:**
- Create: `Python/src/cadac/vehicles/round3/cruise5/__init__.py`, `Python/src/cadac/vehicles/round3/cruise5/aero.py`
- Test: `Python/tests/unit/test_cruise5_aero.py`

**Interfaces:**
- Consumes: `Datadeck.look_up` 1D; `parse_asc_deck` of `CADAC_Simulations/CRUISE5_250115/CRUISE5/cruise3_aero_deck.asc`
- Produces: `Cruise5Aero(deck).name=="aerodynamics"`. `define` registers C++ `def_aerodynamics`: `cl` out, `cd` out, `cl_ov_cd` diag `scrn,plot`, `area` data default **0.929**, `cla` out. Does **not** define `alphax` or `mach`. `execute` ports `Cruise::aerodynamics`: `cd0=look_up("cd0_vs_mach",mach)`, `cl0=look_up("cl0_vs_mach",mach)`, `cla=look_up("cla_vs_mach",mach)`, `ckk=look_up("ckk_vs_mach",mach)`, `cla0=look_up("cla0_vs_mach",mach)`; `cl=cla0+cla*alphax`; `cd=cd0+ckk*(cl-cl0)**2`; `cl_ov_cd=cl/cd`. Do not import `Cruise3Aero`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest

from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.round3.cruise5.aero import Cruise5Aero

CRUISE5 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/CRUISE5_250115/CRUISE5"
RTOL = 1e-12
ATOL = 1e-14
MACH = 0.7
ALPHAX = 0.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _deck():
    _, tables = parse_asc_deck(CRUISE5 / "cruise3_aero_deck.asc")
    return Datadeck.from_tables(tables)


def test_name_is_aerodynamics():
    assert Cruise5Aero(_deck()).name == "aerodynamics"


def test_define_area_default_0_929_and_no_alphax():
    vehicle = _Vehicle()
    Cruise5Aero(_deck()).define(vehicle)
    assert vehicle.store.get("area") == pytest.approx(0.929, rel=RTOL, abs=ATOL)
    assert "alphax" not in vehicle.store.names()
    field = vehicle.store.field("cl_ov_cd")
    assert field.module == "aerodynamics"
    assert field.outputs == ("scrn", "plot")


def test_execute_drag_polar_vs_lookup_formulas():
    deck = _deck()
    vehicle = _Vehicle()
    aero = Cruise5Aero(deck)
    aero.define(vehicle)
    vehicle.store.define(Field("mach", MACH, "real", "out", "environment"))
    vehicle.store.define(Field("alphax", ALPHAX, "real", "out", "control"))
    cd0 = deck.look_up("cd0_vs_mach", MACH)
    cl0 = deck.look_up("cl0_vs_mach", MACH)
    cla = deck.look_up("cla_vs_mach", MACH)
    ckk = deck.look_up("ckk_vs_mach", MACH)
    cla0 = deck.look_up("cla0_vs_mach", MACH)
    cl = cla0 + cla * ALPHAX
    cd = cd0 + ckk * (cl - cl0) ** 2
    aero.execute(vehicle, None)
    store = vehicle.store
    assert store.get("cl") == pytest.approx(cl, rel=RTOL, abs=ATOL)
    assert store.get("cd") == pytest.approx(cd, rel=RTOL, abs=ATOL)
    assert store.get("cla") == pytest.approx(cla, rel=RTOL, abs=ATOL)
    assert store.get("cl_ov_cd") == pytest.approx(cl / cd, rel=RTOL, abs=ATOL)
```

- [ ] **Step 2:** `cd Python && python -m pytest tests/unit/test_cruise5_aero.py -v` — FAIL (`Cruise5Aero` not defined)
- [ ] **Step 3:** Implement `Cruise5Aero` from C++ `Cruise::aerodynamics` (not HYPER3 aero)
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: CRUISE5 UAV drag-polar aerodynamics`

---

### Task 3: Cruise5 propulsion

**Files:**
- Create: `Python/src/cadac/vehicles/round3/cruise5/propulsion.py`
- Test: `Python/tests/unit/test_cruise5_propulsion.py`

**Interfaces:**
- Consumes: `integrate`, `Datadeck`, `RAD` from constants. Constructor takes Datadeck (UAV always has a prop deck).
- Produces: `Cruise5Propulsion`. `define` C++ `def_propulsion` names: `mprop` int data/diag `scrn,plot` default 0; `cg` diag plot; `fidle` diag; `thrust_com` data; `thrust` out `scrn,plot,com`; `treqd`/`treq` state; `fmassed`/`fmasse` state; `fuelmass` save; `mach_com`/`gfthm`/`tfth` data; `mass` out scrn; `tav` diag; `mass_init`/`fuel_init` data. `initialize`: `mass=mass_init`. `execute` ports `Cruise::propulsion`: `mprop==0` thrust=0, `cg=look_up("cg_vs_mass",mass)`, write thrust, **return** (no fuel integrate). Else look up `fidle_vs_alt_mach`, `tav_vs_alt_mach`. `1` thrust=`thrust_com`, `ff=look_up("ff_vs_thrust_alt_mach",thrust,alt,mach)`, `treq=thrust_com`. `2` thrust=`fidle`, `ff=look_up("iff_vs_alt",alt)`. `3` thrust=`tav`, `ff=look_up("ff_vs_thrust_alt_mach",...)`. `mprop>3`: set `mprop=4`; `treqs=cd*pdynmc*area`; `epsmch=mach_com-mach`; `tcom=epsmch*gfthm+treqs`; `treqd_new=(tcom-2*treq)/tfth`; `treq=integrate(treqd_new,treqd,treq,int_step)`; `treqb=treq/cos(alphax*RAD)`; if `treqb<fidle`: `mprop=5`, `treqb=fidle`; if `treqb>tav`: `mprop=6`, `treqb=tav`; `thrust=treqb`; `ff=look_up("ff_vs_thrust_alt_mach",...)`. Then `fmasse=integrate(ff,fmassed,fmasse,int_step)`; `mass=mass_init-fmasse`; `fuelmass=fuel_init-fmasse`; `cg=look_up("cg_vs_mass",mass)`; if `fuelmass<=0`: `thrust=0`. `mprop<0` → `ValueError`.

- [ ] **Step 1: Write the failing test**

Use `input_1` ICs: `mass_init=1000`, `fuel_init=150`, `mach_com=0.7`, `gfthm=893620`, `tfth=1`, `alphax=0`, `area=0.929`, `int_step=0.05`. Parse `cruise3_prop_deck.asc`. Fixture store also needs `pdynmc`, `mach`, `alt=7000`, `cd` (from aero formula at mach 0.7 alphax 0), `mass` after initialize = 1000.

```python
from math import cos
from pathlib import Path

import pytest

from cadac.constants import RAD
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.round3.cruise5.propulsion import Cruise5Propulsion

CRUISE5 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/CRUISE5_250115/CRUISE5"
RTOL = 1e-12
ATOL = 1e-14
INT_STEP = 0.05
MASS_INIT = 1000.0
FUEL_INIT = 150.0
ALT = 7000.0
MACH = 0.7
ALPHAX = 0.0
AREA = 0.929
MACH_COM = 0.7
GFTHM = 893620.0
TFTH = 1.0


def _deck():
    _, tables = parse_asc_deck(CRUISE5 / "cruise3_prop_deck.asc")
    return Datadeck.from_tables(tables)


def _ctx():
    return SimContext(
        sim_time=0.0, int_step=INT_STEP, event_time=0.0,
        out_fact=0.0, combus=None, vehicle_slot=0,
    )


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ready(mprop, cd=0.05, pdynmc=5000.0, thrust_com=1000.0):
    vehicle = _Vehicle()
    prop = Cruise5Propulsion(_deck())
    prop.define(vehicle)
    store = vehicle.store
    for name, value, ftype, role, module in (
        ("pdynmc", pdynmc, "real", "out", "environment"),
        ("mach", MACH, "real", "out", "environment"),
        ("alt", ALT, "real", "init/out", "newton"),
        ("cd", cd, "real", "out", "aerodynamics"),
        ("area", AREA, "real", "data", "aerodynamics"),
        ("alphax", ALPHAX, "real", "out", "control"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, role, module))
        else:
            store.set(name, value)
    store.set("mprop", mprop)
    store.set("mass_init", MASS_INIT)
    store.set("fuel_init", FUEL_INIT)
    store.set("mach_com", MACH_COM)
    store.set("gfthm", GFTHM)
    store.set("tfth", TFTH)
    store.set("thrust_com", thrust_com)
    prop.initialize(vehicle, _ctx())
    return vehicle, prop


def test_name_is_propulsion():
    assert Cruise5Propulsion(_deck()).name == "propulsion"


def test_mprop_0_zero_thrust_no_fuel_burn():
    vehicle, prop = _ready(0)
    store = vehicle.store
    prop.execute(vehicle, _ctx())
    assert store.get("thrust") == 0.0
    assert store.get("mass") == pytest.approx(MASS_INIT, rel=RTOL, abs=ATOL)
    assert store.get("fmasse") == 0.0


def test_mprop_1_commanded_thrust_and_fuel_table():
    vehicle, prop = _ready(1, thrust_com=1000.0)
    deck = _deck()
    ff = deck.look_up("ff_vs_thrust_alt_mach", 1000.0, ALT, MACH)
    fmasse = integrate(ff, 0.0, 0.0, INT_STEP)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("thrust") == pytest.approx(1000.0, rel=RTOL, abs=ATOL)
    assert store.get("fmasse") == pytest.approx(fmasse, rel=RTOL, abs=ATOL)
    assert store.get("mass") == pytest.approx(MASS_INIT - fmasse, rel=RTOL, abs=ATOL)


def test_mprop_2_idle_iff_vs_alt():
    vehicle, prop = _ready(2)
    deck = _deck()
    fidle = deck.look_up("fidle_vs_alt_mach", ALT, MACH)
    ff = deck.look_up("iff_vs_alt", ALT)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("thrust") == pytest.approx(fidle, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("fidle") == pytest.approx(fidle, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("fmasse") == pytest.approx(
        integrate(ff, 0.0, 0.0, INT_STEP), rel=RTOL, abs=ATOL
    )


def test_mprop_3_max_tav():
    vehicle, prop = _ready(3)
    deck = _deck()
    tav = deck.look_up("tav_vs_alt_mach", ALT, MACH)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("thrust") == pytest.approx(tav, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("tav") == pytest.approx(tav, rel=RTOL, abs=ATOL)


def test_mprop_4_mach_hold_one_step():
    cd = 0.05
    pdynmc = 5000.0
    vehicle, prop = _ready(4, cd=cd, pdynmc=pdynmc)
    deck = _deck()
    fidle = deck.look_up("fidle_vs_alt_mach", ALT, MACH)
    tav = deck.look_up("tav_vs_alt_mach", ALT, MACH)
    treqs = cd * pdynmc * AREA
    tcom = (MACH_COM - MACH) * GFTHM + treqs
    treqd_new = (tcom - 2.0 * 0.0) / TFTH
    treq = integrate(treqd_new, 0.0, 0.0, INT_STEP)
    treqb = treq / cos(ALPHAX * RAD)
    mprop = 4
    if treqb < fidle:
        mprop = 5
        treqb = fidle
    if treqb > tav:
        mprop = 6
        treqb = tav
    ff = deck.look_up("ff_vs_thrust_alt_mach", treqb, ALT, MACH)
    fmasse = integrate(ff, 0.0, 0.0, INT_STEP)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("thrust") == pytest.approx(treqb, rel=RTOL, abs=ATOL)
    assert store.get("mprop") == mprop
    assert store.get("fmasse") == pytest.approx(fmasse, rel=RTOL, abs=ATOL)


def test_mprop_negative_raises():
    vehicle, prop = _ready(-1)
    with pytest.raises(ValueError, match="mprop"):
        prop.execute(vehicle, _ctx())
```

- [ ] **Step 2:** FAIL
- [ ] **Step 3:** Port C++ `Cruise::propulsion` including `mprop==0` early return
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: CRUISE5 turbojet propulsion mprop 0-4`

---

### Task 4: Cruise5 forces

**Files:**
- Create: `Python/src/cadac/vehicles/round3/cruise5/forces.py`
- Test: `Python/tests/unit/test_cruise5_forces.py`

**Interfaces:**
- Produces: `Cruise5Forces.name=="forces"`. `define` `FSPV` vec out plot only; skip-if-name-exists. `execute` C++ `Cruise::forces`: `fspv1=(-pdynmc*area*cd+thrust*cos(alpha))/mass`, `fspv2=sin(phimv)*(pdynmc*area*cl+thrust*sin(alpha))/mass`, `fspv3=-cos(phimv)*(pdynmc*area*cl+thrust*sin(alpha))/mass` with `alpha=alphax*RAD`, `phimv=phimvx*RAD`.

- [ ] **Step 1: Write the failing test**

```python
from math import cos, sin

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.cruise5.forces import Cruise5Forces

RTOL = 1e-12
ATOL = 1e-14
PDYNMC = 5000.0
AREA = 0.929
CD = 0.05
CL = 0.2
THRUST = 1500.0
MASS = 1000.0
ALPHAX = 0.0
PHIMVX = 0.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ready(phimvx=PHIMVX):
    vehicle = _Vehicle()
    forces = Cruise5Forces()
    forces.define(vehicle)
    store = vehicle.store
    for name, value in (
        ("pdynmc", PDYNMC), ("area", AREA), ("cd", CD), ("cl", CL),
        ("thrust", THRUST), ("mass", MASS), ("alphax", ALPHAX),
        ("phimvx", phimvx),
    ):
        if name not in store.names():
            store.define(Field(name, value, "real", "data", "test"))
        else:
            store.set(name, value)
    return vehicle, forces


def test_name_is_forces():
    assert Cruise5Forces().name == "forces"


def test_fspv_phimvx_0_and_90_match_cpp():
    alpha = ALPHAX * RAD
    for phimvx in (0.0, 90.0):
        phimv = phimvx * RAD
        fspv1 = (-PDYNMC * AREA * CD + THRUST * cos(alpha)) / MASS
        fspv2 = sin(phimv) * (PDYNMC * AREA * CL + THRUST * sin(alpha)) / MASS
        fspv3 = -cos(phimv) * (PDYNMC * AREA * CL + THRUST * sin(alpha)) / MASS
        vehicle, forces = _ready(phimvx)
        forces.execute(vehicle, None)
        np.testing.assert_allclose(
            vehicle.store.get("FSPV"), [fspv1, fspv2, fspv3], rtol=RTOL, atol=ATOL
        )


def test_define_skips_existing_fspv():
    vehicle = _Vehicle()
    vehicle.store.define(Field("FSPV", (1.0, 2.0, 3.0), "vec", "out", "newton"))
    Cruise5Forces().define(vehicle)
    np.testing.assert_allclose(vehicle.store.get("FSPV"), [1.0, 2.0, 3.0])
```

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 UAV forces FSPV`

---

### Task 5: control_bank

**Files:**
- Create: `Python/src/cadac/vehicles/round3/cruise5/control.py`
- Test: `Python/tests/unit/test_cruise5_control_bank.py`

**Interfaces:**
- Produces: `Cruise5Control.name=="control"`. `define` C++ `def_control` fields used by bank and later helpers (full `def_control` list: `mcontrol`, `psivgcx`, `thtvgcx`, `alphacx`, `phimvcx`, `TBV`, `TBG`, `gain_thtvg`, `gain_psivg`, `anx`, `avx`, `alphax`, `phimvx`, `phicx`, `phix`, `phixd`, `philimx`, `tphi`, `anposlimx`, `anneglimx`, `gacp`, `ta`, `xi`, `xid`, `qq`, `tip`, `alp`, `alpd`, `alpposlimx`, `alpneglimx`, `ancomx`, `alcomx`, `allimx`, `gcp`, `alx`, `altdlim`, `gh`, `gv`, `altd`, `altcom`) with C++ roles/outputs. `control_bank(vehicle, phicx, int_step) -> float` ports `Cruise::control_bank`: clip `phicx` to `±philimx`; `phixd_new=(phicx-phix)/tphi`; `phix=integrate(...)`; return `phix`. Does not write `phimvx`. `execute` is pass until Task 8. `input_1`: `philimx=70`, `tphi=0.5`, `int_step=0.05`.

- [ ] **Step 1:** one- and two-step lag vs C++ replica; limiter at `±philimx`; `tphi=0.5` so first step `phicx=30` → `phix=(30-0)/0.5 * 0.05 / 2 = 1.5`

```python
import pytest
from cadac.kernel.state import StateStore
from cadac.vehicles.round3.cruise5.control import Cruise5Control

PHILIMX = 70.0
TPHI = 0.5
INT_STEP = 0.05
RTOL = 1e-12
ATOL = 1e-14


def _expected_bank(phicx, phix, phixd, philimx, tphi, int_step):
    if phicx > philimx:
        phicx = philimx
    if phicx < -philimx:
        phicx = -philimx
    phixd_new = (phicx - phix) / tphi
    phix = phix + (phixd_new + phixd) * int_step / 2
    return phix, phixd_new


def test_control_bank_one_step_input1_tphi():
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    store.set("philimx", PHILIMX)
    store.set("tphi", TPHI)
    expected, expected_d = _expected_bank(30.0, 0.0, 0.0, PHILIMX, TPHI, INT_STEP)
    phix = control.control_bank(vehicle, 30.0, INT_STEP)
    assert phix == pytest.approx(expected, rel=RTOL, abs=ATOL)
    assert expected == pytest.approx(1.5, rel=RTOL, abs=ATOL)
    assert store.get("phixd") == pytest.approx(expected_d, rel=RTOL, abs=ATOL)
    assert store.get("phimvx") == 0.0
```

Also: second-step stored slope; clip `phicx=90` to 70; `test_name_is_control`; `test_define_registers_bank_fields` for `phimvx`/`phicx`/`phix`/`phixd`/`philimx`/`tphi`.

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 bank-angle control`

---

### Task 6: control_load

**Files:** Modify `control.py`; Test: `Python/tests/unit/test_cruise5_control_load.py`

**Interfaces:** Port `Cruise::control_load`. `control_load(vehicle, ancomx, int_step) -> float` (deg). Formula: `TBV=cadtbv(phimvx*RAD, alphax*RAD)`; `FSPB=TBV@FSPV`; clip `ancomx` to `[anneglimx, anposlimx]`; `anx=-fspb3/grav`; `eanx=ancomx-anx`; `tip=dvbe*mass/(pdynmc*area*cla/RAD+thrust)`; if `ta>0`: `gr=gacp*tip/dvbe`, `gi=gr/ta`, integrate `xi`; else `xi=0`; `qq=gr*eanx+xi`; integrate `alp` with `alpd_new=qq-alp/tip`; `alpx=alp*DEG` clipped to `[alpneglimx, alpposlimx]`. `input_1`: `gacp=10`, `ta=0.8`, `anposlimx=3`, `anneglimx=-1`, `alpposlimx=15`, `alpneglimx=-10`, `mass=1000`, `dvbe=200`, `area=0.929`. `execute` still pass.

- [ ] **Step 1:** one-step vs C++ replica using `input_1` gains and a frozen `FSPV=np.array([2.0, 1.0, -12.0])`, `grav=9.81`, `pdynmc=5000`, `thrust=1500`, `cla=0.11`, `ancomx=1.5`, `phimvx=0`, `alphax=0`. Replica must call `integrate` the same order as C++.

```python
from cadac.constants import DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.math.frames import cadtbv

def _expected_load(ancomx, int_step, phimvx, alphax, anposlimx, anneglimx,
                   gacp, ta, alpposlimx, alpneglimx, fspv, grav, mass, dvbe,
                   pdynmc, thrust, area, cla, xi, xid, alp, alpd):
    tbv = cadtbv(phimvx * RAD, alphax * RAD)
    fspb = tbv @ fspv
    if ancomx > anposlimx:
        ancomx = anposlimx
    if ancomx < anneglimx:
        ancomx = anneglimx
    anx = -fspb[2] / grav
    eanx = ancomx - anx
    tip = dvbe * mass / (pdynmc * area * cla / RAD + thrust)
    gr = 0.0
    if ta > 0:
        gr = gacp * tip / dvbe
        gi = gr / ta
        xid_new = gi * eanx
        xi = integrate(xid_new, xid, xi, int_step)
        xid = xid_new
    else:
        xi = 0.0
    qq = gr * eanx + xi
    alpd_new = qq - alp / tip
    alp = integrate(alpd_new, alpd, alp, int_step)
    alpx = alp * DEG
    if alpx > alpposlimx:
        alpx = alpposlimx
    if alpx < alpneglimx:
        alpx = alpneglimx
    return alpx, xi, alp, tip, qq, anx
```

Assert `control_load(...)` matches `_expected_load(...)` at `rtol=1e-12`.

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 load-factor control`

---

### Task 7: control_altitude

**Files:** Modify `control.py`; Test: `Python/tests/unit/test_cruise5_control_altitude.py`

**Interfaces:** Port `Cruise::control_altitude(altcom, phimvx)`. `ealt=gh*(altcom-alt)` clipped to `±altdlim`; `altd=-VBEG[2]`; `ancomx=(gv*(ealt-altd)/grav+1)*(1/cos(phimvx*RAD))`; clip to `[anneglimx, anposlimx]`; write `altd`. `input_1`: `gh=0.3`, `gv=1.0`, `altdlim=50`, `altcom=7000`, `alt=7000`, `anposlimx=3`, `anneglimx=-1`.

- [ ] **Step 1:** `alt=7000`, `altcom=7000`, `phimvx=0`, `vbeg=[200,0,0]`, `grav=9.81` → `ealt=0`, `altd=0`, `ancomx=1.0`. Second case `alt=6900` → `ealt=gh*100=30` (under `altdlim=50`), `ancomx=(1.0*(30-0)/9.81+1)*1`. Third case `alt=6000` → `ealt` clipped to 50.

```python
from math import cos
import numpy as np
import pytest
from cadac.constants import RAD
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.cruise5.control import Cruise5Control

RTOL = 1e-12
ATOL = 1e-14
GH = 0.3
GV = 1.0
ALTDLIM = 50.0
ALTCOM = 7000.0
GRAV = 9.81
ANPOS = 3.0
ANNEG = -1.0


def _expected_alt(altcom, phimvx, alt, vbeg, gh, gv, altdlim, grav, anpos, anneg):
    ealt = gh * (altcom - alt)
    if ealt > altdlim:
        ealt = altdlim
    if ealt < -altdlim:
        ealt = -altdlim
    altd = -vbeg[2]
    ancomx = (gv * (ealt - altd) / grav + 1) * (1 / cos(phimvx * RAD))
    if ancomx > anpos:
        ancomx = anpos
    if ancomx < anneg:
        ancomx = anneg
    return ancomx, altd
```

Wire a `_ready()` that `define`s control and sets `gh,gv,altdlim,anposlimx,anneglimx,alt,grav,vbeg`. Assert `control_altitude(vehicle, 7000.0, 0.0)` vs replica for `alt` in `{7000, 6900, 6000}`.

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 altitude control`

---

### Task 8: control_lateral and mcontrol dispatcher

**Files:** Modify `control.py`; Test: `Python/tests/unit/test_cruise5_mcontrol.py`

**Interfaces:** Port `Cruise::control_lateral`: `TBV=cadtbv(phimv,alpha)`; `anx=-FSPB[2]/grav`; clip `alcomx` to `±allimx`; `sign = 1 if anx>=0 else -1`; `phicx = DEG * gcp*sign/(abs(anx)+0.001)*alcomx`; diagnostic `alx=FSPV[1]/grav`. Dispatcher `execute`: `mcontrol` in `{0,44,46}` else `ValueError`. `0`: `phimvx=0`, `alphax=0`. `44`: `phicx=control_lateral(alcomx)`; `phimvx=control_bank(phicx,int_step)`; `alphax=control_load(ancomx,int_step)`. `46`: same lateral+bank then `ancomx=control_altitude(altcom,phimvx)`; `alphax=control_load(ancomx,int_step)`. Then `TBV=cadtbv(phimvx*RAD,alphax*RAD)`, `TBG=TBV@tgv.T`. Write `phicx`,`TBV`,`TBG`,`alphax`,`phimvx`,`ancomx`. `input_1`: `gcp=2`, `allimx=1`, `mcontrol=46`.

- [ ] **Step 1:**

```python
import numpy as np
import pytest
from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadtbv
from cadac.vehicles.round3.cruise5.control import Cruise5Control

RTOL = 1e-12
INT_STEP = 0.05


def _ctx():
    return SimContext(0.0, INT_STEP, 0.0, 0.0, None, 0)


def test_mcontrol_0_zeros_phimvx_alphax():
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field(
        "tgv", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "mat", "init", "newton",
    ))
    store.set("phimvx", 12.0)
    store.set("alphax", 5.0)
    store.set("mcontrol", 0)
    control.execute(vehicle, _ctx())
    assert store.get("phimvx") == 0.0
    assert store.get("alphax") == 0.0


def test_mcontrol_46_writes_finite_phimvx_alphax():
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field(
        "tgv", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "mat", "init", "newton",
    ))
    for name, value, ftype in (
        ("FSPV", np.array([2.0, 1.0, -12.0]), "vec"),
        ("grav", 9.81, "real"),
        ("mass", 1000.0, "real"),
        ("dvbe", 200.0, "real"),
        ("pdynmc", 5000.0, "real"),
        ("thrust", 1500.0, "real"),
        ("area", 0.929, "real"),
        ("cla", 0.11, "real"),
        ("vbeg", np.array([200.0, 0.0, 0.0]), "vec"),
        ("alt", 7000.0, "real"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, "out", "test"))
        else:
            store.set(name, value)
    store.set("mcontrol", 46)
    store.set("alcomx", 0.2)
    store.set("gcp", 2.0)
    store.set("allimx", 1.0)
    store.set("philimx", 70.0)
    store.set("tphi", 0.5)
    store.set("gacp", 10.0)
    store.set("ta", 0.8)
    store.set("anposlimx", 3.0)
    store.set("anneglimx", -1.0)
    store.set("alpposlimx", 15.0)
    store.set("alpneglimx", -10.0)
    store.set("altcom", 7000.0)
    store.set("gh", 0.3)
    store.set("gv", 1.0)
    store.set("altdlim", 50.0)
    control.execute(vehicle, _ctx())
    assert np.isfinite(store.get("phimvx"))
    assert np.isfinite(store.get("alphax"))
    assert store.get("TBV").shape == (3, 3)
    assert store.get("TBG").shape == (3, 3)


def test_mcontrol_44_uses_ancomx_not_altitude():
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field(
        "tgv", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "mat", "init", "newton",
    ))
    for name, value, ftype in (
        ("FSPV", np.array([2.0, 1.0, -12.0]), "vec"),
        ("grav", 9.81, "real"),
        ("mass", 1000.0, "real"),
        ("dvbe", 200.0, "real"),
        ("pdynmc", 5000.0, "real"),
        ("thrust", 1500.0, "real"),
        ("area", 0.929, "real"),
        ("cla", 0.11, "real"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, "out", "test"))
    store.set("mcontrol", 44)
    store.set("alcomx", 0.2)
    store.set("ancomx", 1.5)
    store.set("gcp", 2.0)
    store.set("allimx", 1.0)
    store.set("philimx", 70.0)
    store.set("tphi", 0.5)
    store.set("gacp", 10.0)
    store.set("ta", 0.8)
    store.set("anposlimx", 3.0)
    store.set("anneglimx", -1.0)
    store.set("alpposlimx", 15.0)
    store.set("alpneglimx", -10.0)
    control.execute(vehicle, _ctx())
    assert np.isfinite(store.get("alphax"))
    assert np.isfinite(store.get("phimvx"))


def test_mcontrol_unknown_raises():
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    vehicle.store.define(Field(
        "tgv", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "mat", "init", "newton",
    ))
    vehicle.store.set("mcontrol", 16)
    with pytest.raises(ValueError, match="mcontrol"):
        control.execute(vehicle, _ctx())
```

Also a unit of `control_lateral` vs replica: `anx=-fspb3/grav`; `phicx=DEG*gcp*sign(anx)/(abs(anx)+0.001)*alcomx` with `input_1` `gcp=2`, `allimx=1`, `alcomx=0.5`, `FSPV=[2,1,-12]`, `alphax=0`, `phimvx=0`, `grav=9.81`.

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 mcontrol 0/44/46 dispatcher`

---

### Task 9: guidance_line

**Files:**
- Create: `Python/src/cadac/vehicles/round3/cruise5/guidance.py`
- Test: `Python/tests/unit/test_cruise5_guidance_line.py`

**Interfaces:**
- Consumes: `cadine`, `polar_from_cart`, `mat2tr`, store `time`,`grav`,`tig`,`thtvgx`,`vbeg`,`sbii`,`philimx`
- Produces: `Cruise5Guidance`. `define` C++ `def_guidance`: `mguidance`, `pronav_gain`, `line_gain`, `nl_gain_fact` default 1, `decrement`, `wp_lonx`/`wp_latx`/`wp_alt`, `psifgx`/`thtfgx`, `point_gain`, `wp_sltrange` default 999999, `nl_gain`, `VBEO`, `VBEF`, `wp_grdrange` default 999999, `SWBG`, `rad_min`, `rad_geometric`, `wp_flag`. `guidance_line(vehicle) -> ndarray (3,)` ports `Cruise::guidance_line`: `TFG=mat2tr(psifgx*RAD, thtfgx*RAD)`; `SWII=cadine(wp_lonx*RAD, wp_latx*RAD, wp_alt, time)`; `SWBG=TIG.T@(SWII-SBII)`; polar → `wp_sltrange`,`psiog`,`thtog`; `TOG=mat2tr(psiog,thtog)`; `wp_grdrange=hypot(swbg1,swbg2)`; `VBEO=TOG@VBEG`; `VBEF=TFG@VBEG`; `nl_gain=nl_gain_fact*(1-exp(-wp_sltrange/decrement))`; `algv1=grav*sin(thtvgx*RAD)`; `algv2=line_gain*(-vbeo2+nl_gain*vbef2)`; `algv3=line_gain*(-vbeo3+nl_gain*vbef3)-grav*cos(thtvgx*RAD)`; `rad_min=dvbe**2/(grav*tan(philimx*RAD))`; if `wp_grdrange<2*rad_min`: `wp_flag=sign(VH·SH)` with CADAC `sign` (`<0 → -1` else `+1`) else 0. `execute` pass until Task 11. `input_1` waypoint 1: `wp_lonx=14.9`, `wp_latx=35.4`, `wp_alt=0`, `psifgx=90`, `thtfgx=0`, `line_gain=1`, `nl_gain_fact=0.6`, `decrement=1000`, UAV `lonx=14.7`, `latx=35.4`, `alt=7000`, `thtvgx=0`, `dvbe=200`, `philimx=70`.

- [ ] **Step 1:** Build `tig` from Round3 init or `cadtei`/`cadtge` at `time=0`, `lon=14.7*RAD`, `lat=35.4*RAD`; `sbii=cadine(...)`; `vbeg` from `mat2tr(psivgx*RAD, thtvgx*RAD).T @ [dvbe,0,0]` geographic velocity as Round3. Call `guidance_line`; assert `ALGV` finite, `wp_grdrange>0`, `wp_flag` in `{-1,0,1}`, and `ALGV` equals an in-test replica of the C++ lines above at `rtol=1e-12`. Replica must use the same `cadine`/`mat2tr`/`polar_from_cart` calls.

```python
from math import cos, exp, hypot, sin, sqrt, tan
import numpy as np
from cadac.constants import RAD
from cadac.math.earth import cadine
from cadac.math.frames import mat2tr, polar_from_cart

def _sign(variable):
    if variable < 0:
        return -1
    return 1

def _expected_line(wp_lonx, wp_latx, wp_alt, psifgx, thtfgx, line_gain,
                   nl_gain_fact, decrement, time, grav, tig, thtvgx, vbeg,
                   sbii, philimx):
    tfg = mat2tr(psifgx * RAD, thtfgx * RAD)
    swii = cadine(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
    swbg = tig.T @ (swii - sbii)
    polar = polar_from_cart(swbg)
    wp_sltrange = float(polar[0])
    tog = mat2tr(float(polar[1]), float(polar[2]))
    wp_grdrange = hypot(float(swbg[0]), float(swbg[1]))
    vbeo = tog @ vbeg
    vbef = tfg @ vbeg
    nl_gain = nl_gain_fact * (1 - exp(-wp_sltrange / decrement))
    algv = np.array([
        grav * sin(thtvgx * RAD),
        line_gain * (-vbeo[1] + nl_gain * vbef[1]),
        line_gain * (-vbeo[2] + nl_gain * vbef[2]) - grav * cos(thtvgx * RAD),
    ])
    dvbe = sqrt(float(vbeg @ vbeg))
    rad_min = dvbe * dvbe / (grav * tan(philimx * RAD))
    if wp_grdrange < 2 * rad_min:
        sh = np.array([swbg[0], swbg[1], 0.0])
        vh = np.array([vbeg[0], vbeg[1], 0.0])
        wp_flag = _sign(float(vh @ sh))
    else:
        wp_flag = 0
    return algv, wp_flag, wp_grdrange, nl_gain
```

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 line guidance`

---

### Task 10: guidance_point

**Files:** Modify `guidance.py`; Test: `Python/tests/unit/test_cruise5_guidance_point.py`

**Interfaces:** `guidance_point(vehicle) -> ndarray (3,)` ports `Cruise::guidance_point`: same `cadine`/`SWBG`/`TOG` as line; **no** LOA `TFG`/`VBEF`/`nl_gain`; `apgv1=grav*sin(thtvgx*RAD)`; `apgv2=point_gain*(-vbeo2)`; `apgv3=point_gain*(-vbeo3)-grav*cos(thtvgx*RAD)`; same `wp_flag` as line. Writes `wp_sltrange`,`VBEO`,`wp_grdrange`,`SWBG`,`rad_min`,`wp_flag`. Event-3 ICs: `point_gain=1`, `wp_lonx=15.4`, `wp_latx=35.3`, `wp_alt=100` (tank after targeting), UAV same as Task 9.

- [ ] **Step 1:** replica:

```python
def _expected_point(wp_lonx, wp_latx, wp_alt, point_gain, time, grav, tig,
                    thtvgx, vbeg, sbii, philimx):
    swii = cadine(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
    swbg = tig.T @ (swii - sbii)
    polar = polar_from_cart(swbg)
    tog = mat2tr(float(polar[1]), float(polar[2]))
    vbeo = tog @ vbeg
    apgv = np.array([
        grav * sin(thtvgx * RAD),
        point_gain * (-vbeo[1]),
        point_gain * (-vbeo[2]) - grav * cos(thtvgx * RAD),
    ])
    wp_grdrange = hypot(float(swbg[0]), float(swbg[1]))
    dvbe = sqrt(float(vbeg @ vbeg))
    rad_min = dvbe * dvbe / (grav * tan(philimx * RAD))
    if wp_grdrange < 2 * rad_min:
        sh = np.array([swbg[0], swbg[1], 0.0])
        vh = np.array([vbeg[0], vbeg[1], 0.0])
        wp_flag = _sign(float(vh @ sh))
    else:
        wp_flag = 0
    return apgv, wp_flag, wp_grdrange
```

Define `_sign` in this test file (CADAC: `<0 → -1` else `+1`). Assert `guidance_point` vs `_expected_point` at `rtol=1e-12` with `point_gain=1`, tank waypoint `15.4/35.3/100`, UAV `14.7/35.4/7000`.

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 point guidance`

---

### Task 11: mguidance dispatcher

**Files:** Modify `guidance.py`; Test: `Python/tests/unit/test_cruise5_mguidance.py`

**Interfaces:** `execute`: `mguidance==0` return (do not write `alcomx`/`ancomx`). Locked C++ mapping **before** limiters: `30` → `alcomx=ALGV[1]/grav`, `ancomx` stays 0; `43` → `alcomx=APGV[1]/grav`, `ancomx=-ALGV[2]/grav`. Else `ValueError` (including 03, 33, 40, 60, 66, 70). Then clip `ancomx` to `[anneglimx, anposlimx]` and `alcomx` to `±allimx`. Write `phicx` unchanged (not 70), `ancomx`, `alcomx`. `input_1` start `mguidance=30`; event 3 `mguidance=43`. Limiters `anposlimx=3`, `anneglimx=-1`, `allimx=1`.

- [ ] **Step 1:** Numeric mapping asserts (not “finite”). Build the Task 9 UAV/waypoint-1 store (`tig`,`sbii`,`vbeg`,`grav`,`philimx=70`,`anposlimx=3`,`anneglimx=-1`,`allimx=1`,`phicx`, line/point fields). Helpers:

```python
import pytest
from cadac.kernel.executive import SimContext
from cadac.vehicles.round3.cruise5.guidance import Cruise5Guidance

RTOL = 1e-12
ATOL = 1e-14


def _clip(alcomx, ancomx, anposlimx, anneglimx, allimx):
    if ancomx > anposlimx:
        ancomx = anposlimx
    if ancomx < anneglimx:
        ancomx = anneglimx
    if alcomx > allimx:
        alcomx = allimx
    if alcomx < -allimx:
        alcomx = -allimx
    return alcomx, ancomx


def test_mguidance_30_alcomx_from_line_ancomx_stays_0():
    vehicle, guidance = _ready(mguidance=30)
    store = vehicle.store
    store.set("alcomx", 7.0)
    store.set("ancomx", 7.0)
    algv = guidance.guidance_line(vehicle)
    grav = store.get("grav")
    want_al, want_an = _clip(
        float(algv[1] / grav),
        0.0,
        store.get("anposlimx"),
        store.get("anneglimx"),
        store.get("allimx"),
    )
    guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
    assert store.get("alcomx") == pytest.approx(want_al, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(want_an, rel=RTOL, abs=ATOL)
    assert want_an == 0.0


def test_mguidance_43_alcomx_from_point_ancomx_from_line_pitch():
    vehicle, guidance = _ready(mguidance=43)
    store = vehicle.store
    algv = guidance.guidance_line(vehicle)
    apgv = guidance.guidance_point(vehicle)
    grav = store.get("grav")
    want_al, want_an = _clip(
        float(apgv[1] / grav),
        float(-algv[2] / grav),
        store.get("anposlimx"),
        store.get("anneglimx"),
        store.get("allimx"),
    )
    guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
    assert store.get("alcomx") == pytest.approx(want_al, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(want_an, rel=RTOL, abs=ATOL)


def test_mguidance_0_does_not_write_commands():
    vehicle, guidance = _ready(mguidance=0)
    store = vehicle.store
    store.set("alcomx", 7.0)
    store.set("ancomx", 7.0)
    guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
    assert store.get("alcomx") == 7.0
    assert store.get("ancomx") == 7.0


def test_mguidance_66_and_negative_raise():
    vehicle, guidance = _ready(mguidance=66)
    with pytest.raises(ValueError, match="mguidance"):
        guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
    vehicle.store.set("mguidance", -1)
    with pytest.raises(ValueError, match="mguidance"):
        guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
```

`_ready(mguidance)` must define `Cruise5Guidance`, set UAV/WP1 kinematics from `input_1` (`lonx=14.7`, `latx=35.4`, `alt=7000`, `wp_lonx=14.9`, `wp_latx=35.4`, `wp_alt=0`, `psifgx=90`, `thtfgx=0`, `line_gain=1`, `nl_gain_fact=0.6`, `decrement=1000`, `point_gain=1`, `philimx=70`), and the limiter fields. Calling `guidance_line`/`guidance_point` before `execute` is for the expected values; `execute` recomputes from the same store.

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 mguidance 0/30/43 dispatcher`

---

### Task 12: seeker (mseeker 0)

**Files:**
- Create: `Python/src/cadac/vehicles/round3/cruise5/seeker.py`
- Test: `Python/tests/unit/test_cruise5_seeker.py`

**Interfaces:** `Cruise5Seeker.name=="seeker"`. `define` C++ `def_seeker`: `mseeker` int data/save scrn default 0, `acq_range`, `range_go` out plot,scrn, `STBG` vec out plot, `WOEB` vec out, `closing_speed` out, `time_go` out plot,scrn, `psisbx`/`thtsbx` out plot,scrn, `targ_com_slot` int save, `UTBB` vec out, `acquisition` int init/save scrn. `execute`: `mseeker==0` return; else `ValueError`. `input_1` `mseeker=0`, `acq_range=10000`.

- [ ] **Step 1:**

```python
import pytest
from cadac.kernel.executive import SimContext
from cadac.kernel.state import StateStore
from cadac.vehicles.round3.cruise5.seeker import Cruise5Seeker

def test_mseeker_0_no_write():
    vehicle = type("V", (), {"store": StateStore()})()
    seeker = Cruise5Seeker()
    seeker.define(vehicle)
    store = vehicle.store
    store.set("mseeker", 0)
    store.set("range_go", 7.0)
    seeker.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, [], 0))
    assert store.get("range_go") == 7.0
    assert store.get("mseeker") == 0

def test_mseeker_1_raises():
    vehicle = type("V", (), {"store": StateStore()})()
    seeker = Cruise5Seeker()
    seeker.define(vehicle)
    vehicle.store.set("mseeker", 1)
    with pytest.raises(ValueError, match="mseeker"):
        seeker.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, [], 0))
```

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 seeker mseeker 0 stub`

---

### Task 13: Cruise intercept

**Files:**
- Create: `Python/src/cadac/vehicles/round3/cruise5/intercept.py`
- Test: `Python/tests/unit/test_cruise5_intercept.py`

**Interfaces:** Port `Cruise::intercept` without `cout`/`exit`. `define` C++ `def_intercept`: `write` int save default **1**, `miss`, `hit_time`, `MISS_G`, `time_m`, `SBTGM`, `STMEG`, `SBMEG`. No `halt`. `execute`: if `alt<=0` and `write`: `write=0`, `vehicle.health=0`, `ctx.combus[ctx.vehicle_slot].status=0`. if `mguidance in (33, 43)` and `alt<=wp_alt` and `write`: `write=0`, `miss=||SWBG||`, health/status 0. `mguidance in (30, 40, 70)` and `wp_flag==-1`: compute horizontal miss, do not print, do not kill. Optional `mseeker==3` and `range_go<100` closest-approach as C++ (target `sbeg` by name; missile status 0, target status -1). No `sys.exit`.

- [ ] **Step 1:**

```python
import numpy as np
import pytest
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.cruise5.intercept import Cruise5Intercept

def _vehicle():
    v = type("V", (), {})()
    v.store = StateStore()
    v.health = 1
    return v

def _ctx(status=1):
    pkt = Packet(name="UAV", type="CRUISE3", status=status, vars={})
    return SimContext(0.0, 0.05, 0.0, 0.0, [pkt], 0)

def test_ground_impact_kills():
    v = _vehicle()
    mod = Cruise5Intercept()
    mod.define(v)
    for name, value, ftype in (
        ("alt", -1.0, "real"), ("time", 1.0, "real"), ("dvbe", 200.0, "real"),
        ("psivgx", 90.0, "real"), ("thtvgx", 0.0, "real"),
        ("sbeg", np.zeros(3), "vec"), ("mguidance", 30, "int"),
        ("wp_lonx", 14.9, "real"), ("wp_latx", 35.4, "real"), ("wp_alt", 0.0, "real"),
        ("SWBG", np.array([10.0, 20.0, 30.0]), "vec"), ("wp_flag", 0, "int"),
        ("mseeker", 0, "int"), ("range_go", 0.0, "real"),
        ("STBG", np.zeros(3), "vec"), ("closing_speed", 0.0, "real"),
        ("targ_com_slot", 0, "int"),
    ):
        if name not in v.store.names():
            v.store.define(Field(name, value, ftype, "data", "test"))
    ctx = _ctx()
    mod.execute(v, ctx)
    assert v.health == 0
    assert ctx.combus[0].status == 0
    assert v.store.get("write") == 0

def test_mguidance_43_alt_below_wp_alt_kills():
    v = _vehicle()
    mod = Cruise5Intercept()
    mod.define(v)
    swbg = np.array([3.0, 4.0, 0.0])
    for name, value, ftype in (
        ("alt", 50.0, "real"), ("time", 1.0, "real"), ("dvbe", 200.0, "real"),
        ("psivgx", 90.0, "real"), ("thtvgx", -50.0, "real"),
        ("sbeg", np.zeros(3), "vec"), ("mguidance", 43, "int"),
        ("wp_lonx", 15.4, "real"), ("wp_latx", 35.3, "real"), ("wp_alt", 100.0, "real"),
        ("SWBG", swbg, "vec"), ("wp_flag", 0, "int"),
        ("mseeker", 0, "int"), ("range_go", 0.0, "real"),
        ("STBG", np.zeros(3), "vec"), ("closing_speed", 0.0, "real"),
        ("targ_com_slot", 0, "int"),
    ):
        if name not in v.store.names():
            v.store.define(Field(name, value, ftype, "data", "test"))
    ctx = _ctx()
    mod.execute(v, ctx)
    assert v.health == 0
    assert ctx.combus[0].status == 0
    assert v.store.get("miss") == pytest.approx(5.0)

def test_mguidance_30_wp_flag_minus1_does_not_kill():
    v = _vehicle()
    mod = Cruise5Intercept()
    mod.define(v)
    for name, value, ftype in (
        ("alt", 7000.0, "real"), ("time", 1.0, "real"), ("dvbe", 200.0, "real"),
        ("psivgx", 90.0, "real"), ("thtvgx", 0.0, "real"),
        ("sbeg", np.zeros(3), "vec"), ("mguidance", 30, "int"),
        ("wp_lonx", 14.9, "real"), ("wp_latx", 35.4, "real"), ("wp_alt", 0.0, "real"),
        ("SWBG", np.array([10.0, 0.0, 0.0]), "vec"), ("wp_flag", -1, "int"),
        ("mseeker", 0, "int"), ("range_go", 0.0, "real"),
        ("STBG", np.zeros(3), "vec"), ("closing_speed", 0.0, "real"),
        ("targ_com_slot", 0, "int"),
    ):
        if name not in v.store.names():
            v.store.define(Field(name, value, ftype, "data", "test"))
    ctx = _ctx()
    mod.execute(v, ctx)
    assert v.health == 1
    assert ctx.combus[0].status == 1
```

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 intercept ground and point/line impact`

---

### Task 14: targeting mtargeting 0 and 1

**Files:**
- Create: `Python/src/cadac/vehicles/round3/cruise5/targeting.py`
- Test: `Python/tests/unit/test_cruise5_targeting.py`

**Interfaces:** `Cruise5Targeting`. `define` C++ `def_targeting`: `mtargeting` int data scrn,plot; `del_radius` data; `clost_tgt_slot`/`tgtng_sat_slot` int out. Module-level `BIG=1e10`. Local `_angle` / CADAC `sign` not required for targeting except `angle` as C++ `utility_functions.cpp`. `execute`: `mtargeting==0` return; `!=1` `ValueError`. `==1`: `targeting_satellite` then first `visibility` entry with `tracking`; `targeting_grnd_ranges` closest `Packet.type=="TARGET3"`; if satellite found write `wp_lonx`/`wp_latx`/`wp_alt` from that packet **by name**. `targeting_satellite`: packets `type=="SATELLITE3"`; `radius=REARTH+del_radius`; grazing `acos(radius/||SSII||)`; missile-sat `angle(SBII,SSII)`; if angle < grazing: tracking 1; else critical radius `radius/cos(angle-grazing)` if `|cos|>EPS`; tracking if `||SBII||>radius_crit`. Then **C++** `satellite_target_angle=angle(SBII, STII)` (missile vs first TARGET3 `sbii`, not SSII); if `> grazing_angle` set tracking 0. Ground range: `REARTH*acos(sin lat_t sin lat_c + cos lat_t cos lat_c cos(lon_t-lon_c))` with lon/lat in rad. `input_1` event 3: `del_radius=5000`. UAV `sbii` from `cadine(14.7*RAD, 35.4*RAD, 7000, 0)`; Tank `15.4/35.3/100`; Sat `10/30/500000`.

- [ ] **Step 1:** `mtargeting=0` does not write waypoints (pre-set `wp_lonx=14.9` stays). `mtargeting=2` raises. `mtargeting=1` with one UAV packet (`type="CRUISE3"`), one `TARGET3`, one `SATELLITE3` on combus: `wp_lonx==15.4`, `wp_latx==35.3`, `wp_alt==100`. Two TARGET3: closer ground range wins. Identify by `Packet.type` not name prefix. Import `cadine` for `sbii` in packets.

```python
from cadac.constants import RAD
from cadac.kernel.combus import Packet
from cadac.math.earth import cadine
from cadac.vehicles.round3.cruise5.targeting import Cruise5Targeting

UAV_LONX, UAV_LATX, UAV_ALT = 14.7, 35.4, 7000.0
TGT_LONX, TGT_LATX, TGT_ALT = 15.4, 35.3, 100.0
SAT_LONX, SAT_LATX, SAT_ALT = 10.0, 30.0, 500000.0
```

Build combus with `vars` keys `lonx`,`latx`,`alt`,`sbii` on Target and `sbii` on Satellite. UAV store: `mtargeting`,`del_radius=5000`,`lonx`,`latx`,`sbii`, plus `wp_lonx`/`wp_latx`/`wp_alt` fields.

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 satellite targeting`

---

### Task 15: Cruise5Target

**Files:**
- Create: `Python/src/cadac/vehicles/round3/cruise5/target.py`
- Test: `Python/tests/unit/test_cruise5_target.py`

**Interfaces:** `Cruise5Target.type=="TARGET3"`. Constructor `(name, events=None)` — no decks. `health=1`. Modules: `Round3Environment`, `Round3Newton`, `Cruise5TargetForces`, `Cruise5TargetIntercept` (forces may sit before newton as HYPER5 Target or env/newton/forces — C++ MODULES order is env, aero, prop, forces, newton, …; Target skips aero/prop; use `[Round3Environment(), Cruise5TargetForces(), Round3Newton(), Cruise5TargetIntercept()]` so FSPV exists before newton exec). Forces: C++ `Target::forces` — `WEIG=TGE@WEII@TEG`; `CORIO_V=TVG@WEIG@VBEG*2`; `CENTR_V=TVG@WEIG@WEIG@TGI@SBII`; `GRAV_V=TVG@[0,0,grav]`; `FSPV=[acc_v0+fwd_accel, acc_v1+side_accel, acc_v2]`. Skip-if-exists `FSPV`. Intercept: `targ_health=ctx.combus[slot].status`. `com_names` from `"com"` in outputs. Do not import `cadac.vehicles.round3.hyper5.target.Target3`.

- [ ] **Step 1:** `fwd_accel=side_accel=0`; after `define` + set Tank ICs `lonx=15.4`, `latx=35.3`, `alt=100`, `psivgx=45`, `dvbe=10`, `thtvgx=0`; `initialize` env+newton; `execute` forces → `FSPV` finite shape `(3,)`. Intercept: combus status 0 → `targ_health==0`. Constructor `Cruise5Target("Tank_t1")` has no aero_deck attribute required. `type=="TARGET3"`.

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 TARGET3 tank forces and intercept`

---

### Task 16: Cruise5Satellite

**Files:**
- Create: `Python/src/cadac/vehicles/round3/cruise5/satellite.py`
- Test: `Python/tests/unit/test_cruise5_satellite.py`

**Interfaces:** `Cruise5Satellite.type=="SATELLITE3"`. Constructor `(name, events=None)`. Modules: `Round3Environment`, `Cruise5SatelliteForces`, `Round3Newton`. `sat_thrust` data default 0, `sat_mass` data default **100**. `FSPV=np.array([sat_thrust/sat_mass, 0.0, 0.0])`. Skip-if-exists `FSPV`. After `define()`, `com_names` is the same collection as Cruise5Target: store fields with `"com"` in outputs (Round3: at least `lonx`, `latx`, `alt`, `sbii`). Targeting reads SATELLITE3 packets by those names. Do not import HYPER5 `Satellite3`.

- [ ] **Step 1:** `sat_thrust=0`, `sat_mass=100` → `FSPV[0]==0`; `sat_thrust=100` → `FSPV[0]==1`. `type=="SATELLITE3"`. `Cruise5Satellite("Sat_s1")`. After `define()`, `{"lonx", "latx", "alt", "sbii"} <= set(vehicle.com_names)`.

- [ ] **Step 2–5:** implement, pass, commit `feat: CRUISE5 SATELLITE3 forces`

---

### Task 17: Cruise5 vehicle + family registry + translate input_1

**Files:**
- Create: `Python/src/cadac/vehicles/round3/cruise5/vehicle.py`
- Modify: `Python/src/cadac/cli.py` — register three family pairs only
- Translate: `CADAC_Simulations/CRUISE5_250115/CRUISE5/input_1.asc` + `cruise3_aero_deck.asc` + `cruise3_prop_deck.asc` → `Python/cases/cruise5/` with `family="cruise5"`
- Test: `Python/tests/unit/test_cruise5_one_step.py`

**Interfaces:**
- `Cruise5.type=="CRUISE3"`. Constructor `(name, aero_deck, prop_deck, events=None)` — both decks required. `health=1`. Modules in `input_1.asc` order: `Round3Environment`, `Cruise5Aero(aero_deck)`, `Cruise5Propulsion(prop_deck)`, `Cruise5Forces`, `Round3Newton`, `Cruise5Targeting`, `Cruise5Seeker`, `Cruise5Guidance`, `Cruise5Control`, `Cruise5Intercept`.
- `_VEHICLE_FAMILIES[("cruise5","CRUISE3")]=Cruise5`, `[("cruise5","TARGET3")]=Cruise5Target`, `[("cruise5","SATELLITE3")]=Cruise5Satellite`. Do **not** assign `_VEHICLE_TYPES["CRUISE3"]` etc.
- `run_scenario`: family `cruise5` + `CRUISE3` requires aero **and** prop decks; `TARGET3`/`SATELLITE3` still `_NO_DECK_TYPES` constructors `(name, events)`.
- Translated JSONC: `end_time==410`, UAV params `mprop==4`, `mcontrol==46`, `mguidance==30`, `mseeker==0`, `alt==7000`, `lonx==14.7`; Tank `lonx==15.4`; Sat `alt==500000`; every vehicle `family=="cruise5"`. All three UAV events are `wp_flag = -1` with the exact `when`/`set` dicts below.
- Smoke: `run_scenario` with `end_time` 0.05 on a tmp copy of the case (or `run_loop` 0.05 s on constructed vehicles). UAV `alt` finite; all three `health==1`.
- `family=None` `type=="CRUISE3"` still builds HYPER3 `Cruise3`. `family="cruise5"` `type=="PLANE"` raises (no fallthrough).

- [ ] **Step 1:** smoke FAIL then PASS; translated JSONC loads; Tank/Sat have no `aero_deck`; `_VEHICLE_TYPES["CRUISE3"]` is still `Cruise3`; `_VEHICLE_FAMILIES[("cruise5","CRUISE3")]` is `Cruise5`.

```python
from pathlib import Path

from cadac.cli import _VEHICLE_FAMILIES, _VEHICLE_TYPES
from cadac.io.scenario import load_scenario
from cadac.vehicles.round3.hyper3.vehicle import Cruise3
from cadac.vehicles.round3.cruise5.satellite import Cruise5Satellite
from cadac.vehicles.round3.cruise5.target import Cruise5Target
from cadac.vehicles.round3.cruise5.vehicle import Cruise5
from cadac.vehicles.round3.hyper5.target import Target3

CASES = Path(__file__).resolve().parents[2] / "cases" / "cruise5"

WP_FLAG_WHEN = {"wp_flag": {"=": -1}}

def test_family_pairs_do_not_overwrite_globals():
    assert _VEHICLE_TYPES["CRUISE3"] is Cruise3
    assert _VEHICLE_TYPES["TARGET3"] is Target3
    assert _VEHICLE_FAMILIES[("cruise5", "CRUISE3")] is Cruise5
    assert _VEHICLE_FAMILIES[("cruise5", "TARGET3")] is Cruise5Target
    assert _VEHICLE_FAMILIES[("cruise5", "SATELLITE3")] is Cruise5Satellite

def test_translated_input1_three_wp_flag_events():
    cfg = load_scenario(CASES / "input_1.jsonc")
    assert cfg.end_time == 410
    assert cfg.vehicles[0].family == "cruise5"
    assert cfg.vehicles[0].params["gfthm"] == 893620
    events = cfg.vehicles[0].events
    assert len(events) == 3
    assert events[0].when == WP_FLAG_WHEN
    assert events[0].set == {
        "wp_lonx": 15.25, "wp_latx": 35.54, "psifgx": 90, "altcom": 5000,
    }
    assert events[1].when == WP_FLAG_WHEN
    assert events[1].set == {
        "wp_lonx": 15.43, "wp_latx": 35.44, "psifgx": 180, "altcom": 2000,
    }
    assert events[2].when == WP_FLAG_WHEN
    assert events[2].set == {
        "mtargeting": 1, "del_radius": 5000, "mguidance": 43,
        "point_gain": 1, "line_gain": 1, "nl_gain_fact": 0.6,
        "decrement": 1000, "thtfgx": -50, "mcontrol": 44,
    }
```

Event `set` values follow translator `_parse_number` (ints stay int).

- [ ] **Step 2–4:** wire, then PASS `tests/unit/test_cruise5_one_step.py` **and** `tests/unit/test_cruise3_one_step.py` **and** `tests/unit/test_hyper5_one_step.py`
- [ ] **Step 5: Commit** `feat: run CRUISE5 family vehicles from JSONC`

---

### Task 18: CRUISE5 e2e golden (optional file)

**Files:** `Python/tests/e2e/test_cruise5_input1.py`

**Interfaces:** Skip if `tests/e2e/goldens/cruise5/plot.csv` absent. Else `run_scenario` on `Python/cases/cruise5/input_1.jsonc` (the translated stem). Compare UAV (slot 0) plot columns present in both; skip sentinel `time=-1`; CSV `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`. Helpers: `require_golden`, `load_cadac_plot_csv`, `_compare_alt_t0`, `_compare_shared_columns` — write the functions in this file (do not import HYPER5 e2e helpers by private name; copy the same comparison logic). Pattern of `tests/e2e/test_hyper5_pronav.py`. Do not create the golden file.

- [ ] **Step 1:**

```python
from pathlib import Path
import numpy as np
import pytest
from cadac import run_scenario
from cadac.io.plot import write_plot_csv

CASE = Path(__file__).resolve().parents[2] / "cases" / "cruise5" / "input_1.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "cruise5" / "plot.csv"
RTOL = 1e-5

def _csv_atol(golden):
    return max(1e-6, 5e-6 * abs(golden))

def require_golden(path: Path):
    if not path.is_file():
        pytest.skip("CRUISE5 golden plot.csv is absent")

def test_require_golden_skips_when_missing(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="CRUISE5 golden plot.csv is absent"):
        require_golden(tmp_path / "plot.csv")

def test_alt_matches_golden_at_t0():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    # load golden, skip time==-1, compare alt at t=0 with RTOL/_csv_atol
```

Include `test_shared_plot_columns_match_golden` that zips shared column names at each golden time. Fixture helpers that round-trip a tiny CSV via `write_plot_csv` so skip/compare logic is tested without the golden.

- [ ] **Step 2–5:** implement, pass, commit `test: CRUISE5 e2e gate (skip without golden)`

---

## Self-review

- Spec coverage: family dispatch (scenario default + vehicle wins; no `_VEHICLE_FAMILIES={}`), plot-by-class, aero (area 0.929, no alphax), propulsion 0–4 with early return, forces FSPV, control bank/load/lateral/altitude and mcontrol 0/44/46, guidance_line/point and locked mguidance 30/43 mapping, seeker 0, intercept ground+43, targeting 0/1 with named combus, Cruise5Target, Cruise5Satellite `com_names`, vehicle registry + three `wp_flag` events, translate `input_1` family=cruise5, e2e skip — each has a task
- Unused C++ guidance 03/33/40/60/06/66/70 and mcontrol 1/10/11/3/4/40/6/16/36 and mseeker 1/3: ValueError (spec)
- No HYPER3/HYPER5 class imports
- Global CRUISE3/TARGET3/SATELLITE3 not overwritten; Task 17 asserts `_VEHICLE_TYPES["CRUISE3"] is Cruise3` and re-runs `test_cruise3_one_step` / `test_hyper5_one_step`
- Unknown-type tests not retargeted (no new AIM5 test in this plan)
- `input.asc` 9-vehicle out of scope
- cadine / skip-missing / plot slot 0 reused, not reimplemented
- No TBD / “similar to Task N”
- Type names: `Cruise5`, `Cruise5Target`, `Cruise5Satellite`, `Cruise5Aero`, `Cruise5Propulsion`, `Cruise5Forces`, `Cruise5Control`, `Cruise5Guidance`, `Cruise5Seeker`, `Cruise5Intercept`, `Cruise5Targeting` consistent across tasks
- `mass_init`/`fuel_init` (not HYPER3 `mass0`); `area` default 0.929; `tphi=0.5`; event 3 `mguidance=43` `mcontrol=44` `thtfgx=-50`
