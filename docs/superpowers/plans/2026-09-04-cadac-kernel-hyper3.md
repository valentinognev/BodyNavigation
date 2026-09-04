# CADAC kernel + HYPER3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Installable `cadac` package that loads JSONC, interpolates CADAC decks, steps a CRUISE3/HYPER3 vehicle, and matches `plot1.csv` within CSV tolerances.

**Architecture:** Named numpy state, module protocol, CADAC trapezoidal `integrate` and `look_up`, ISO 62 + Newtonian gravity, Round3 EOM + Cruise3 aero/prop/forces. Runtime JSONC only; ASC is a translator source.

**Tech Stack:** Python >= 3.11, numpy, pytest. Work in `Python/`.

**Spec:** `docs/superpowers/specs/2026-09-04-cadac-python-design.md`

**Follow-on plans (do not implement in this plan):** `docs/superpowers/plans/2026-09-04-cadac-falcon5.md`, `docs/superpowers/plans/2026-09-04-cadac-falcon6.md`

## Global Constraints

- Python >= 3.11; dependencies numpy and pytest only
- Runtime reads JSONC only; CADAC `.asc` is translator input
- `integrate`: `y + (dydx_new + dydx) * int_step / 2` (CADAC stored-slope trapezoid)
- `look_up` / `find_index` / extrapolation match `CADAC_Simulations/HYPER3_250114/HYPER3/utility_functions.cpp`
- Constants verbatim from spec (`REARTH`, `WEII3`, `RAD`, `DEG`, `AGRAV`, `G`, `EARTH_MASS`, `R`, `PI`, `EPS=1e-10`)
- Unit tests: `rtol=1e-12`, `atol=1e-14` unless a task says otherwise
- E2E vs CSV: `rtol=1e-5`, `atol=max(1e-6, 5e-6*abs(golden))`
- Named state, numpy `(3,)` / `(3,3)`; no `Variable[i]` arrays
- No scipy ODE, no RK4, no ISO/US76 merge
- TDD: failing test, then minimal code
- Grok implementer and Grok reviewer; no Composer Fast, no Kimi 3, no Fast mode
- Package root `Python/`; import `cadac`
- Do not implement FALCON5, FALCON6, MAGSIX, or other vehicles in this plan

## File map

Create under `Python/`:

- `pyproject.toml` — package metadata, pytest, script `cadac`
- `src/cadac/__init__.py` — public exports as tasks add them
- `src/cadac/constants.py`
- `src/cadac/cli.py` — `run` / `translate-asc` (wired in later tasks)
- `src/cadac/io/jsonc.py`, `deck.py`, `asc_deck.py`, `scenario.py`, `asc_scenario.py`, `plot.py`, `translate.py`
- `src/cadac/tables/lookup.py`
- `src/cadac/kernel/integrate.py`, `state.py`, `events.py`, `module.py`, `executive.py`, `combus.py`
- `src/cadac/env/us76.py`, `iso62.py`, `gravity.py`
- `src/cadac/math/frames.py`, `earth.py`
- `src/cadac/eom/round3.py`
- `src/cadac/vehicles/cruise3/{aero,propulsion,forces,vehicle}.py`
- `tests/unit/...`, `tests/translate/...`, `tests/e2e/...`
- `cases/hyper3/` — translated JSONC + decks after translator exists

C++ sources of truth: `CADAC_Simulations/HYPER3_250114/HYPER3/` (`utility_functions.cpp`, `round3_modules.cpp`, `cruise_modules.cpp`, `execution.cpp`, `input_climb.asc`, `plot1.csv`).

---

### Task 1: Package scaffold

**Files:**
- Create: `Python/pyproject.toml`
- Create: `Python/src/cadac/__init__.py` (empty `__all__ = []`)
- Create: `Python/tests/unit/test_scaffold.py`
- Create: `Python/src/cadac/io/__init__.py`, `kernel/__init__.py`, `tables/__init__.py`, `env/__init__.py`, `math/__init__.py`, `eom/__init__.py`, `vehicles/__init__.py`, `vehicles/cruise3/__init__.py`

**Interfaces:**
- Consumes: nothing
- Produces: installable `cadac` 0.1.0, pytest collects tests

- [ ] **Step 1: Write the failing test**

```python
# Python/tests/unit/test_scaffold.py
import cadac

def test_package_importable():
    assert cadac.__name__ == "cadac"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && python -m pytest tests/unit/test_scaffold.py -v`
Expected: FAIL (package not installed / not found)

- [ ] **Step 3: Write pyproject.toml and empty package**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cadac"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Empty `__init__.py` files as listed. Install editable: `pip install -e ".[dev]"` from `Python/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && python -m pytest tests/unit/test_scaffold.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Python/pyproject.toml Python/src/cadac Python/tests/unit/test_scaffold.py
git commit -m "feat: scaffold cadac Python package"
```

---

### Task 2: CADAC constants

**Files:**
- Create: `Python/src/cadac/constants.py`
- Create: `Python/tests/unit/test_constants.py`
- Modify: `Python/src/cadac/__init__.py` — do not re-export constants yet unless needed

**Interfaces:**
- Consumes: nothing
- Produces: module `cadac.constants` with floats named `REARTH`, `WEII3`, `RAD`, `DEG`, `AGRAV`, `G`, `EARTH_MASS`, `R`, `PI`, `EPS` at spec values

- [ ] **Step 1: Write the failing test**

```python
from cadac.constants import REARTH, WEII3, RAD, DEG, AGRAV, G, EARTH_MASS, R, PI, EPS

def test_cadac_constants():
    assert REARTH == 6370987.308
    assert WEII3 == 7.292115e-5
    assert RAD == 0.0174532925199432
    assert DEG == 57.2957795130823
    assert AGRAV == 9.80675445
    assert G == 6.673e-11
    assert EARTH_MASS == 5.973e24
    assert R == 287.053
    assert PI == 3.1415927
    assert EPS == 1e-10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && python -m pytest tests/unit/test_constants.py -v`
Expected: FAIL import error

- [ ] **Step 3: Write constants.py with those exact assignments**

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && python -m pytest tests/unit/test_constants.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Python/src/cadac/constants.py Python/tests/unit/test_constants.py
git commit -m "feat: add CADAC physical constants"
```

---

### Task 3: JSONC loader

**Files:**
- Create: `Python/src/cadac/io/jsonc.py`
- Create: `Python/tests/unit/test_jsonc.py`

**Interfaces:**
- Consumes: nothing
- Produces: `loads(text: str) -> object` and `load(path) -> object`. Strip `//` to EOL and `/* */` (non-nested), then `json.loads`. Do not allow trailing commas.

- [ ] **Step 1: Write the failing test**

```python
from cadac.io.jsonc import loads

def test_loads_line_comment_after_value():
    text = '{ "lonx": -80.55, // Vehicle longitude - deg\n  "alt": 3000 }'
    assert loads(text) == {"lonx": -80.55, "alt": 3000}

def test_loads_block_comment():
    assert loads('{ "a": 1, /* skip */ "b": 2 }') == {"a": 1, "b": 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && python -m pytest tests/unit/test_jsonc.py -v`
Expected: FAIL

- [ ] **Step 3: Implement comment strip then json.loads.** Scan characters; if inside a JSON string (`"` with escapes), do not treat `//` as comment.

- [ ] **Step 4: Run tests**

Run: `cd Python && python -m pytest tests/unit/test_jsonc.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Python/src/cadac/io/jsonc.py Python/tests/unit/test_jsonc.py
git commit -m "feat: load JSONC with CADAC-style comments"
```

---

### Task 4: Table dataclass and JSONC deck load

**Files:**
- Create: `Python/src/cadac/tables/lookup.py` (Table + Datadeck load only; look_up in later tasks)
- Create: `Python/src/cadac/io/deck.py`
- Create: `Python/tests/unit/test_deck_jsonc.py`

**Interfaces:**
- Consumes: `cadac.io.jsonc.loads`
- Produces:
  - `@dataclass Table`: `name: str`, `dim: int`, `x1: np.ndarray`, `x2: np.ndarray | None`, `x3: np.ndarray | None`, `values: np.ndarray`
  - `load_deck(path) -> list[Table]`
  - `Datadeck.from_tables(tables)` with `.table(name) -> Table` raising `KeyError` if missing
  - 1D `values.shape == (len(x1),)`; 2D `(len(x1), len(x2))`; 3D `(len(x1), len(x2), len(x3))` — validate and raise `ValueError` on mismatch

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from pathlib import Path
from cadac.io.deck import load_deck

def test_load_1d_table(tmp_path: Path):
    p = tmp_path / "d.jsonc"
    p.write_text(
        '{ "title": "t", "tables": [ { "name": "cd0_vs_mach", "dim": 1, '
        '"x1": [0.4, 0.6], "values": [0.034, 0.0337] } ] }'
    )
    tables = load_deck(p)
    t = tables[0]
    assert t.name == "cd0_vs_mach"
    assert t.dim == 1
    np.testing.assert_allclose(t.x1, [0.4, 0.6])
    np.testing.assert_allclose(t.values, [0.034, 0.0337])
```

- [ ] **Step 2: Run to verify fail**

Run: `cd Python && python -m pytest tests/unit/test_deck_jsonc.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Table, Datadeck, load_deck**

- [ ] **Step 4: Run tests — PASS**

- [ ] **Step 5: Commit** `feat: load JSONC aerodynamic/propulsion decks`

---

### Task 5: Parse CADAC 1DIM ASC deck

**Files:**
- Create: `Python/src/cadac/io/asc_deck.py`
- Create: `Python/tests/translate/test_asc_deck_1d.py`

**Interfaces:**
- Consumes: `Table` from Task 4
- Produces: `parse_asc_deck(path) -> tuple[str, list[Table]]` (title, tables). 1DIM block: `1DIM name` then `NX1 n` then `n` lines `x y`. Ignore non-table prose after TITLE until `1DIM`/`2DIM`/`3DIM`.

- [ ] **Step 1: Failing test using real file**

```python
from pathlib import Path
from cadac.io.asc_deck import parse_asc_deck

HYPER3 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/HYPER3_250114/HYPER3/ghame3_aero_deck.asc"

def test_parse_cd0_vs_mach():
    title, tables = parse_asc_deck(HYPER3)
    by = {t.name: t for t in tables}
    t = by["cd0_vs_mach"]
    assert t.dim == 1
    assert t.x1[0] == 0.4
    assert t.values[0] == 0.0340
    assert len(t.x1) == 13
```

- [ ] **Step 2: Run — FAIL**

Run: `cd Python && python -m pytest tests/translate/test_asc_deck_1d.py -v`

- [ ] **Step 3: Implement 1DIM parser (2DIM/3DIM may raise `NotImplementedError` until later tasks)**

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit** `feat: parse CADAC 1DIM table decks`

---

### Task 6: Parse CADAC 2DIM ASC deck

**Files:**
- Modify: `Python/src/cadac/io/asc_deck.py`
- Create: `Python/tests/translate/test_asc_deck_2d.py`

**Interfaces:**
- Consumes: Task 5 parser
- Produces: 2DIM `values[i1, i2]` with `x1` = row breakpoints, `x2` = column breakpoints. CADAC layout: each data row starts with x1, first row also has first x2 then row of values; remaining x2 listed after the matrix as a dangling column (see `ghame3_prop_deck.asc`). Port the packing used by HYPER3 `Cruise::read_tables` in `cruise_functions.cpp`.

- [ ] **Step 1: Failing test**

```python
from pathlib import Path
from cadac.io.asc_deck import parse_asc_deck

PROP = Path(__file__).resolve().parents[3] / "CADAC_Simulations/HYPER3_250114/HYPER3/ghame3_prop_deck.asc"

def test_parse_ca_vs_alpha_mach():
    _, tables = parse_asc_deck(PROP)
    t = {x.name: x for x in tables}["ca_vs_alpha_mach"]
    assert t.dim == 2
    assert t.x1.shape == (9,)
    assert t.x2.shape == (13,)
    assert t.values.shape == (9, 13)
    assert t.x1[0] == -3
    assert t.x2[0] == 0.4
    assert t.values[0, 0] == 1.09449
```

Read `read_tables` in `CADAC_Simulations/HYPER3_250114/HYPER3/cruise_functions.cpp` and match its 2DIM extraction exactly.

- [ ] **Step 2: FAIL** `pytest tests/translate/test_asc_deck_2d.py -v`
- [ ] **Step 3: Implement 2DIM**
- [ ] **Step 4: PASS**
- [ ] **Step 5: Commit** `feat: parse CADAC 2DIM table decks`

---

### Task 7: 1D look_up

**Files:**
- Modify: `Python/src/cadac/tables/lookup.py`
- Create: `Python/tests/unit/test_lookup_1d.py`

**Interfaces:**
- Consumes: `Table`, `EPS` from constants
- Produces: `Datadeck.look_up(name, x1) -> float`. `find_index` as C++ `Datadeck::find_index`. 1D: if loc==n-1 return data[-1]; else linear interpolate; slope extrapolation below min (`loc=0` still interpolates toward index 1). `dx>EPS` else dumx=0.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from cadac.tables.lookup import Datadeck, Table

def deck():
    t = Table(name="y", dim=1, x1=np.array([0.0, 10.0]), x2=None, x3=None, values=np.array([0.0, 10.0]))
    return Datadeck.from_tables([t])

def test_midpoint():
    assert deck().look_up("y", 5.0) == 5.0

def test_upper_constant():
    assert deck().look_up("y", 20.0) == 10.0

def test_lower_slope():
    assert deck().look_up("y", -5.0) == -5.0

def test_on_node():
    assert deck().look_up("y", 10.0) == 10.0
```

- [ ] **Step 2: FAIL** `pytest tests/unit/test_lookup_1d.py -v`
- [ ] **Step 3: Implement find_index + 1D interpolate + look_up**
- [ ] **Step 4: PASS**
- [ ] **Step 5: Commit** `feat: CADAC 1D table look_up`

---

### Task 8: 2D look_up

**Files:**
- Modify: `Python/src/cadac/tables/lookup.py`
- Create: `Python/tests/unit/test_lookup_2d.py`

**Interfaces:**
- Consumes: Task 7 `find_index`
- Produces: `look_up(name, x1, x2)` bilinear as C++ `interpolate` 2D (constant upper per axis, slope lower)

- [ ] **Step 1: Failing test**

```python
import numpy as np
from cadac.tables.lookup import Datadeck, Table

def deck():
    t = Table(
        name="z", dim=2,
        x1=np.array([0.0, 10.0]), x2=np.array([0.0, 10.0]), x3=None,
        values=np.array([[0.0, 10.0], [10.0, 20.0]]),
    )
    return Datadeck.from_tables([t])

def test_center():
    assert deck().look_up("z", 5.0, 5.0) == 10.0

def test_upper_x1_constant():
    assert deck().look_up("z", 20.0, 0.0) == 10.0
```

- [ ] **Step 2: FAIL**
- [ ] **Step 3: Implement 2D interpolate**
- [ ] **Step 4: PASS**
- [ ] **Step 5: Commit** `feat: CADAC 2D table look_up`

---

### Task 9: ASC deck to JSONC round-trip

**Files:**
- Create: `Python/src/cadac/io/translate.py` function `deck_asc_to_jsonc(src, dst)`
- Create: `Python/tests/translate/test_deck_roundtrip.py`

**Interfaces:**
- Consumes: `parse_asc_deck`, `load_deck`
- Produces: writes JSONC; `x1/x2/values` from reload equal ASC parse (`np.testing.assert_array_equal`)

- [ ] **Step 1: Failing test** on `ghame3_aero_deck.asc` and `ghame3_prop_deck.asc`
- [ ] **Step 2: FAIL**
- [ ] **Step 3: Emit JSONC (comments optional in this task). Use json dump of title+tables with lists.**
- [ ] **Step 4: PASS**
- [ ] **Step 5: Commit** `feat: translate CADAC decks ASC to JSONC`

---

### Task 10: integrate

**Files:**
- Create: `Python/src/cadac/kernel/integrate.py`
- Create: `Python/tests/unit/test_integrate.py`

**Interfaces:**
- Produces: `integrate(dydx_new, dydx, y, dt)` returning `y + (dydx_new + dydx) * dt / 2` for scalars and ndarrays

- [ ] **Step 1:**

```python
import numpy as np
from cadac.kernel.integrate import integrate

def test_scalar():
    assert integrate(3.0, 1.0, 10.0, 0.5) == 11.0

def test_vec():
    y = integrate(np.array([2.0, 0.0]), np.array([0.0, 2.0]), np.array([0.0, 0.0]), 1.0)
    np.testing.assert_allclose(y, [1.0, 1.0])
```

- [ ] **Step 2: FAIL**
- [ ] **Step 3: Implement**
- [ ] **Step 4: PASS**
- [ ] **Step 5: Commit** `feat: CADAC trapezoidal integrate`

---

### Task 11: US76 atmosphere

**Files:**
- Create: `Python/src/cadac/env/us76.py`
- Create: `Python/tests/unit/test_us76.py`

**Interfaces:**
- Produces: `atmosphere76(balt_m) -> tuple[rho, press, tempk]` copied from `atmosphere76` in HYPER3 `utility_functions.cpp` (internal `rearth=6369.0` km, not `REARTH`)

- [ ] **Step 1:** sea-level and 11 km geometric. At balt=0: rho=1.225, press=101325, tempk=288.15 (CADAC tables). Implement C++ then assert those.

```python
from cadac.env.us76 import atmosphere76

def test_sea_level():
    rho, press, tempk = atmosphere76(0.0)
    assert abs(rho - 1.225) < 1e-12
    assert abs(press - 101325) < 1e-12
    assert abs(tempk - 288.15) < 1e-12
```

- [ ] **Step 2: FAIL**
- [ ] **Step 3: Port C++ function line-for-line to Python**
- [ ] **Step 4: PASS**
- [ ] **Step 5: Commit** `feat: US 1976 standard atmosphere`

---

### Task 12: ISO 62 atmosphere (Round3)

**Files:**
- Create: `Python/src/cadac/env/iso62.py`
- Create: `Python/tests/unit/test_iso62.py`

**Interfaces:**
- Consumes: `R` from constants
- Produces: `iso62(alt_m, dvbe) -> dict` with keys `k` (temperature), `press`, `rho`, `vsound`, `mach`, `pdynmc` using spec ISO 62 formulas

- [ ] **Step 1:**

```python
from cadac.constants import R
from cadac.env.iso62 import iso62

def test_below_tropopause():
    out = iso62(3000.0, 250.0)
    k = 288.15 - 0.0065 * 3000.0
    press = 101325.0 * (k / 288.15) ** 5.2559
    rho = press / (R * k)
    vsound = (1.4 * R * k) ** 0.5
    assert abs(out["press"] - press) < 1e-12
    assert abs(out["rho"] - rho) < 1e-12
    assert abs(out["mach"] - abs(250.0 / vsound)) < 1e-12
```

- [ ] **Step 2: FAIL**
- [ ] **Step 3: Implement both branches (alt<11000 and else)**
- [ ] **Step 4: PASS** including a test at alt=20000 for the stratosphere branch
- [ ] **Step 5: Commit** `feat: ISO 62 atmosphere for Round3`

---

### Task 13: Newtonian gravity

**Files:**
- Create: `Python/src/cadac/env/gravity.py`
- Create: `Python/tests/unit/test_gravity.py`

**Interfaces:**
- Produces: `gravity(alt_m) -> float` = `G * EARTH_MASS / (REARTH + alt)**2`

- [ ] **Step 1:**

```python
from cadac.constants import G, EARTH_MASS, REARTH
from cadac.env.gravity import gravity

def test_at_altitude():
    alt = 3000.0
    assert gravity(alt) == G * EARTH_MASS / (REARTH + alt) ** 2
```

- [ ] **Step 2–5:** implement, pass, commit `feat: Newtonian gravity`

---

### Task 14: polar_from_cart and mat2tr

**Files:**
- Create: `Python/src/cadac/math/frames.py`
- Create: `Python/tests/unit/test_frames.py`

**Interfaces:**
- Produces: `polar_from_cart(v: ndarray (3,)) -> (d, azimuth, elevation)` CADAC `Matrix::pol_from_cart`; `mat2tr(psivg, thtvg) -> (3,3)` CADAC `mat2tr`

- [ ] **Step 1:**

```python
import numpy as np
from cadac.math.frames import polar_from_cart, mat2tr

def test_polar_east():
    d, az, el = polar_from_cart(np.array([0.0, 250.0, 0.0]))
    np.testing.assert_allclose([d, az, el], [250.0, np.arctan2(250.0, 0.0), 0.0], atol=1e-14)

def test_mat2tr_level_north():
    T = mat2tr(0.0, 0.0)
    np.testing.assert_allclose(T, np.eye(3), atol=1e-14)
```

Port `mat2tr` element-by-element from `utility_functions.cpp` (do not invent a DCM).

- [ ] **Step 2–5:** implement, pass, commit `feat: CADAC polar_from_cart and mat2tr`

---

### Task 15: cadtei, cadtge, cadsph

**Files:**
- Create: `Python/src/cadac/math/earth.py`
- Create: `Python/tests/unit/test_earth.py`

**Interfaces:**
- Consumes: `WEII3`, `REARTH`, `PI`
- Produces: `cadtei(sim_time) -> (3,3)`, `cadtge(lon_rad, lat_rad) -> (3,3)`, `cadsph(sbie) -> (lon, lat, alt)` with CADAC longitude quadrant logic from `cadsph` in `utility_functions.cpp` (copy all four quadrant `if`s plus the remaining C++ branches)

- [ ] **Step 1:** `cadtei(0)` is identity. `cadsph` of `[REARTH, 0, 0]` → lon=0, lat=0, alt=0. `cadtge(0,0)` matches C++ assignments.
- [ ] **Step 2–5:** port functions, pass, commit `feat: CADAC spherical-earth frame transforms`

---

### Task 16: Named state store

**Files:**
- Create: `Python/src/cadac/kernel/state.py`
- Create: `Python/tests/unit/test_state.py`

**Interfaces:**
- Produces: `Field(name, value, type="real"|"int"|"vec"|"mat", role, module, outputs=())`. `StateStore.define(field)`, `.get(name)`, `.set(name, value)` (int values stored as int), `.names()`. Duplicate `define` same name raises `ValueError`. Unknown `set`/`get` raises `KeyError`. Vec must be shape (3,), mat (3,3).

- [ ] **Step 1:** tests for define/get/set, duplicate error, unknown error, int vs real
- [ ] **Step 2–5:** implement, pass, commit `feat: named CADAC module-variable store`

---

### Task 17: Event engine

**Files:**
- Create: `Python/src/cadac/kernel/events.py`
- Create: `Python/tests/unit/test_events.py`

**Interfaces:**
- Consumes: `StateStore`
- Produces: `@dataclass EventSpec: when: dict, set: dict`. `EventEngine(events: list[EventSpec])` with `.evaluate(store) -> bool` (event_epoch). One event armed. Predicate `when={"time": {">": 10}}` reads `store.get("time")`, or `when={"var": "wp_flag", "op": "=", "value": -1}`. On fire: `store.set` each key in `set`, advance index. After last event, evaluate is False forever.

- [ ] **Step 1:**

```python
from cadac.kernel.state import Field, StateStore
from cadac.kernel.events import EventEngine, EventSpec

def test_time_then_set():
    s = StateStore()
    s.define(Field("time", 0.0, "real", "exec", "environment"))
    s.define(Field("mprop", 1, "int", "data", "propulsion"))
    eng = EventEngine([EventSpec(when={"time": {">": 10}}, set={"mprop": 2})])
    s.set("time", 10.0)
    assert eng.evaluate(s) is False
    s.set("time", 10.01)
    assert eng.evaluate(s) is True
    assert s.get("mprop") == 2
    s.set("time", 11.0)
    assert eng.evaluate(s) is False
```

- [ ] **Step 2–5:** implement, pass, commit `feat: CADAC sequential event engine`

---

### Task 18: Executive loop with dummy module

**Files:**
- Create: `Python/src/cadac/kernel/module.py`
- Create: `Python/src/cadac/kernel/executive.py`
- Create: `Python/tests/unit/test_executive.py`

**Interfaces:**
- Consumes: events, state
- Produces:
  - `SimContext` dataclass: `sim_time, int_step, event_time, out_fact, combus, vehicle_slot`
  - `Module` protocol `name`, `define`, `initialize`, `execute`, `terminate`
  - `run_loop(vehicles, modules_by_vehicle, module_order, end_time, int_step) -> list[float]` of sim_time at each completed step
  - Loop: `while sim_time <= end_time + int_step`; per vehicle: events; if health==1 execute modules in `module_order`; `event_time += int_step`; `sim_time += int_step`
  - Dummy module increments `store.time` to `ctx.sim_time`

- [ ] **Step 1:** one dummy vehicle, `end_time=0.2`, `int_step=0.1` → times include 0 (init separate — loop starts at 0 or after first increment per CADAC: CADAC initialize then loop with sim_time starting 0, modules run, then sim_time += int_step, while `sim_time <= end_time+int_step`. Match `execution.cpp`: sim_time starts 0, loop condition `while (sim_time<=(end_time+int_step))`, advance at end. Test: collect sim_time at start of each iteration = `[0.0, 0.1, 0.2, 0.3]` for end_time=0.2, int_step=0.1.
- [ ] **Step 2–5:** implement, pass, commit `feat: CADAC executive time loop`

---

### Task 19: Combus packet

**Files:**
- Create: `Python/src/cadac/kernel/combus.py`
- Create: `Python/tests/unit/test_combus.py`
- Modify: executive to publish after modules if vehicle has `com_names`

**Interfaces:**
- Produces: `@dataclass Packet: name, type, status: int, vars: dict`. `packet_from_store(store, names) -> Packet`. Status 1/0/-1.

- [ ] **Step 1:** build packet with two named vars
- [ ] **Step 2–5:** implement, pass, commit `feat: combus packets`

---

### Task 20: Scenario JSONC schema

**Files:**
- Create: `Python/src/cadac/io/scenario.py`
- Create: `Python/tests/unit/test_scenario.py`

**Interfaces:**
- Consumes: jsonc
- Produces: dataclasses `RunConfig(title, options: dict[str,bool], modules: list[ModuleSpec], timing: dict[str,float], end_time: float, vehicles: list[VehicleSpec])`. `VehicleSpec(type, name, aero_deck: Path|None, prop_deck: Path|None, params: dict, events: list[EventSpec])`. `ModuleSpec(name, phases: list[str])`. `load_scenario(path) -> RunConfig`. Relative deck paths resolved against scenario parent. Unknown option keys ignored only if you document them — spec says known keys; extra keys in `params` stay in the dict (vehicle validate later). `options` omitted flags False.

- [ ] **Step 1:** load a minimal JSONC string/file with one CRUISE3 vehicle and one time event
- [ ] **Step 2–5:** implement, pass, commit `feat: load CADAC scenario JSONC`

---

### Task 21: Translate input.asc (no events)

**Files:**
- Modify: `Python/src/cadac/io/translate.py` add `translate_scenario_asc(src, dst_dir)`
- Create: `Python/tests/translate/test_asc_scenario.py`

**Interfaces:**
- Consumes: CADAC `input_climb.asc` structure: TITLE, OPTIONS, MODULES…END, TIMING…END, VEHICLES n, type name, params, END, ENDTIME, STOP
- Produces: JSONC with title, options (y_* → true, n_* → false), modules+phases, timing, end_time, vehicles[0].type `CRUISE3`, params lonx/latx/alt/… AERO_DECK/PROP_DECK become `aero_deck`/`prop_deck` filenames with `.jsonc` suffix (do not rewrite files in this task — path strings only)

- [ ] **Step 1:** parse `CADAC_Simulations/HYPER3_250114/HYPER3/input_climb.asc` → `type=="CRUISE3"`, `params["lonx"]==-80.55`, `end_time==90`, first module name `environment` with phases including `init`
- [ ] **Step 2–5:** implement, pass, commit `feat: translate CADAC input.asc header and params`

---

### Task 22: Translate IF/ENDIF events

**Files:**
- Modify: `Python/src/cadac/io/translate.py` / `asc_scenario.py`
- Create: `Python/tests/translate/test_asc_events.py`

**Interfaces:**
- Produces: nested IF blocks become **ordered** `events` list. `IF time > 10` → `when: {time: {">": 10}}` with `set` of assignments until ENDIF. Sequential IFs in `input_climb.asc` → two events (time>10 and time>50).

- [ ] **Step 1:** `input_climb.asc` yields two events; first set contains `mprop==2` and `qhold==50000`
- [ ] **Step 2–5:** implement, pass, commit `feat: translate CADAC IF/ENDIF events to JSONC`

---

### Task 23: Plot CSV writer

**Files:**
- Create: `Python/src/cadac/io/plot.py`
- Create: `Python/tests/unit/test_plot_csv.py`

**Interfaces:**
- Produces: `write_plot_csv(path, title: str, columns: list[str], rows: list[list[float]])` — header line 1 title, line 2 ignored or `0 0 N`, line 3 `col,col,`, then data rows comma-separated (CADAC `plot1.csv` shape)

- [ ] **Step 1:** round-trip two rows, parse back columns `time,alt`
- [ ] **Step 2–5:** implement, pass, commit `feat: write CADAC-style plot.csv`

---

### Task 24: Round3 environment module

**Files:**
- Create: `Python/src/cadac/eom/round3.py` (environment only)
- Create: `Python/tests/unit/test_round3_environment.py`

**Interfaces:**
- Consumes: iso62, gravity, StateStore
- Produces: class `Round3Environment` with `name="environment"`. `define` registers spec fields: `time`, `event_time`, `int_step_new`, `out_step_fact`, `grav`, `rho`, `pdynmc`, `mach`, `vsound`, `press`. `initialize` sets time=sim_time, int_step_new=int_step. `execute` reads `alt`, `dvbe`; writes ISO62 outputs, `grav=gravity(alt)`, copies `ctx.sim_time` to `time`, sets `ctx.int_step` from `int_step_new`, `ctx.out_fact` from `out_step_fact`.

- [ ] **Step 1:** store alt=3000, dvbe=250; after execute, mach/pdynmc match `iso62(3000,250)` and grav matches `gravity(3000)`
- [ ] **Step 2–5:** implement, pass, commit `feat: Round3 environment module`

---

### Task 25: Round3 newton initialize

**Files:**
- Modify: `Python/src/cadac/eom/round3.py`
- Create: `Python/tests/unit/test_round3_newton_init.py`

**Interfaces:**
- Produces: `Round3Newton` `name="newton"`. `define` fields from spec/C++ `def_newton` (lonx, latx, alt, dvbe, psivgx, thtvgx, sbii, vbii, abii, sbeg, vbeg, tgv, tig, tge, weii, …). `initialize` ports `Round3::init_newton` in `round3_modules.cpp` (lon/lat/alt → SBII, heading/FPA/speed → VBEG/VBII, WEII skew-sym with WEII3).

- [ ] **Step 1:** params lonx=-80.55, latx=28.43, alt=3000, psivgx=90, thtvgx=0, dvbe=250. After init, `alt==3000`, `dvbe` ≈ 250, `abs(cadsph(sbii)[2] - 3000) < 1e-6` (meters).
- [ ] **Step 2–5:** port init_newton, pass, commit `feat: Round3 newton initialization`

---

### Task 26: Round3 newton step

**Files:**
- Modify: `Python/src/cadac/eom/round3.py`
- Create: `Python/tests/unit/test_round3_newton_step.py`

**Interfaces:**
- Produces: `execute` ports `Round3::newton` exactly: `abii_new = TIG @ ((TGV @ fspv) + grav_vec)`; integrate vbii then sbii; update TGE/TGI/VBEG/TVG as C++. `fspv` from store name `FSPV`, `grav` from store.

- [ ] **Step 1:** after init (Task 25 ICs), zero FSPV, one step dt=0.01; `alt` remains finite; `sbii` changes; assert `integrate` order by checking vbii used new accel (compare to a one-liner replica in the test)
- [ ] **Step 2–5:** port newton(), pass, commit `feat: Round3 newton integration step`

---

### Task 27: Cruise3 aerodynamics

**Files:**
- Create: `Python/src/cadac/vehicles/cruise3/aero.py`
- Create: `Python/tests/unit/test_cruise3_aero.py`

**Interfaces:**
- Consumes: Datadeck look_up
- Produces: `Cruise3Aero(deck)`. `execute`: `cl=cla0+cla*alphax`, `cd=cd0+ckk*(cl-cl0)**2`, `cl_ov_cd=cl/cd` with table names `cd0_vs_mach`, `cl0_vs_mach`, `cla_vs_mach`, `ckk_vs_mach`, `cla0_vs_mach`. Define alphax, area, cl, cd, cla, cl_ov_cd.

- [ ] **Step 1:** load translated or JSONC from parsed `ghame3_aero_deck.asc` via Tasks 5–9; mach=0.760854, alphax=7; compute expected with look_up then the two formulas; assert store matches
- [ ] **Step 2–5:** implement, pass, commit `feat: Cruise3 drag-polar aerodynamics`

---

### Task 28: Cruise3 propulsion mprop 0 and 1

**Files:**
- Create: `Python/src/cadac/vehicles/cruise3/propulsion.py`
- Create: `Python/tests/unit/test_cruise3_prop_fixed.py`

**Interfaces:**
- Produces: `mprop==0` → thrust=0, fmassd=0. `mprop==1` → `spi=look_up("spi_vs_throttle_mach", throttle, mach)`, `ca=look_up("ca_vs_alpha_mach", alphax, mach)`, `thrust=spi*0.029*throttle*AGRAV*rho*dvbe*ca*acowl`. Fuel: `fmassd_next=thrust/(spi*AGRAV)` if spi!=0; `fmasse=integrate(...)`; `mass=mass0-fmasse`; `fmassr=fmass0-fmasse`; if fmassr<=0 then mprop=0.

- [ ] **Step 1:** unit test mprop=0 and mprop=1 with numeric rho, dvbe, throttle=0.2 from climb IC (mass0=136077, fmasse=0)
- [ ] **Step 2–5:** implement, pass, commit `feat: Cruise3 fixed-throttle propulsion`

---

### Task 29: Cruise3 autothrottle mprop 2

**Files:**
- Modify: `Python/src/cadac/vehicles/cruise3/propulsion.py`
- Create: `Python/tests/unit/test_cruise3_prop_auto.py`

**Interfaces:**
- Produces: C++ `mprop==2` block from `cruise_modules.cpp` (thrst_req, throtl_req, gainq, ethrotl, idle/max limiters, then spi look_up and thrust)

- [ ] **Step 1:** fixture numbers: copy one C++ path with qhold=50000, tq=1, alphax=2.5, known rho/pdynmc/cd/area; assert throttle clipped and thrust formula
- [ ] **Step 2–5:** implement, pass, commit `feat: Cruise3 autothrottle propulsion`

---

### Task 30: Cruise3 forces

**Files:**
- Create: `Python/src/cadac/vehicles/cruise3/forces.py`
- Create: `Python/tests/unit/test_cruise3_forces.py`

**Interfaces:**
- Produces: FSPV components as `Cruise::forces` (spec / cruise_modules.cpp)

- [ ] **Step 1:**

```python
import numpy as np
from cadac.constants import RAD

def test_fspv_level():
    pdynmc, area, cd, cl, thrust, mass, alphax, phimvx = 28410.0, 557.42, 0.05, 0.2, 239241.0, 136077.0, 7.0, 0.0
    alpha = alphax * RAD
    phimv = phimvx * RAD
    f1 = (-pdynmc * area * cd + thrust * np.cos(alpha)) / mass
    f2 = np.sin(phimv) * (pdynmc * area * cl + thrust * np.sin(alpha)) / mass
    f3 = -np.cos(phimv) * (pdynmc * area * cl + thrust * np.sin(alpha)) / mass
    # after execute, store FSPV equals [f1,f2,f3]
```

Wire the test through `Cruise3Forces.execute` on a StateStore.

- [ ] **Step 2–5:** implement, pass, commit `feat: Cruise3 specific-force module`

---

### Task 31: CRUISE3 vehicle + run_scenario

**Files:**
- Create: `Python/src/cadac/vehicles/cruise3/vehicle.py`
- Create: `Python/src/cadac/cli.py` (`run_scenario(path)`)
- Modify: `Python/src/cadac/__init__.py` export `run_scenario`
- Create: `Python/tests/unit/test_cruise3_one_step.py`
- Create: `Python/cases/hyper3/` JSONC+decks via translator

**Interfaces:**
- Produces: `Cruise3` registers modules environment, aerodynamics, propulsion, forces, newton. `run_scenario` loads JSONC, builds CRUISE3 only (raise on other types), runs executive, returns object with `.plot_rows` for plot-flagged names. Translate `input_climb.asc` + decks into `Python/cases/hyper3/`.

- [ ] **Step 1:** run 1.0 s of climb case; `time` reaches 1.0; `mass < mass0`; `alt` > 2999
- [ ] **Step 2–5:** wire vehicle, pass, commit `feat: run CRUISE3 scenario from JSONC`

---

### Task 32: HYPER3 e2e vs plot1.csv

**Files:**
- Create: `Python/tests/e2e/test_hyper3_climb.py`
- Create: `Python/tests/e2e/goldens/hyper3/plot1.csv` (copy from `CADAC_Simulations/HYPER3_250114/HYPER3/plot1.csv`)
- Modify: plot writer / run_scenario to emit the same column set as golden: `time,FSPV1,FSPV2,FSPV3,pdynmc,mach,lonx,latx,alt,dvbe,psivgx,thtvgx,SBEG1,SBEG2,SBEG3,VBEG1,VBEG2,VBEG3,throttle,mass,thrust,fmassr,cl_ov_cd`

**Interfaces:**
- Consumes: full climb run to end_time 90, plot_step 0.2
- Produces: pytest compares overlapping times with spec CSV tolerances

- [ ] **Step 1:** failing e2e that loads golden and compares `alt` at t=0 and t=0.2
- [ ] **Step 2: FAIL** (until run matches)
- [ ] **Step 3: Fix plot flags / column order / init outputs so t=0 row matches CADAC (CADAC writes after init). Compare all columns on the plot grid.**
- [ ] **Step 4: PASS** `pytest tests/e2e/test_hyper3_climb.py -v`
- [ ] **Step 5: Commit** `test: HYPER3 climb e2e against CADAC plot1.csv`

---

## Self-review

- Spec coverage: JSONC, decks, look_up 1D/2D, integrate, US76, ISO62, gravity, frames, earth, state, events, executive, combus, ASC translate, plot CSV, Round3, Cruise3, e2e. 3D look_up deferred to FALCON6 plan. mat3tr deferred to FALCON6 plan.
- No TBD in task steps.
- Signatures: `look_up`, `integrate`, `run_scenario`, `load_scenario`, `EventEngine.evaluate` consistent across tasks.
