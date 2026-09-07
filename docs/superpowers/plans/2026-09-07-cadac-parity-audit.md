# CADAC C++ → Python parity audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inventory every CADAC++ vehicle/module/mode against Python, harvest Linux g++ plot goldens for the existing JSONC cases, run pytest e2e, and record leftover gaps without porting them.

**Architecture:** Tools live in `Python/tools/cadac_cpp/`. Phase A writes `inventory.json` (scripted extract + status heuristic). Phase B compiles each C++ program with a thin `-include` compat shim, runs the mapped `input*.asc` as `input.asc`, copies plot CSV into `Python/tests/e2e/goldens/`. Existing e2e tests compare `run_scenario` to those goldens. HYPER3 is the g++ canary against the already-committed golden.

**Tech Stack:** Python >= 3.11, pytest, numpy, g++ -std=c++17, make.

**Spec:** `docs/superpowers/specs/2026-09-07-cadac-parity-audit-design.md`

## Global Constraints

- Spec + parent `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
- Do not port stubbed/missing vehicles, modes, or I/O
- Do not rewrite CADAC++ math; compat shim only (`system("pause")` no-op)
- Do not loosen CSV tolerances (`rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`)
- Do not harvest live `rand()` / MONTE-on goldens; force `nmonte==0` if needed
- Do not e2e SAM6 RF (`test_sam6_rf.py`); leave skip-without-golden
- Do not git commit unless the user explicitly asked this session to commit
- Grok `cursor-grok-4.6-high` implementer + reviewer; TDD; no Fast / Composer / other models
- Work from `Python/` for pytest; C++ cwd is the CADAC program folder
- Add `Python/tools` to pytest `pythonpath`; do not put audit code in `cadac` runtime package

## File map

- `Python/tools/cadac_cpp/__init__.py`
- `Python/tools/cadac_cpp/schema.py` — `InventoryRow`, dump/load
- `Python/tools/cadac_cpp/harvest_table.py` — JSONC ↔ ASC ↔ golden paths
- `Python/tools/cadac_cpp/extract_cpp.py` — vehicle types, `def_*`, modes
- `Python/tools/cadac_cpp/extract_python.py` — registry, modules, modes
- `Python/tools/cadac_cpp/kernel_rows.py` — kernel/I/O inventory rows
- `Python/tools/cadac_cpp/inventory.py` — `scan_all` → `inventory.json`
- `Python/tools/cadac_cpp/compat.hpp` — Linux shim
- `Python/tools/cadac_cpp/Makefile.cadac` — parameterized g++ recipe
- `Python/tools/cadac_cpp/harvest.py` — backup/copy/run/restore/copy golden
- `Python/tools/cadac_cpp/inventory.json` — generated matrix
- `Python/tests/unit/test_cadac_cpp_harvest_table.py`
- `Python/tests/unit/test_cadac_cpp_schema.py`
- `Python/tests/unit/test_cadac_cpp_extract_cpp.py`
- `Python/tests/unit/test_cadac_cpp_extract_python.py`
- `Python/tests/unit/test_cadac_cpp_inventory.py`
- `Python/tests/unit/test_cadac_cpp_harvest.py`
- Goldens under `Python/tests/e2e/goldens/` (harvested, except HYPER3 already present)

---

### Task 1: Harvest mapping table

**Files:**
- Create: `Python/tools/cadac_cpp/__init__.py`
- Create: `Python/tools/cadac_cpp/harvest_table.py`
- Create: `Python/tests/unit/test_cadac_cpp_harvest_table.py`
- Modify: `Python/pyproject.toml` — `pythonpath = ["src", "tools"]`
- Modify: `.gitignore` — add `Python/tools/cadac_cpp/build/`

**Interfaces:**
- Consumes: spec harvest table
- Produces: `HARVEST_ROWS: list[HarvestRow]`; `HarvestRow(jsonc, cpp_dir, asc_name, golden)` with paths relative to repo root

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from cadac_cpp.harvest_table import HARVEST_ROWS, repo_root


def test_fourteen_harvest_rows():
    assert len(HARVEST_ROWS) == 14


def test_hyper3_canary_mapping():
    row = next(r for r in HARVEST_ROWS if "hyper3" in r.jsonc)
    root = repo_root()
    assert row.asc_name == "input_climb.asc"
    assert (root / row.cpp_dir / row.asc_name).is_file()
    assert row.golden == "Python/tests/e2e/goldens/hyper3/plot1.csv"


def test_hyper6_sat_type_not_in_harvest():
    assert all("RADAR" not in r.asc_name for r in HARVEST_ROWS)


def test_agm6_and_magsix_titles():
    names = {r.asc_name for r in HARVEST_ROWS}
    assert "input_3_1 AGM6 Free Flight.asc" in names
    assert "input_2_1 AGM6 Test Case.asc" in names
    assert "input_attitudeMR1.asc" in names
    assert "input_trajectoryMR1.asc" in names
    assert "input_Demo_4_7_pro_nav.asc" in names
    assert "input_insertion.asc" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_harvest_table.py -v`

Expected: FAIL with `ModuleNotFoundError: cadac_cpp` or `HARVEST_ROWS` missing

- [ ] **Step 3: Write minimal implementation**

`Python/pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "tools"]
```

`Python/tools/cadac_cpp/harvest_table.py`:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarvestRow:
    jsonc: str
    cpp_dir: str
    asc_name: str
    golden: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


HARVEST_ROWS = [
    HarvestRow(
        "Python/cases/hyper3/input_climb.jsonc",
        "CADAC_Simulations/HYPER3_250114/HYPER3",
        "input_climb.asc",
        "Python/tests/e2e/goldens/hyper3/plot1.csv",
    ),
    HarvestRow(
        "Python/cases/falcon5/input_turning_to_IP.jsonc",
        "CADAC_Simulations/FALCON5_250116/FALCON5",
        "input_turning_to_IP.asc",
        "Python/tests/e2e/goldens/falcon5/plot.csv",
    ),
    HarvestRow(
        "Python/cases/falcon6/input_gamma.jsonc",
        "CADAC_Simulations/FALCON6_250201/FALCON6",
        "input_gamma.asc",
        "Python/tests/e2e/goldens/falcon6/plot.csv",
    ),
    HarvestRow(
        "Python/cases/hyper5/input.jsonc",
        "CADAC_Simulations/HYPER5_250113/HYPER5",
        "input_Demo_4_7_pro_nav.asc",
        "Python/tests/e2e/goldens/hyper5/plot.csv",
    ),
    HarvestRow(
        "Python/cases/hyper6/input_climb.jsonc",
        "CADAC_Simulations/HYPER6_250125/HYPER6",
        "input_climb.asc",
        "Python/tests/e2e/goldens/hyper6/plot.csv",
    ),
    HarvestRow(
        "Python/cases/aim5/input_hori.jsonc",
        "CADAC_Simulations/AIM5_250114/AIM5",
        "input_hori.asc",
        "Python/tests/e2e/goldens/aim5/plot.csv",
    ),
    HarvestRow(
        "Python/cases/cruise5/input_1.jsonc",
        "CADAC_Simulations/CRUISE5_250115/CRUISE5",
        "input_1.asc",
        "Python/tests/e2e/goldens/cruise5/plot.csv",
    ),
    HarvestRow(
        "Python/cases/magsix/input.jsonc",
        "CADAC_Simulations/MAGSIX_231111/MAGSIX",
        "input_attitudeMR1.asc",
        "Python/tests/e2e/goldens/magsix/plot.csv",
    ),
    HarvestRow(
        "Python/cases/magsix/input_trajectoryMR1.jsonc",
        "CADAC_Simulations/MAGSIX_231111/MAGSIX",
        "input_trajectoryMR1.asc",
        "Python/tests/e2e/goldens/magsix/trajectory/plot.csv",
    ),
    HarvestRow(
        "Python/cases/rocket6/input.jsonc",
        "CADAC_Simulations/ROCKET6_250122/ROCKET6",
        "input_insertion.asc",
        "Python/tests/e2e/goldens/rocket6/plot.csv",
    ),
    HarvestRow(
        "Python/cases/sam6/input_SAM_autopilot.jsonc",
        "CADAC_Simulations/SAM6_250217/SAM6",
        "input_SAM_autopilot.asc",
        "Python/tests/e2e/goldens/sam6/plot.csv",
    ),
    HarvestRow(
        "Python/cases/sraam6/input_1v1.jsonc",
        "CADAC_Simulations/SRAAM6_250130/SRAAM6",
        "input_1v1.asc",
        "Python/tests/e2e/goldens/sraam6/plot.csv",
    ),
    HarvestRow(
        "Python/cases/agm6/input_freeflight.jsonc",
        "CADAC_Simulations/AGM6_250217/AGM6",
        "input_3_1 AGM6 Free Flight.asc",
        "Python/tests/e2e/goldens/agm6/plot.csv",
    ),
    HarvestRow(
        "Python/cases/agm6/input_testcase.jsonc",
        "CADAC_Simulations/AGM6_250217/AGM6",
        "input_2_1 AGM6 Test Case.asc",
        "Python/tests/e2e/goldens/agm6/test_case_plot.csv",
    ),
]
```

Empty `__init__.py`. Append `Python/tools/cadac_cpp/build/` to `.gitignore`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_harvest_table.py -v`

Expected: PASS

- [ ] **Step 5: Commit** (skip unless the user asked)

```bash
git add Python/pyproject.toml Python/tools/cadac_cpp .gitignore Python/tests/unit/test_cadac_cpp_harvest_table.py
git commit -m "test: CADAC harvest JSONC-to-ASC mapping table"
```

---

### Task 2: Inventory row schema

**Files:**
- Create: `Python/tools/cadac_cpp/schema.py`
- Create: `Python/tests/unit/test_cadac_cpp_schema.py`

**Interfaces:**
- Consumes: spec field list
- Produces: `InventoryRow(program, kind, name, cpp, python, status, note)`; `dump_inventory(path, rows)`; `load_inventory(path) -> list[InventoryRow]`; `KINDS`; `STATUSES`

- [ ] **Step 1: Write the failing test**

```python
from cadac_cpp.schema import InventoryRow, dump_inventory, load_inventory, KINDS, STATUSES


def test_roundtrip(tmp_path):
    row = InventoryRow(
        program="HYPER6",
        kind="vehicle",
        name="RADAR0",
        cpp="HYPER6/global_functions.cpp:set_obj_type",
        python=None,
        status="missing",
        note="not registered",
    )
    path = tmp_path / "inventory.json"
    dump_inventory(path, [row])
    got = load_inventory(path)
    assert got == [row]


def test_kinds_and_statuses():
    assert KINDS == ("vehicle", "module", "mode", "kernel", "e2e", "harvest")
    assert STATUSES == ("ported", "stubbed", "missing", "deferred", "diverged")


def test_rejects_bad_status():
    try:
        InventoryRow("HYPER3", "vehicle", "CRUISE3", "x", "y", "nope", "")
    except ValueError as exc:
        assert "status" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_schema.py -v`

Expected: FAIL import

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import asdict, dataclass
import json
from pathlib import Path

KINDS = ("vehicle", "module", "mode", "kernel", "e2e", "harvest")
STATUSES = ("ported", "stubbed", "missing", "deferred", "diverged")


@dataclass(frozen=True)
class InventoryRow:
    program: str
    kind: str
    name: str
    cpp: str
    python: str | None
    status: str
    note: str

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind {self.kind!r}")
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r}")


def dump_inventory(path: Path, rows: list[InventoryRow]) -> None:
    path.write_text(
        json.dumps([asdict(r) for r in rows], indent=2) + "\n",
        encoding="utf-8",
    )


def load_inventory(path: Path) -> list[InventoryRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [InventoryRow(**item) for item in data]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_schema.py -v`

Expected: PASS

- [ ] **Step 5: Commit** (skip unless the user asked)

---

### Task 3: Extract C++ vehicle types

**Files:**
- Create: `Python/tools/cadac_cpp/extract_cpp.py`
- Create: `Python/tests/unit/test_cadac_cpp_extract_cpp.py`

**Interfaces:**
- Consumes: C++ source text
- Produces: `extract_vehicle_types(text) -> list[str]` from `set_obj_type` bodies: `strcmp(temp,"TYPE")` / `!strcmp(temp,"TYPE")`; `PROGRAM_DIRS: dict[str, str]` mapping program name to CADAC folder relative to repo root

- [ ] **Step 1: Write the failing test**

```python
from cadac_cpp.extract_cpp import extract_vehicle_types, PROGRAM_DIRS


HYPER6_SNIPPET = r'''
Cadac *set_obj_type(fstream &input, Module *module_list, int num_modules)
{
	if (!strcmp(temp,"HYPER6"))
		obj=new Hyper(...);
	else if (!strcmp(temp,"SAT3"))
		obj=new Satellite(...);
	else if (!strcmp(temp,"RADAR0"))
		obj=new Radar(...);
	return obj;
}
'''


def test_hyper6_types_include_sat3_and_radar0():
    assert extract_vehicle_types(HYPER6_SNIPPET) == ["HYPER6", "SAT3", "RADAR0"]


def test_ignores_commented_strcmp():
    text = '// if(!strcmp(temp,"GHOST"))\nif(!strcmp(temp,"CRUISE3")) {}'
    assert extract_vehicle_types(text) == ["CRUISE3"]


def test_twelve_program_dirs():
    assert set(PROGRAM_DIRS) == {
        "HYPER3", "FALCON5", "FALCON6", "HYPER5", "HYPER6",
        "AIM5", "CRUISE5", "MAGSIX", "ROCKET6", "SAM6", "SRAAM6", "AGM6",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_extract_cpp.py::test_hyper6_types_include_sat3_and_radar0 -v`

Expected: FAIL import or empty list

- [ ] **Step 3: Write minimal implementation**

In `extract_cpp.py`:

```python
import re

PROGRAM_DIRS = {
    "HYPER3": "CADAC_Simulations/HYPER3_250114/HYPER3",
    "FALCON5": "CADAC_Simulations/FALCON5_250116/FALCON5",
    "FALCON6": "CADAC_Simulations/FALCON6_250201/FALCON6",
    "HYPER5": "CADAC_Simulations/HYPER5_250113/HYPER5",
    "HYPER6": "CADAC_Simulations/HYPER6_250125/HYPER6",
    "AIM5": "CADAC_Simulations/AIM5_250114/AIM5",
    "CRUISE5": "CADAC_Simulations/CRUISE5_250115/CRUISE5",
    "MAGSIX": "CADAC_Simulations/MAGSIX_231111/MAGSIX",
    "ROCKET6": "CADAC_Simulations/ROCKET6_250122/ROCKET6",
    "SAM6": "CADAC_Simulations/SAM6_250217/SAM6",
    "SRAAM6": "CADAC_Simulations/SRAAM6_250130/SRAAM6",
    "AGM6": "CADAC_Simulations/AGM6_250217/AGM6",
}

_STRCMP_TYPE = re.compile(
    r'^[ \t]*else[ \t]+if[ \t]*\([ \t]*!?strcmp[ \t]*\([ \t]*temp[ \t]*,[ \t]*"([^"]+)"[ \t]*\)'
    r'|^[ \t]*if[ \t]*\([ \t]*!?strcmp[ \t]*\([ \t]*temp[ \t]*,[ \t]*"([^"]+)"[ \t]*\)',
    re.MULTILINE,
)


def extract_vehicle_types(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("/*"):
            continue
        match = _STRCMP_TYPE.search(line)
        if match:
            name = match.group(1) or match.group(2)
            if name not in names:
                names.append(name)
    return names
```

Keep other extract functions unimplemented until later tasks (or stub `raise NotImplementedError` only if tests do not import them yet). Put only vehicle extraction in this file for now.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_extract_cpp.py -v`

Expected: PASS

- [ ] **Step 5: Commit** (skip unless the user asked)

---

### Task 4: Extract C++ modules and integer modes

**Files:**
- Modify: `Python/tools/cadac_cpp/extract_cpp.py`
- Modify: `Python/tests/unit/test_cadac_cpp_extract_cpp.py`

**Interfaces:**
- Consumes: C++ source text
- Produces: `extract_def_modules(text) -> list[str]` from `void Class::def_name(`; `extract_modes(text) -> list[tuple[str, int]]` of `(flag, value)` from executable `if`/`else if` comparing the spec flag list to integer literals; `MODE_FLAGS` frozenset from the spec

- [ ] **Step 1: Write the failing test** (append to `test_cadac_cpp_extract_cpp.py`)

```python
from cadac_cpp.extract_cpp import extract_def_modules, extract_modes


def test_def_modules():
    text = """
void Cruise::def_aerodynamics() {}
void Cruise::def_propulsion() {}
void Round3::def_newton() {}
// void Cruise::def_ghost() {}
"""
    assert extract_def_modules(text) == ["aerodynamics", "propulsion", "newton"]


def test_modes_from_if_not_comments():
    text = """
	if(mprop==1||mprop==2){
	if(mprop==0){
	// if(mprop==9){
	if(mins==0)
	else if(maut==24)
"""
    assert ("mprop", 1) in extract_modes(text)
    assert ("mprop", 2) in extract_modes(text)
    assert ("mprop", 0) in extract_modes(text)
    assert ("mins", 0) in extract_modes(text)
    assert ("maut", 24) in extract_modes(text)
    assert ("mprop", 9) not in extract_modes(text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_extract_cpp.py::test_def_modules tests/unit/test_cadac_cpp_extract_cpp.py::test_modes_from_if_not_comments -v`

Expected: FAIL `extract_def_modules` not defined

- [ ] **Step 3: Write minimal implementation**

```python
MODE_FLAGS = (
    "mguid", "mguidance", "mguide", "maut", "mauty", "mcontrol", "mprop",
    "mseek", "mseeker", "mins", "maero", "mact", "mtvc", "mair", "matmo",
    "mturb", "mwind", "mnav", "mterm", "mtrack", "mtarget", "skr_dyn",
    "skr_type", "minit", "mroll", "mrcs_force", "mrcs_moment", "mgps",
    "mstar", "mtargeting", "tgt_option", "acft_option", "guid_mid", "guid_term",
)

_DEF = re.compile(r"void\s+\w+::def_([A-Za-z0-9_]+)\s*\(")
_MODE = re.compile(
    r"\b(" + "|".join(MODE_FLAGS) + r")\s*==\s*(-?\d+)"
)


def extract_def_modules(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        match = _DEF.search(line)
        if match:
            name = match.group(1)
            if name not in names:
                names.append(name)
    return names


def extract_modes(text: str) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        for match in _MODE.finditer(line):
            pair = (match.group(1), int(match.group(2)))
            if pair not in found:
                found.append(pair)
    return found
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_extract_cpp.py -v`

Expected: PASS

- [ ] **Step 5: Commit** (skip unless the user asked)

---

### Task 5: Extract Python registry and modes

**Files:**
- Create: `Python/tools/cadac_cpp/extract_python.py`
- Create: `Python/tests/unit/test_cadac_cpp_extract_python.py`

**Interfaces:**
- Consumes: Python source text (not importing `cadac.cli` at collect time for the unit fixture)
- Produces: `extract_family_keys(text) -> list[tuple[str, str]]` from `("fam", "TYPE")`; `extract_global_types(text) -> list[str]` from `"TYPE": Class`; `extract_python_modes(text) -> tuple[list[tuple[str, int]], list[str]]` = implemented `(flag, value)` plus flags that have `unknown {flag}` ValueError stubs

- [ ] **Step 1: Write the failing test**

```python
from cadac_cpp.extract_python import (
    extract_family_keys,
    extract_global_types,
    extract_python_modes,
)

CLI_SNIPPET = '''
_VEHICLE_TYPES = {
    "CRUISE3": Cruise3,
    "HYPER6": Hyper6,
    "TARGET3": Target3,
}
_VEHICLE_FAMILIES: dict[tuple[str, str], type] = {
    ("aim5", "AIM5"): Aim5,
    ("rocket6", "HYPER6"): Rocket6,
}
'''

CONTROL_SNIPPET = '''
        if maut == 0:
            return
        if maut not in (24,):
            raise ValueError(f"unknown maut {maut}")
        if mauty == 2:
            pass
'''


def test_global_types():
    assert extract_global_types(CLI_SNIPPET) == ["CRUISE3", "HYPER6", "TARGET3"]


def test_family_keys():
    assert extract_family_keys(CLI_SNIPPET) == [("aim5", "AIM5"), ("rocket6", "HYPER6")]


def test_python_modes_implemented_and_stubbed():
    implemented, stub_flags = extract_python_modes(CONTROL_SNIPPET)
    assert ("maut", 0) in implemented
    assert ("maut", 24) in implemented
    assert ("mauty", 2) in implemented
    assert "maut" in stub_flags
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_extract_python.py -v`

Expected: FAIL import

- [ ] **Step 3: Write minimal implementation**

```python
import re

from cadac_cpp.extract_cpp import MODE_FLAGS

_TYPE_KEY = re.compile(r'"([A-Z][A-Z0-9]+)"\s*:')
_FAMILY = re.compile(r'\("([a-z0-9]+)",\s*"([A-Z][A-Z0-9]+)"\)')
_EQ = re.compile(r"\b(" + "|".join(MODE_FLAGS) + r")\s*==\s*(-?\d+)")
_IN_TUPLE = re.compile(
    r"\b(" + "|".join(MODE_FLAGS) + r")\s+not\s+in\s+\(([^)]*)\)"
    r"|\b(" + "|".join(MODE_FLAGS) + r")\s+in\s+\(([^)]*)\)"
)
_UNKNOWN = re.compile(r'unknown (' + "|".join(MODE_FLAGS) + r")")


def extract_global_types(text: str) -> list[str]:
    if "_VEHICLE_TYPES" not in text:
        return []
    start = text.index("_VEHICLE_TYPES")
    block = text[start : text.index("}", start) + 1]
    return _TYPE_KEY.findall(block)


def extract_family_keys(text: str) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for match in _FAMILY.finditer(text):
        pair = (match.group(1), match.group(2))
        if pair not in keys:
            keys.append(pair)
    return keys


def extract_python_modes(text: str) -> tuple[list[tuple[str, int]], list[str]]:
    implemented: list[tuple[str, int]] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        for match in _EQ.finditer(line):
            pair = (match.group(1), int(match.group(2)))
            if pair not in implemented:
                implemented.append(pair)
        for match in _IN_TUPLE.finditer(line):
            flag = match.group(1) or match.group(3)
            inner = match.group(2) or match.group(4)
            for part in inner.split(","):
                part = part.strip().rstrip(",")
                if not part:
                    continue
                implemented.append((flag, int(part)))
    stubs = []
    for match in _UNKNOWN.finditer(text):
        if match.group(1) not in stubs:
            stubs.append(match.group(1))
    return implemented, stubs
```

Deduplicate implemented pairs after the `in` tuple loop.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_extract_python.py -v`

Expected: PASS

- [ ] **Step 5: Commit** (skip unless the user asked)

---

### Task 6: Kernel rows and scan_all inventory

**Files:**
- Create: `Python/tools/cadac_cpp/kernel_rows.py`
- Create: `Python/tools/cadac_cpp/inventory.py`
- Create: `Python/tests/unit/test_cadac_cpp_inventory.py`

**Interfaces:**
- Consumes: `PROGRAM_DIRS`, extractors, `repo_root()`, `schema.InventoryRow`
- Produces: `kernel_rows() -> list[InventoryRow]`; `status_for_mode(cpp_pair, py_implemented, stub_flags)`; `scan_all(root) -> list[InventoryRow]`; CLI `python -m cadac_cpp.inventory` writes `Python/tools/cadac_cpp/inventory.json`

Status heuristic for modes: `ported` if `(flag, value)` in Python implemented; `stubbed` if flag in stub_flags; else `missing`. Vehicles: `ported` if type in `_VEHICLE_TYPES` or any family key; `missing` otherwise. HYPER6 `SAT3` is not Python `SATELLITE3` — do not alias. Modules: `ported` if `def_<name>` appears as a Python class `name=` or filename stem; else `missing`. Kernel statuses are fixed in `kernel_rows()`.

- [ ] **Step 1: Write the failing test**

```python
from cadac_cpp.inventory import status_for_mode, scan_all
from cadac_cpp.harvest_table import repo_root
from cadac_cpp.kernel_rows import kernel_rows


def test_status_for_mode():
    assert status_for_mode(("maut", 24), {("maut", 24)}, ["maut"]) == "ported"
    assert status_for_mode(("maut", 99), {("maut", 24)}, ["maut"]) == "stubbed"
    assert status_for_mode(("mguide", 5), set(), []) == "missing"


def test_kernel_includes_deferred_monte_and_ported_lookup():
    rows = {r.name: r for r in kernel_rows()}
    assert rows["look_up"].status == "ported"
    assert rows["nmonte"].status == "deferred"
    assert rows["scrn"].status == "deferred"


def test_scan_hyper6_radar_missing():
    rows = scan_all(repo_root())
    radar = [
        r for r in rows
        if r.program == "HYPER6" and r.kind == "vehicle" and r.name == "RADAR0"
    ]
    assert len(radar) == 1
    assert radar[0].status == "missing"
    sat3 = [
        r for r in rows
        if r.program == "HYPER6" and r.kind == "vehicle" and r.name == "SAT3"
    ]
    assert sat3[0].status == "missing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_inventory.py -v`

Expected: FAIL import

- [ ] **Step 3: Write minimal implementation**

`kernel_rows.py` returns rows for program `"kernel"`:

| name | status | note |
|---|---|---|
| events | ported | `cadac.kernel.events` |
| look_up | ported | `cadac.tables.lookup` |
| integrate | ported | CADAC stored-slope |
| combus | ported | `cadac.kernel.executive` |
| atmosphere76 | ported | |
| iso62 | ported | HYPER3 |
| GAUSS | ported | stores mean, no sample |
| RAYL | ported | stores mean |
| MARKOV | ported | stores 0 |
| markov_noise | deferred | live Monte Carlo |
| nmonte | deferred | |
| plot | ported | slot 0 only |
| csv | ported | |
| merge | deferred | CADAC Studio |
| scrn | deferred | |
| tabout | deferred | |
| doc | deferred | |
| traj | deferred | |
| comscrn | deferred | |

`inventory.py`:

- Read each `PROGRAM_DIRS` `*.cpp` / `*.hpp`
- Vehicle rows from `extract_vehicle_types` on `global_functions.cpp` (fallback: all files concatenated if `set_obj_type` lives elsewhere)
- Module rows from `extract_def_modules` on all sources
- Mode rows from `extract_modes`
- Python: read `src/cadac/cli.py` plus all `src/cadac/**/*.py`
- `status_for_mode`: `ported` if the `(flag, value)` pair is in Python implemented set; `stubbed` if the flag is in stub_flags; else `missing`
- Vehicle python path: `"cadac.cli:_VEHICLE_TYPES"` or `"cadac.cli:_VEHICLE_FAMILIES"` or `None`
- Append `kernel_rows()`
- `if __name__ == "__main__"`: `dump_inventory(Path(__file__).parent / "inventory.json", scan_all(repo_root()))`

Need `scan_all` to be deterministic: sort rows by `(program, kind, name)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_inventory.py -v`

Expected: PASS. Then run `cd Python && python -m cadac_cpp.inventory` and confirm `tools/cadac_cpp/inventory.json` exists with HYPER6 `RADAR0` `missing`.

- [ ] **Step 5: Human-pass notes**

Open `inventory.json`. For each `missing`/`stubbed` vehicle or mode that is a known slice limit (HYPER6 Radar/Satellite/Ground0, unused `maut` values), set `note` to `slice limit` vs `drop-out` only when the C++ ability was never given a Python class or ValueError. Do not change `status` from the heuristic except if the scanner aliased `SAT3` to `SATELLITE3` (must stay `missing` on HYPER6). Re-run the Task 6 tests.

- [ ] **Step 6: Commit** (skip unless the user asked)

---

### Task 7: Linux compat shim and HYPER3 g++ build

**Files:**
- Create: `Python/tools/cadac_cpp/compat.hpp`
- Create: `Python/tools/cadac_cpp/Makefile.cadac`
- Create: `Python/tests/unit/test_cadac_cpp_hyper3_build.py`

**Interfaces:**
- Consumes: HYPER3 `*.cpp` in `PROGRAM_DIRS["HYPER3"]`
- Produces: `Python/tools/cadac_cpp/build/HYPER3/hyper3` executable; `build_program(program: str) -> Path` in a small `cadac_cpp/build_cadac.py`

- [ ] **Step 1: Write the failing test**

```python
import shutil
from pathlib import Path

import pytest

from cadac_cpp.build_cadac import build_program


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ missing")
def test_hyper3_binary_exists():
    binary = build_program("HYPER3")
    assert binary.is_file()
    assert os.access(binary, os.X_OK)
```

Add `import os`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_hyper3_build.py -v`

Expected: FAIL import or compile error

- [ ] **Step 3: Write minimal implementation**

`compat.hpp`:

```cpp
#pragma once
static inline int cadac_system(const char *) { return 0; }
#define system cadac_system
```

`Makefile.cadac`:

```makefile
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -include $(COMPAT)
COMPAT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/compat.hpp)
CADAC_DIR ?=
PROGRAM ?=
BUILD ?=
SOURCES := $(wildcard $(CADAC_DIR)/*.cpp)
$(BUILD)/$(PROGRAM): $(SOURCES) $(COMPAT)
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(CADAC_DIR) -o $@ $(SOURCES)
```

`build_cadac.py`:

```python
import subprocess
from pathlib import Path

from cadac_cpp.extract_cpp import PROGRAM_DIRS
from cadac_cpp.harvest_table import repo_root


def build_program(program: str) -> Path:
    root = repo_root()
    tools = Path(__file__).resolve().parent
    build = tools / "build" / program
    build.mkdir(parents=True, exist_ok=True)
    binary = build / program.lower()
    subprocess.run(
        [
            "make", "-f", str(tools / "Makefile.cadac"),
            f"CADAC_DIR={root / PROGRAM_DIRS[program]}",
            f"PROGRAM={program.lower()}",
            f"BUILD={build}",
            str(binary),
        ],
        check=True,
    )
    return binary
```

If g++ errors on `int file_ptr=NULL`, add `-Wno-conversion -fpermissive` to `CXXFLAGS` in the Makefile only. Do not edit CADAC `.cpp` files.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_hyper3_build.py -v`

Expected: PASS, or skip if no g++. If compile fails after `-fpermissive`, record a `kind=harvest` row later; do not rewrite vehicle math.

- [ ] **Step 5: Commit** (skip unless the user asked)

---

### Task 8: HYPER3 canary harvest

**Files:**
- Create: `Python/tools/cadac_cpp/harvest.py`
- Create: `Python/tests/unit/test_cadac_cpp_harvest.py`

**Interfaces:**
- Consumes: `HARVEST_ROWS`, `build_program`
- Produces: `find_plot_csv(cwd: Path) -> Path` (`plot1.csv` then `plot.csv`); `force_monte_off(text) -> str` replacing `MONTE` on with off / `nmonte` 0 without touching other lines; `harvest_row(row, timeout_s=600) -> Path` backup `input.asc`, write mapped ASC (MONTE-off copy), run binary with cwd=`cpp_dir`, copy plot to golden path, restore `input.asc` in a `finally`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from cadac_cpp.harvest import find_plot_csv, force_monte_off, harvest_row
from cadac_cpp.harvest_table import HARVEST_ROWS, repo_root


def load_cadac_plot_csv(path):
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


def test_force_monte_off():
    text = "MONTE 1\nnmonte 5\n"
    out = force_monte_off(text)
    assert "MONTE 0" in out or "n_monte 0" in out or "nmonte 0" in out.lower()
    assert "MONTE 1" not in out


def test_find_plot_prefers_plot1(tmp_path):
    (tmp_path / "plot.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "plot1.csv").write_text("y\n", encoding="utf-8")
    assert find_plot_csv(tmp_path).name == "plot1.csv"


def test_hyper3_harvest_matches_checked_in_golden():
    row = next(r for r in HARVEST_ROWS if r.golden.endswith("hyper3/plot1.csv"))
    produced = harvest_row(row)
    golden = repo_root() / row.golden
    _, got_rows = load_cadac_plot_csv(produced)
    _, want_rows = load_cadac_plot_csv(golden)
    assert len(got_rows) == len(want_rows)
    for got, want in zip(got_rows, want_rows):
        np.testing.assert_allclose(
            got["alt"], want["alt"],
            rtol=1e-5, atol=max(1e-6, 5e-6 * abs(want["alt"])),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_harvest.py::test_find_plot_prefers_plot1 -v`

Expected: FAIL import

- [ ] **Step 3: Write minimal implementation**

`force_monte_off`: regex on tokens `MONTE` / `nmonte` / `NMONTE` integer following; set to `0`. CADAC often has `MONTE` in OPTIONS as `n_monte` — inspect HYPER3 `input_climb.asc` OPTIONS line and neutralize the monte flag actually present.

`harvest_row`:

```python
def harvest_row(row, timeout_s=600):
    root = repo_root()
    cwd = root / row.cpp_dir
    original = (cwd / "input.asc").read_bytes() if (cwd / "input.asc").exists() else None
    source = (cwd / row.asc_name).read_text(encoding="utf-8", errors="replace")
    (cwd / "input.asc").write_text(force_monte_off(source), encoding="utf-8")
    try:
        from cadac_cpp.extract_cpp import PROGRAM_DIRS
        from cadac_cpp.build_cadac import build_program
        key = next(k for k, v in PROGRAM_DIRS.items() if v == row.cpp_dir)
        binary = build_program(key)
        subprocess.run([str(binary)], cwd=cwd, check=True, timeout=timeout_s)
        plot = find_plot_csv(cwd)
        if row.golden.endswith("hyper3/plot1.csv"):
            dest = root / "Python/tests/e2e/goldens/hyper3/plot1.gpp.csv"
        else:
            dest = root / row.golden
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(plot.read_bytes())
        return dest
    finally:
        path = cwd / "input.asc"
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(original)
```

HYPER3 writes `plot1.gpp.csv` and must not overwrite the committed `plot1.csv`. Other families write `row.golden`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_harvest.py -v`

Expected: PASS (`alt` on `plot1.gpp.csv` matches committed `plot1.csv` within CSV tolerances). If it does not match, leave the committed golden, keep `plot1.gpp.csv`, fail this canary test, and continue Task 9 using Linux g++ goldens for the other families. Do not retune tolerances.

- [ ] **Step 5: Commit** (skip unless the user asked)

---

### Task 9: Build remaining programs and harvest goldens

**Files:**
- Modify: `Python/tools/cadac_cpp/harvest.py` — `harvest_all(skip_failed=True)`
- Modify: `Python/tests/unit/test_cadac_cpp_harvest.py`
- Create: goldens listed in the spec table (except HYPER3 already present)

**Interfaces:**
- Consumes: Task 7 Makefile (same recipe, different `CADAC_DIR`)
- Produces: goldens for every harvest row whose program built; `kind=harvest` inventory rows `ported` or `missing`

- [ ] **Step 1: Write the failing test**

```python
from cadac_cpp.harvest import harvest_all
from cadac_cpp.harvest_table import HARVEST_ROWS, repo_root


def test_harvest_all_writes_or_skips():
    results = harvest_all()
    assert set(results) == {r.golden for r in HARVEST_ROWS}
    root = repo_root()
    for golden, status in results.items():
        assert status in {"ok", "build_failed", "run_failed"}
        if status == "ok":
            assert (root / golden).is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_harvest.py::test_harvest_all_writes_or_skips -v`

Expected: FAIL `harvest_all` missing

- [ ] **Step 3: Write minimal implementation**

`harvest_all()`: for each `HARVEST_ROWS` item, try `build_program` then `harvest_row`. Catch `subprocess.CalledProcessError` → `build_failed` or `run_failed`. Never leave `input.asc` unrestored (already in `finally`). Timeouts: HYPER3 120s, CRUISE5 900s, ROCKET6 600s, others 300s.

Do not edit CADAC sources. If a program needs extra `-l` libraries, add them to `Makefile.cadac` `LDLIBS` only.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_harvest.py::test_harvest_all_writes_or_skips -v`

Expected: PASS. Some values may be `build_failed`; that is allowed.

- [ ] **Step 5: Commit** (skip unless the user asked)

---

### Task 10: Pytest e2e vs goldens and inventory e2e/harvest rows

**Files:**
- Modify: `Python/tools/cadac_cpp/inventory.py` — append harvest + e2e rows after scan
- Modify: `Python/tests/unit/test_cadac_cpp_inventory.py`
- Modify: `UPDATES.md` (new top entry, bump `subver`)
- Modify: `README.md` only if the architecture paragraph should mention the audit tools

**Interfaces:**
- Consumes: harvested goldens, existing `Python/tests/e2e/test_*.py`
- Produces: pytest results; `kind=e2e` rows `ported` or `diverged`; `kind=harvest` rows; UPDATES summary counts (`ported`/`stubbed`/`missing`/`deferred`/`diverged`)

- [ ] **Step 1: Write the failing test**

```python
from cadac_cpp.schema import load_inventory
from cadac_cpp.harvest_table import repo_root


def test_inventory_has_e2e_and_harvest_kinds():
    path = repo_root() / "Python/tools/cadac_cpp/inventory.json"
    rows = load_inventory(path)
    kinds = {r.kind for r in rows}
    assert "harvest" in kinds
    assert "e2e" in kinds
```

This test is expected to fail until Step 3 writes those rows. Do not add e2e rows before running pytest.

- [ ] **Step 2: Run live e2e**

Run: `cd Python && pytest tests/e2e -v --tb=short`

Record pass/fail/skip per file. `test_sam6_rf.py` must still skip without its golden. Families whose harvest was `build_failed` still skip. Families with goldens must not skip.

If a comparison fails, leave the failing test red. Do not change `RTOL`. Add `kind=e2e` `status=diverged` for that golden name.

- [ ] **Step 3: Write harvest/e2e rows into inventory.json**

After pytest, re-run or extend `scan_all` to append:

- one `harvest` row per `HARVEST_ROWS` entry (`ok` → `ported`, else `missing`, note = stderr snippet)
- one `e2e` row per e2e test file that has a JSONC case (`passed` → `ported`, `failed` → `diverged`, `skipped` → `missing`)

- [ ] **Step 4: Re-run unit + e2e**

Run: `cd Python && pytest tests/unit/test_cadac_cpp_inventory.py tests/unit/test_cadac_cpp_harvest.py tests/e2e -v --tb=short`

Expected: inventory unit PASS; e2e green or recorded `diverged`; SAM6 RF skipped.

- [ ] **Step 5: UPDATES.md**

Newest entry on top. Bump `subver` (feature: audit tools + goldens). Include counts from `inventory.json` by status. Mention HYPER6 `SAT3`/`RADAR0` missing if still true. Do not rewrite architecture unless README needs a one-line pointer to `Python/tools/cadac_cpp/`.

- [ ] **Step 6: Commit** (skip unless the user asked)

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|---|---|
| Vehicle types | 3, 6 |
| Modules `def_*` | 4, 6 |
| Integer modes, comments ignored | 4, 5, 6 |
| Kernel/I/O rows | 6 |
| Python registry + ValueError stubs | 5, 6 |
| HYPER6 Radar/SAT3 missing, no SAT3↔SATELLITE3 alias | 6 |
| `inventory.json` | 6, 10 |
| Harvest table 14 JSONC cases | 1, 9 |
| Linux shim, no CADAC rewrite | 7 |
| HYPER3 canary vs committed golden | 8 |
| g++ fail → skip harvest, record | 9, 10 |
| MONTE off | 8 |
| Existing e2e harness, same CSV tolerances | 10 |
| SAM6 RF skip | 10 |
| No mode ports, no tolerance loosening | Global |
| UPDATES.md, no extra gap markdown | 10 |
