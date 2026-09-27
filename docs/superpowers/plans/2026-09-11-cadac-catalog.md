# CADAC Zipfel catalog (ASC → JSONC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. After **each** task: Grok non-fast reviewer (`cursor-grok-4.6-high`). After the last task: whole-plan reviewer.

**Goal:** Translate every Zipfel CADAC scenario and table deck `.asc` under `CADAC_Simulations/` into missing JSONC files under `Python/cases/<program>/`, with a machine-readable report, without overwriting committed cases or editing C++.

**Architecture:** `cadac.io.catalog` walks `PROGRAM_DIRS` plus two extra folders, classifies each `.asc` (skip / scenario / deck), and calls existing `translate_scenario_asc` / `deck_asc_to_jsonc`. Family stamps match current translate tests. Re-run is idempotent.

**Tech Stack:** Python >= 3.11, pytest. No new runtime deps. pytest `pythonpath` is `["src", "tools"]`. Run from `Python/`.

**Spec:** `docs/superpowers/specs/2026-09-11-cadac-workbench-ui-design.md` (Catalog section). This is **plan 1 of 5**. Next: `docs/superpowers/plans/2026-09-11-cadac-workbench.md`.

## Global Constraints

- Do not edit `CADAC_Simulations/` (no `.cpp` / `.hpp` / `.asc`)
- Do not overwrite existing `Python/cases/**/*.jsonc`
- Do not change `run_scenario` numerics or harvested goldens
- Skip stems: `readme`, `documentation`, `doc`, `input_copy` (case-insensitive)
- Family: AIM5 `aim5`, CRUISE5 `cruise5`, MAGSIX `magsix`, ROCKET6 `rocket6`, SAM6 `sam6`, SRAAM6 `sraam6`, AGM6 `agm6`; others omit
- Extra sources: `CADAC_Simulations/HYPER6 Input Problems for Sec 10_4/` → `Python/cases/hyper6/`; `CADAC_Simulations/AGM6_250217/Additional input Files/` → `Python/cases/agm6/`
- Repo root for `PROGRAM_DIRS`: `Path(__file__).resolve().parents[4]` from `Python/src/cadac/io/catalog.py`
- Implementer + reviewer: `cursor-grok-4.6-high` (no Fast, no non-Grok)
- TDD: failing test → implement → pass. Reviewer must see the test command and output
- Parent does **not** `git commit` unless the user asked; still prepare the message in the Commit step
- Each task bumps `UPDATES.md` (newest on top). Start at `0.170.0` for Task 1, then `0.170.1`, …

## File map

- Create: `Python/src/cadac/io/catalog.py`
- Modify: `Python/src/cadac/io/translate.py` (export `parse_scenario_asc`)
- Modify: `Python/src/cadac/cli.py` (optional `cadac catalog` later only if Task 4 needs it)
- Test: `Python/tests/unit/test_catalog.py`
- Output: `Python/cases/<program>/*.jsonc` (new only), `Python/cases/catalog-report.json`
- Docs: `UPDATES.md`

---

### Task 1: Public scenario parse + classify skip / scenario / deck

**Files:**
- Modify: `Python/src/cadac/io/translate.py` — rename `_parse_scenario_asc` to `parse_scenario_asc` (keep `_parse_scenario_asc = parse_scenario_asc` alias so existing internal uses still work)
- Create: `Python/src/cadac/io/catalog.py`
- Test: `Python/tests/unit/test_catalog.py`

**Interfaces:**
- Consumes: `cadac.io.translate.parse_scenario_asc`, `cadac.io.translate.deck_asc_to_jsonc` / `parse_asc_deck`
- Produces:
  - `SKIP_STEMS: frozenset[str]` = `frozenset({"readme", "documentation", "doc", "input_copy"})`
  - `classify_asc(path: Path) -> str` returns `"skip"` | `"scenario"` | `"deck"` | `"fail"`
  - `parse_scenario_asc(src: Path) -> dict` with keys `title`, `options`, `modules`, `timing`, `end_time`, `vehicles`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
from cadac.io.catalog import SKIP_STEMS, classify_asc

ROOT = Path(__file__).resolve().parents[2].parent  # BodyNavigation
HYPER3 = ROOT / "CADAC_Simulations/HYPER3_250114/HYPER3"
SEC104 = ROOT / "CADAC_Simulations/HYPER6 Input Problems for Sec 10_4"

def test_skip_stems():
    assert "readme" in SKIP_STEMS
    assert classify_asc(HYPER3 / "readme.asc") == "skip"
    assert classify_asc(HYPER3 / "input_copy.asc") == "skip"

def test_scenario_climb():
    assert classify_asc(HYPER3 / "input_climb.asc") == "scenario"

def test_deck_aero():
    assert classify_asc(HYPER3 / "ghame3_aero_deck.asc") == "deck"

def test_sec104_is_scenario():
    assert classify_asc(SEC104 / "10_1_1_input_aero.asc") == "scenario"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Python && python -m pytest tests/unit/test_catalog.py::test_skip_stems tests/unit/test_catalog.py::test_scenario_climb tests/unit/test_catalog.py::test_deck_aero tests/unit/test_catalog.py::test_sec104_is_scenario -v`

Expected: FAIL import `cadac.io.catalog` or `classify_asc` not defined.

- [ ] **Step 3: Write minimal implementation**

`catalog.py`:

```python
from pathlib import Path
from cadac.io.asc_deck import parse_asc_deck
from cadac.io.translate import parse_scenario_asc

SKIP_STEMS = frozenset({"readme", "documentation", "doc", "input_copy"})

def classify_asc(path: Path) -> str:
    path = Path(path)
    if path.stem.lower() in SKIP_STEMS:
        return "skip"
    try:
        payload = parse_scenario_asc(path)
    except Exception:
        payload = None
    if payload is not None and (payload.get("modules") or payload.get("vehicles")):
        return "scenario"
    try:
        _title, tables = parse_asc_deck(path)
    except Exception:
        return "fail"
    return "deck" if tables else "fail"
```

In `translate.py`: `parse_scenario_asc =` former `_parse_scenario_asc`; keep name `_parse_scenario_asc` as alias.

- [ ] **Step 4: Run tests to verify they pass**

Run: same pytest command as Step 2.

Expected: PASS.

- [ ] **Step 5: Commit (prepare only)**

```bash
git add Python/src/cadac/io/catalog.py Python/src/cadac/io/translate.py Python/tests/unit/test_catalog.py UPDATES.md
git commit -m "$(cat <<'EOF'
Add ASC classifier for Zipfel catalog (skip / scenario / deck).

EOF
)"
```

Bump `UPDATES.md` `0.170.0`.

---

### Task 2: Destination paths, family stamp, never overwrite

**Files:**
- Modify: `Python/src/cadac/io/catalog.py`
- Modify: `Python/tests/unit/test_catalog.py`

**Interfaces:**
- Consumes: `cadac_cpp.extract_cpp.PROGRAM_DIRS`, `classify_asc`, `translate_scenario_asc`, `deck_asc_to_jsonc`
- Produces:
  - `PROGRAM_FAMILY: dict[str, str | None]` — keys HYPER3…AGM6; values as spec (None omitted)
  - `repo_root() -> Path`
  - `cases_dir(program: str) -> Path` → `repo_root() / "Python/cases" / program.lower()`
  - `dest_jsonc(program: str, src: Path) -> Path` → `cases_dir(program) / f"{src.stem}.jsonc"`
  - `family_for(program: str) -> str | None`
  - `translate_one(program: str, src: Path) -> str` returns `"written"` | `"exists"` | `"skipped"` | `"failed"` (does not write on exists)

- [ ] **Step 1: Write the failing tests**

```python
from cadac.io.catalog import dest_jsonc, family_for, translate_one, repo_root

def test_family_map():
    assert family_for("AGM6") == "agm6"
    assert family_for("HYPER3") is None
    assert family_for("ROCKET6") == "rocket6"

def test_dest_hyper6_sec104():
    root = repo_root()
    src = root / "CADAC_Simulations/HYPER6 Input Problems for Sec 10_4/10_1_1_input_aero.asc"
    dst = dest_jsonc("HYPER6", src)
    assert dst == root / "Python/cases/hyper6/10_1_1_input_aero.jsonc"

def test_translate_one_does_not_overwrite(tmp_path, monkeypatch):
    # monkeypatch cases_dir to tmp; copy existing climb jsonc into dest; call translate_one
    ...
    assert translate_one("HYPER3", src) == "exists"
    assert dest.read_text() == original
```

Fill the overwrite test with a tiny fake `input_climb.asc` path still classified as scenario, dest pre-created with `"keep-me"`.

- [ ] **Step 2: Run tests — expect FAIL** (`family_for` missing)

Run: `cd Python && python -m pytest tests/unit/test_catalog.py::test_family_map tests/unit/test_catalog.py::test_dest_hyper6_sec104 tests/unit/test_catalog.py::test_translate_one_does_not_overwrite -v`

- [ ] **Step 3: Implement**

```python
from cadac_cpp.extract_cpp import PROGRAM_DIRS  # pytest pythonpath includes tools

PROGRAM_FAMILY = {
    "AIM5": "aim5", "CRUISE5": "cruise5", "MAGSIX": "magsix",
    "ROCKET6": "rocket6", "SAM6": "sam6", "SRAAM6": "sraam6", "AGM6": "agm6",
}

def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]

def cases_dir(program: str) -> Path:
    return repo_root() / "Python" / "cases" / program.lower()

def dest_jsonc(program: str, src: Path) -> Path:
    return cases_dir(program) / f"{src.stem}.jsonc"

def family_for(program: str) -> str | None:
    return PROGRAM_FAMILY.get(program)

def translate_one(program: str, src: Path) -> str:
    kind = classify_asc(src)
    if kind == "skip":
        return "skipped"
    if kind == "fail":
        return "failed"
    dst = dest_jsonc(program, src)
    if dst.exists():
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "scenario":
            translate_scenario_asc(src, dst.parent, family=family_for(program))
            # translate_scenario_asc writes dst.parent / f"{src.stem}.jsonc"
        else:
            deck_asc_to_jsonc(src, dst)
    except Exception:
        if dst.exists():
            dst.unlink()
        return "failed"
    return "written"
```

`translate_scenario_asc` already writes `{stem}.jsonc` into `dst_dir`. Do not pass a file path as `dst_dir`.

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Commit prepare** `0.170.1` — catalog dest/family/no-overwrite.

---

### Task 3: Walk sources + catalog-report.json

**Files:**
- Modify: `Python/src/cadac/io/catalog.py`
- Modify: `Python/tests/unit/test_catalog.py`

**Interfaces:**
- Consumes: `PROGRAM_DIRS`, `translate_one`, `classify_asc`
- Produces:
  - `EXTRA_SOURCES: list[tuple[str, str]]` = `[("HYPER6", "CADAC_Simulations/HYPER6 Input Problems for Sec 10_4"), ("AGM6", "CADAC_Simulations/AGM6_250217/Additional input Files")]`
  - `iter_asc_jobs() -> list[tuple[str, Path]]` — `(program, src)` unique by dest stem per program
  - `run_catalog() -> dict` with keys `translated`, `skipped`, `failed` where `failed` is `list[{"path": str, "error": str}]` and the other two are lists of posix relative-to-repo paths
  - `write_report(report: dict, path: Path | None = None) -> Path` default `Python/cases/catalog-report.json`

- [ ] **Step 1: Failing tests**

```python
from cadac.io.catalog import EXTRA_SOURCES, iter_asc_jobs, run_catalog

def test_iter_includes_sec104_and_agm_extra():
    jobs = iter_asc_jobs()
    stems = {(p, s.name) for p, s in jobs}
    assert ("HYPER6", "10_1_1_input_aero.asc") in stems
    assert ("AGM6", "input_3_2 AGM6 Free Flight.asc") in stems or any(
        p == "AGM6" and "Free Flight" in s.name for p, s in jobs
    )

def test_iter_excludes_readme():
    assert all(s.stem.lower() not in {"readme", "doc", "documentation", "input_copy"} or True
               for _p, s in iter_asc_jobs())
    assert not any(s.stem.lower() == "readme" for _p, s in iter_asc_jobs())

def test_run_catalog_writes_missing_sec104(tmp_path, monkeypatch):
    # If 10_1_1 jsonc missing in real cases dir, run_catalog translated list contains it.
    report = run_catalog()
    assert "failed" in report and "translated" in report and "skipped" in report
    dest = repo_root() / "Python/cases/hyper6/10_1_1_input_aero.jsonc"
    assert dest.is_file()
    # existing e2e case still present
    assert (repo_root() / "Python/cases/hyper3/input_climb.jsonc").is_file()
```

Do **not** mock away the real cases dir for `test_run_catalog_writes_missing_sec104` — writing the real missing JSONC is the deliverable. `run_catalog` must not delete or rewrite `input_climb.jsonc`.

- [ ] **Step 2: pytest those three — FAIL** (`run_catalog` missing)

- [ ] **Step 3: Implement `iter_asc_jobs` and `run_catalog`**

Walk `PROGRAM_DIRS.items()` then `EXTRA_SOURCES`. For each directory that exists, glob `*.asc`. Skip jobs whose `classify_asc == "skip"` from the job list (they appear in report `skipped` during run). `run_catalog` calls `translate_one`; on `"failed"` append `{"path": str(src), "error": traceback or classify}`. Capture exception message in `translate_one` (change return to keep error — or `translate_one` returns tuple). Prefer:

```python
def translate_one(program: str, src: Path) -> tuple[str, str | None]:
    ...
    return "failed", f"{type(exc).__name__}: {exc}"
```

Update Task 2 tests if the return type changes in this task (keep Task 2 tests passing: if they assert `== "exists"`, wrap: status, _err = translate_one(...); assert status == "exists"). **Update Task 2 tests in this task** to the tuple return so one source of truth exists.

- [ ] **Step 4: Tests pass.** Confirm `Python/cases/hyper6/10_1_1_input_aero.jsonc` exists and `load_scenario` does not raise:

```python
from cadac.io.scenario import load_scenario
load_scenario(repo_root() / "Python/cases/hyper6/10_1_1_input_aero.jsonc")
```

Add that as `test_sec104_load_scenario`.

- [ ] **Step 5: Commit prepare** `0.170.2` — walk Zipfel ASC and write missing JSONC + report.

---

### Task 4: CLI `python -m cadac.io.catalog` + all eight §10.4 files

**Files:**
- Create: `Python/src/cadac/io/__main__.py` (so `-m cadac.io.catalog` works — implement `__main__` on `catalog.py` via `if __name__` **and** `python -m cadac.io.catalog` requires `catalog.py` runnable)

Use package: `python -m cadac.io.catalog` → add `Python/src/cadac/io/catalog.py` `def main() -> int` + `if __name__ == "__main__"`. Running `python -m cadac.io.catalog` works if `cadac.io.catalog` has no relative-run issue; pytest pythonpath includes `src`.

- Modify: `Python/tests/unit/test_catalog.py`
- Modify: `Python/src/cadac/cli.py` only if you add `cadac catalog`; **prefer** `python -m cadac.io.catalog` as the spec says.

**Interfaces:**
- Produces: `main() -> int` writes report, prints counts, exit 0 even if some failed (failures are in the report). Exit 1 only if `iter_asc_jobs` finds zero directories.

- [ ] **Step 1: Failing tests**

```python
SEC104_STEMS = [
    "10_1_1_input_aero", "10_1_2_input_roll_doublet", "10_1_3_input_roll_doublet_freeze",
    "10_1_4_input_yaw_pitch", "10_1_5_input_altitude_heading", "10_2_2_input_gain_opt",
    "10_3_2_input_gamma_fan", "10_3_3_input_heading_fan",
]

def test_all_sec104_jsonc_exist_after_catalog():
    from cadac.io.catalog import run_catalog, repo_root
    run_catalog()
    cases = repo_root() / "Python/cases/hyper6"
    for stem in SEC104_STEMS:
        assert (cases / f"{stem}.jsonc").is_file(), stem

def test_main_writes_report(tmp_path, monkeypatch):
    from cadac.io import catalog
    rc = catalog.main()
    assert rc == 0
    report = repo_root() / "Python/cases/catalog-report.json"
    assert report.is_file()
    data = json.loads(report.read_text())
    assert set(data) >= {"translated", "skipped", "failed"}
```

- [ ] **Step 2: pytest — FAIL** (`main` missing)

- [ ] **Step 3: Implement `main`**, `write_report`, run `run_catalog()` for real so all eight files exist.

- [ ] **Step 4: Pass.** Also: `cd Python && python -m cadac.io.catalog` exit 0.

- [ ] **Step 5: Commit prepare** `0.170.3` — catalog CLI; HYPER6 §10.4 JSONC present.

Do not add an e2e `run_scenario` matrix for the new cases in this plan (types may be unregistered; spec: translate but Run may fail). `load_scenario` on §10.4 is enough.

---
