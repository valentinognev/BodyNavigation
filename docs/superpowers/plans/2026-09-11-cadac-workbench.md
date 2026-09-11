# CADAC workbench (library, forms, run, plot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. After **each** task: Grok non-fast reviewer (`cursor-grok-4.6-high`). After the last task: whole-plan reviewer.

**Goal:** Local Vite/React + FastAPI workbench that lists Zipfel JSONC cases, edits the full scenario as forms + JSONC drawer, runs `cadac.run_scenario`, and plots the CSV in the browser.

**Architecture:** `workbench/api` FastAPI wraps `cadac` (no EOM copy). `workbench/web` matches MISDC chrome: start library → top bar + left nav + form + right plots + bottom drawer. Ports API **8001**, Vite **5174**. Handshake/Launch buttons are stubs until plan 5.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, pytest; Vite, React 18, TypeScript, Tailwind 3 (`darkMode: ["selector", ".dark"]`), Zustand, Vitest. Plot: a small SVG/canvas line chart (no Chart.js required; a `<svg>` polyline is enough).

**Spec:** `docs/superpowers/specs/2026-09-11-cadac-workbench-ui-design.md` (CADAC UI, Plots, Error handling). **Plan 2 of 5.** Requires plan 1 catalog JSONC. Next: `docs/superpowers/plans/2026-09-11-cadac-handshake.md`.

## Global Constraints

- Do not edit `CADAC_Simulations/`
- Do not change `run_scenario` numerics / goldens
- Workbench API does not reimplement EOM; it calls `cadac.run_scenario` / `load_scenario`
- CORS: allow `http://127.0.0.1:5173` and `http://127.0.0.1:5175` (siblings) plus `http://127.0.0.1:5174`
- Vite proxy `/catalog`, `/cases`, `/run` → `http://127.0.0.1:8001`
- Parse failure does not replace last-good scenario; Run disabled on parse error or in-flight
- Plot: vehicle slot 0 CSV columns; defaults `alt`,`mach`,`dvbe`,`alphax` if present else first four numeric after `time`; extra `latx` vs `lonx` when both exist
- No Playwright. Vitest for form ↔ JSONC. Pytest for API
- Implementer + reviewer: `cursor-grok-4.6-high`
- TDD. Parent does not commit unless asked; prepare the message
- `UPDATES.md` bumps continue `0.170.N` from plan 1

## File map

- Create: `workbench/api/cadac_web/__init__.py`, `app.py`, `paths.py`, `runs.py`
- Create: `workbench/api/pyproject.toml` (fastapi, uvicorn, pydantic; `cadac` editable from `../../Python`)
- Create: `workbench/api/tests/test_catalog_api.py`, `test_cases_api.py`, `test_run_api.py`
- Create: `workbench/web/` Vite app (`package.json`, `vite.config.ts`, `tailwind.config.js`, `src/`)
- Create: `workbench/start.sh`, `workbench/kill.sh`
- Modify: `.gitignore` (`.run/`, `workbench/web/node_modules/`, `workbench/api/.venv/`)
- Docs: `UPDATES.md`, `README.md` architecture workbench line (ports) if the spec paragraph is still “not built” after Task 8 — update to “scaffold + run/plot”

---

### Task 1: FastAPI catalog index

**Files:**
- Create: `workbench/api/cadac_web/paths.py`, `app.py`, `__init__.py`
- Create: `workbench/api/pyproject.toml`, `workbench/api/tests/conftest.py`, `test_catalog_api.py`

**Interfaces:**
- Consumes: `cadac.io.catalog.repo_root`, `cases_dir` / `Python/cases/<program>/*.jsonc`
- Produces:
  - `CASES_ROOT() -> Path` = `repo_root() / "Python" / "cases"`
  - `GET /catalog` → `{ "programs": [ { "id": "hyper3", "label": "HYPER3", "cases": [ { "stem": "input_climb", "title": "..." } ] } ] }`
  - Program order: HYPER3, FALCON5, FALCON6, HYPER5, HYPER6, AIM5, CRUISE5, MAGSIX, ROCKET6, SAM6, SRAAM6, AGM6 (skip empty folders)
  - `title` from JSONC `title` key via `cadac.io.jsonc.loads`; if load fails, omit that file (do not 500)

- [ ] **Step 1: Failing test**

```python
from fastapi.testclient import TestClient
from cadac_web.app import app

def test_catalog_lists_hyper3_climb():
    client = TestClient(app)
    r = client.get("/catalog")
    assert r.status_code == 200
    programs = {p["id"]: p for p in r.json()["programs"]}
    assert "hyper3" in programs
    stems = {c["stem"] for c in programs["hyper3"]["cases"]}
    assert "input_climb" in stems
```

`conftest.py` must put `workbench/api` on `sys.path`.

- [ ] **Step 2:** `cd workbench/api && python -m pytest tests/test_catalog_api.py::test_catalog_lists_hyper3_climb -v` — FAIL import

- [ ] **Step 3: Minimal FastAPI app** with CORSMiddleware origins `http://127.0.0.1:5173`, `:5174`, `:5175`. `pyproject.toml` depends on `../../Python` editable + fastapi + httpx (TestClient).

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit prepare** catalog GET.

---

### Task 2: GET/PUT case JSONC + validate via load_scenario

**Files:**
- Modify: `workbench/api/cadac_web/app.py`
- Test: `workbench/api/tests/test_cases_api.py`

**Interfaces:**
- Produces:
  - `GET /cases/{program}/{stem}` → `{ "ok": true, "path": "...", "scenario": <dict> }` or 404
  - `PUT /cases/{program}/{stem}` body `{ "scenario": dict }` writes pretty JSONC (indent 2, trailing newline). Reject path traversal (`..`, extra slashes). After write, `cadac.io.scenario.load_scenario(path)` must succeed or return 400 `{ "ok": false, "error": str }` **without** keeping a corrupt file (write to temp then replace only on success; if previous file existed, restore it)
  - `POST /cases/validate` body `{ "scenario": dict }` — write temp JSONC, `load_scenario`, return `{ "ok": true }` or `{ "ok": false, "error": str }` (unknown option key, missing type, unknown param). Does not persist

- [ ] **Step 1: Failing tests**

```python
def test_get_hyper3_climb():
    r = TestClient(app).get("/cases/hyper3/input_climb")
    assert r.status_code == 200
    assert r.json()["scenario"]["vehicles"][0]["type"] == "CRUISE3"

def test_validate_unknown_option():
    r = TestClient(app).post("/cases/validate", json={"scenario": {
        "title": "x", "options": {"nope": True}, "modules": [],
        "timing": {"int_step": 0.01}, "end_time": 1, "vehicles": []
    }})
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert "nope" in r.json()["error"]

def test_put_rejects_dotdot():
    r = TestClient(app).put("/cases/hyper3/../secret", json={"scenario": {}})
    assert r.status_code in (400, 404)
```

`load_scenario` requires `vehicles` and known option keys from `OPTION_KEYS` in `cadac.io.scenario`.

- [ ] **Step 2: pytest FAIL** (routes missing)

- [ ] **Step 3: Implement GET/PUT/validate.** Sanitize `program` and `stem` with `^[a-z0-9_]+$` / `^[A-Za-z0-9._ -]+$`.

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit prepare**

---

### Task 3: POST /run short scenario + plot rows

**Files:**
- Create: `workbench/api/cadac_web/runs.py`
- Modify: `workbench/api/cadac_web/app.py`
- Test: `workbench/api/tests/test_run_api.py`

**Interfaces:**
- Consumes: `cadac.run_scenario(path) -> RunResult` with `plot_rows: list[dict]`
- Produces:
  - `POST /run` body `{ "program": "hyper3", "stem": "input_climb", "end_time": 0.1 }` — copies case to a temp dir **or** patches `end_time` in a temp JSONC (do not mutate the library file). Calls `run_scenario(temp_path)`. Returns `{ "ok": true, "columns": [...], "rows": [ {col: number} ] }` from `RunResult.plot_rows`. On `ValueError` / unknown type: `{ "ok": false, "error": str }` HTTP 200 (like MISDC parse). Timeout default 120 s via `asyncio` wait in a thread; on timeout `{ "ok": false, "error": "timeout" }`
  - Optional `end_time` in body overrides scenario `end_time` for the temp copy only
  - Test uses `end_time: 0.05` (a few steps) on hyper3 climb

- [ ] **Step 1: Failing test**

```python
def test_run_hyper3_short():
    r = TestClient(app).post("/run", json={"program": "hyper3", "stem": "input_climb", "end_time": 0.05})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "time" in body["columns"]
    assert "alt" in body["columns"]
    assert len(body["rows"]) >= 1
    assert body["rows"][0]["time"] == 0 or body["rows"][0]["time"] >= 0

def test_run_unknown_stem():
    r = TestClient(app).post("/run", json={"program": "hyper3", "stem": "no_such"})
    assert r.json()["ok"] is False
```

- [ ] **Step 2: FAIL** (no /run)

- [ ] **Step 3: Implement temp copy + `run_scenario`.** Copy sibling deck files referenced by the JSONC into the temp dir (same filenames). `shutil.copytree` of the program case folder is acceptable if it only copies `*.jsonc` decks the scenario needs — simplest: copy the whole `Python/cases/hyper3/` jsonc files into temp (they’re small).

- [ ] **Step 4: PASS** (may take ~10–30 s)

- [ ] **Step 5: Commit prepare**

---

### Task 4: Run cancel

**Files:**
- Modify: `workbench/api/cadac_web/runs.py`, `app.py`
- Test: `workbench/api/tests/test_run_api.py`

**Interfaces:**
- Produces:
  - `POST /run` returns immediately with `{ "ok": true, "runId": "<uuid>" }` **or** keep sync run from Task 3 **and** add async variant. Spec: Run becomes Cancel. Implement:
    - `POST /run` starts a thread, returns `{ "ok": true, "runId": str }` while `status` is `running`
    - `GET /run/{runId}` → `{ "status": "running"|"done"|"error"|"cancelled", "ok": bool, "columns"?, "rows"?, "error"? }`
    - `POST /run/{runId}/cancel` sets a flag; the worker checks between… **`run_scenario` is blocking with no cancel hook.** Spec still wants Cancel.
  - Honest v1: Cancel kills the worker thread is unsafe. Instead: **cancel means abandon the client wait**; the worker finishes but `GET` reports `cancelled` if cancel was requested before completion, and the UI does not plot that result. Document this in a comment on `runs.py`.
  - Test: start run, cancel, GET status is `cancelled` **or** `done` if it finished first — assert cancel endpoint returns 200 and subsequent UI payload `ok` is false when cancelled-before-done.

Simpler testable contract:

```python
def test_cancel_marks_run():
    c = TestClient(app)
    started = c.post("/run", json={"program": "hyper3", "stem": "input_climb", "end_time": 5})
    run_id = started.json()["runId"]
    c.post(f"/run/{run_id}/cancel")
    # poll until not running
    ...
    st = c.get(f"/run/{run_id}").json()
    assert st["status"] in {"cancelled", "done"}
    if st["status"] == "cancelled":
        assert st["ok"] is False
```

If the short 5 s run always finishes first, use a monkeypatched `run_scenario` that sleeps 2 s and checks a cancel event.

- [ ] **Step 1: Write test with monkeypatch sleep** so cancel wins

- [ ] **Step 2: FAIL**

- [ ] **Step 3: `RunRegistry` dict of `{event, status, result}`**

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit prepare**

---

### Task 5: Vite + React shell (start screen library)

**Files:**
- Create: `workbench/web/package.json` (mirror MISDC deps minus three/r3f), `vite.config.ts` proxy `/catalog`,`/cases`,`/run` → `:8001`, `tsconfig.json`, `tailwind.config.js` `darkMode: ["selector", ".dark"]`, `index.html`, `src/main.tsx`, `src/App.tsx`, `src/api.ts`, `src/store.ts`, `src/StartScreen.tsx`, `src/ThemeToggle.tsx`
- Test: `workbench/web/src/catalog.test.ts`

**Interfaces:**
- Produces: `fetchCatalog(): Promise<Catalog>` GET `/catalog`
- Zustand: `{ view: "start"|"editor", catalog, theme }`
- Start screen: twelve program folders from catalog; click folder then case (Task 6 loads editor)

- [ ] **Step 1: Vitest**

```ts
import { describe, it, expect, vi } from "vitest";
import { parseCatalog } from "./catalog";

it("maps hyper3 climb", () => {
  const c = parseCatalog({
    programs: [{ id: "hyper3", label: "HYPER3", cases: [{ stem: "input_climb", title: "climb" }] }],
  });
  expect(c.programs[0].cases[0].stem).toBe("input_climb");
});
```

- [ ] **Step 2:** `cd workbench/web && npm test` FAIL

- [ ] **Step 3: Scaffold + `parseCatalog` + StartScreen list.** Theme: first visit `prefers-color-scheme`; after click `localStorage` `cadac-theme` + `html.dark` (same as MISDC `misdc-theme`).

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit prepare**

---

### Task 6: Forms + JSONC drawer (source of truth)

**Files:**
- Create: `workbench/web/src/scenario.ts`, `scenario.test.ts`, `Editor.tsx`, `forms/*.tsx` (Overview, Modules, Vehicles, Timing, Events, Decks), `NamelistDrawer.tsx` (JSONC drawer)
- Modify: `store.ts`, `App.tsx`

**Interfaces:**
- Produces TypeScript `Scenario` matching JSONC: `title`, `options` (all `OPTION_KEYS` booleans), `modules: {name, phases: string[]}[]`, `timing: Record<string, number>`, `end_time`, `family?`, `iseed?`, `vehicles: Vehicle[]`
- `Vehicle`: `type`, `name`, `family?`, `aero_deck?`, `prop_deck?`, `weather_deck?`, `sam_deck?`, `srmb_deck?`, `params: Record<string, number | string>`, `events: { when: object, set: object }[]`
- `applyFormPatch(partial)` bumps `revision`, clears `parseError`
- `setDrawerText` dirty until generate
- Debounce 300 ms: `POST /cases/validate` with current scenario; on fail set `parseError`, **do not** replace `scenario`
- Drawer shows `JSON.stringify(scenario, null, 2)`
- Left nav labels exactly: Overview, Modules, Vehicles, Timing, Events, Decks, Results
- Top bar: case name, Save (`PUT /cases/{program}/{stem}`), Run (disabled if `parseError` or `runInFlight`)

Launch MISDC/AID buttons: render on vehicle row **disabled** with title “plan 5” — do not wire.

- [ ] **Step 1: Vitest round-trip**

```ts
import { scenarioFromJson, scenarioToJson } from "./scenario";

it("roundtrips hyper3-like", () => {
  const raw = {
    title: "t",
    options: { scrn: true, events: true, plot: true, doc: true, csv: true },
    modules: [{ name: "newton", phases: ["def", "init", "exec"] }],
    timing: { int_step: 0.01 },
    end_time: 90,
    vehicles: [{ type: "CRUISE3", name: "HYPER3", params: { alt: 3000 }, events: [] }],
  };
  const s = scenarioFromJson(raw);
  expect(s.vehicles[0].params.alt).toBe(3000);
  expect(scenarioToJson(s).end_time).toBe(90);
});
```

Unknown `options` key: `scenarioFromJson` keeps it in a side channel **or** leaves it in `options` so validate API can reject — keep extra keys on `options` so PUT hits `load_scenario` error. Test: extra key `nope` still present in `scenarioToJson`.

- [ ] **Step 2: FAIL**

- [ ] **Step 3: Forms for each nav section.** Params: one row per key; add-row for new param name. Blank number on blur → delete key.

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit prepare**

---

### Task 7: Results column picker + plots

**Files:**
- Create: `workbench/web/src/plot.ts`, `plot.test.ts`, `ResultsPane.tsx`
- Modify: Editor right pane, store `lastPlot: { columns, rows } | null`

**Interfaces:**
- `defaultColumns(columns: string[]): string[]` — intersection with `["alt","mach","dvbe","alphax"]` preserving that order; if empty, first four numeric names after `time`
- `hasGroundTrack(columns: string[]): boolean` — both `latx` and `lonx`
- Run: `POST /run` with current program/stem; on ok set `lastPlot`. Cancel calls cancel endpoint. Last good plot stays if a new run errors
- SVG: one polyline per selected column vs `time` (normalize y independently). Second SVG if ground track

- [ ] **Step 1:**

```ts
import { defaultColumns, hasGroundTrack } from "./plot";

it("prefers alt mach", () => {
  expect(defaultColumns(["time", "FSPV1", "alt", "mach", "lonx"])).toEqual(["alt", "mach"]);
});
it("falls back", () => {
  expect(defaultColumns(["time", "foo", "bar", "baz", "qux"])).toEqual(["foo", "bar", "baz", "qux"]);
});
it("ground", () => {
  expect(hasGroundTrack(["latx", "lonx", "alt"])).toBe(true);
  expect(hasGroundTrack(["alt"])).toBe(false);
});
```

- [ ] **Step 2–4: TDD then ResultsPane**

- [ ] **Step 5: Commit prepare**

---

### Task 8: start.sh / kill.sh

**Files:**
- Create: `workbench/start.sh`, `workbench/kill.sh` (copy MISDC pattern; `API_PORT=8001`, `WEB_PORT=5174`, API `uvicorn cadac_web.app:app`, prefer `workbench/api/.venv`)
- Modify: `.gitignore` — `workbench/.run/`, `workbench/web/node_modules/`, `workbench/api/.venv/`
- Test: `workbench/api/tests/test_start_script.py` — file exists, mentions `8001` and `5174`

```python
def test_start_sh_ports():
    text = Path("start.sh").read_text() if ... 
```

Read from repo `workbench/start.sh`.

- [ ] **Step 1: test ports in script FAIL**

- [ ] **Step 3: scripts + venv install notes in script comments** (same as MISDC README style, not a new md file)

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit prepare.** Update `README.md` workbench sentence from “specified, not built” to “`workbench/` Vite :5174 + API :8001; catalog/edit/run/plot. Handshake plan 3–5.”

Open file (`.jsonc` / `.asc`) can wait for a follow-up if time is short — spec requires Open file. Add **Task 8b** in this task: `POST /cases/import` multipart file; if `.asc`, `translate_scenario_asc` to temp then return scenario dict (not catalogued). Test with HYPER3 `input_climb.asc`.

```python
def test_import_asc():
    path = repo / "CADAC_Simulations/HYPER3_250114/HYPER3/input_climb.asc"
    r = TestClient(app).post("/cases/import", files={"file": (path.name, path.read_bytes())})
    assert r.json()["ok"]
    assert r.json()["scenario"]["vehicles"][0]["type"] == "CRUISE3"
```

Include this in Task 8 so Open file is not dropped.

---
