# Results Curve Groups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group the Results-tab curve checkboxes under a physics heading and, inside that, the CADAC module that defined the variable.

**Architecture:** Physics is a fixed name→bucket table in the web app. A component column such as `SBEL1` uses the stem when that stem is in the table. The module string comes from the vehicle store at the end of the run (`Field.module` of the stem) and rides on the existing plot JSON. The checkbox list merges modules in vehicle order: the first vehicle that has the column wins. Checking a box still plots that column against time.

**Tech Stack:** Python >= 3.11 (`cadac` + FastAPI workbench), TypeScript, React 18, Vitest, pytest.

**Spec:** Approved in chat on 2026-09-28 (bounded change; no separate spec file). The design is copied into Global Constraints below.

## Global Constraints

- Physics headings, in order, skipping empty groups: Position, Velocity, Attitude, Load, Rate, Aero, Propulsion, Guidance, Other.
- Load is accelerations and specific force. Rate is body and rotor rates. Commands, mode flags, and surface deflections are Guidance. Diagnostic errors and termination flags stay Other.
- A component uses the stem only when that stem is in the physics table (`SBEL1` → `SBEL`). `realp1` stays `realp1`.
- Module subheading is the raw `Field.module` string. Fixed order, then any other module name A–Z: `environment`, `kinematics`, `newton`, `aerodynamics`, `forces`, `euler`, `attitude`, `propulsion`, `guidance`, `control`, `actuator`, `seeker`.
- Names with no module are listed under the physics heading, before module subheadings. Within a module, keep the checkbox list's existing first-seen order.
- If two vehicles give the same column different modules, the earlier vehicle wins.
- `time` stays out of the checkbox list. `plot.csv` column layout does not change.
- TDD: failing test, watch it fail, minimal code, watch it pass. Grok non-fast implementers and reviewers only.
- Do not edit `CADAC_Simulations/`, `Python/build/`, or `README.md`. Bump `UPDATES.md` to `0.187.0` in Task 4.
- Commit steps run only when the user has asked for commits in that session.
- Tasks 1 and 2 are independent. Task 3 after Task 2. Task 4 after Tasks 1 and 3.

## File structure

- `workbench/web/src/plotGroups.ts` — physics table, `physicsOf`, `groupCurves`, `mergedColumnModules`. No React.
- `workbench/web/src/plotGroups.test.ts` — table and grouping proofs.
- `Python/src/cadac/io/plot.py` — `column_modules(store, columns)` using the same stem rules as `plot_row`.
- `Python/src/cadac/cli.py` — `RunResult.column_modules`; each track dict gains `"modules"`.
- `Python/tests/unit/test_column_modules.py` — stem map and a short HYPER3 run.
- `workbench/api/cadac_web/runs.py` — JSON `modules` on the done payload.
- `workbench/web/src/store.ts` — `modules?` on `PlotData` and `VehiclePlot`.
- `workbench/web/src/run.ts` — copy `modules` from GET `/run/{id}` onto `lastPlot`.
- `workbench/web/src/ResultsPane.tsx` — render the two-level checklist.
- `UPDATES.md` — `0.187.0` on top.

---

### Task 1: Physics and module grouping

**Files:**
- Create: `workbench/web/src/plotGroups.ts`
- Create: `workbench/web/src/plotGroups.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `PhysicsName` = `"Position" | "Velocity" | "Attitude" | "Load" | "Rate" | "Aero" | "Propulsion" | "Guidance" | "Other"`
  - `PHYSICS_ORDER: readonly PhysicsName[]` in the order above
  - `MODULE_ORDER: readonly string[]` = `environment`, `kinematics`, `newton`, `aerodynamics`, `forces`, `euler`, `attitude`, `propulsion`, `guidance`, `control`, `actuator`, `seeker`
  - `physicsOf(name: string): PhysicsName`
  - `CurveGroup = { physics: PhysicsName; loose: string[]; modules: { module: string; names: string[] }[] }`
  - `groupCurves(names: string[], modules: Record<string, string>): CurveGroup[]`
  - `mergedColumnModules(sources: { columns: string[]; modules?: Record<string, string> }[]): Record<string, string>`

- [ ] **Step 1: Write the failing test**

Create `workbench/web/src/plotGroups.test.ts`:

```typescript
import { expect, it } from "vitest";
import {
  PHYSICS_BY_STEM,
  groupCurves,
  mergedColumnModules,
  physicsOf,
  type PhysicsName,
} from "./plotGroups";

const EXPECTED: Record<string, PhysicsName> = {
  alt: "Position", hbe: "Position", latx: "Position", lonx: "Position", dbi: "Position",
  SBEL: "Position", SAEL: "Position", sbeg: "Position", SBEG: "Position",
  range_go: "Position", wp_grdrange: "Position", dwbh: "Position", STBG: "Position",
  dvbe: "Velocity", dvbi: "Velocity", dvba: "Velocity", mach: "Velocity", vmach: "Velocity",
  VBEL: "Velocity", VBEB: "Velocity", vbeg: "Velocity", VBEG: "Velocity",
  velocityx: "Velocity", altd: "Velocity",
  alphax: "Attitude", alphaix: "Attitude", alppx: "Attitude", alp: "Attitude",
  betax: "Attitude", betaix: "Attitude", phix: "Attitude", phipx: "Attitude",
  phibdx: "Attitude", phiblx: "Attitude", psix: "Attitude", psibd: "Attitude",
  psibdx: "Attitude", psiblx: "Attitude", psivdx: "Attitude", psivgx: "Attitude",
  psivlx: "Attitude", thtbdx: "Attitude", thtblx: "Attitude", thtvdx: "Attitude",
  thtvgx: "Attitude", thtvlx: "Attitude", gamma: "Attitude", psisbx: "Attitude",
  thtsbx: "Attitude",
  alx: "Load", anx: "Load", ayx: "Load", avx: "Load", FSPV: "Load", gmax: "Load", gminx: "Load",
  ppx: "Rate", qqx: "Rate", rrx: "Rate", qq: "Rate", omegax: "Rate", omega_rpm: "Rate",
  cl_ov_cd: "Aero", cla: "Aero", dma: "Aero", dmde: "Aero", pdynmc: "Aero", stmarg: "Aero",
  wnp: "Aero", wny: "Aero", zetp: "Aero", zety: "Aero", realp1: "Aero", realp2: "Aero",
  realy1: "Aero", realy2: "Aero", tpsp_ratio: "Aero",
  thrust: "Propulsion", thrst_stoch: "Propulsion", mass: "Propulsion", fmasse: "Propulsion",
  fmassr: "Propulsion", throttle: "Propulsion", fidle: "Propulsion", idle: "Propulsion",
  mprop: "Propulsion", power: "Propulsion", tav: "Propulsion", mil: "Propulsion",
  max: "Propulsion", cg: "Propulsion", xcg: "Propulsion", phis: "Propulsion", phisd: "Propulsion",
  alcomx: "Guidance", ancomx: "Guidance", altcom: "Guidance", phicx: "Guidance",
  phimvx: "Guidance", psivgcx: "Guidance", psivlcx: "Guidance", thtvgcx: "Guidance",
  delax: "Guidance", delex: "Guidance", delrx: "Guidance", delacx: "Guidance",
  delecx: "Guidance", delrcx: "Guidance", nl_gain: "Guidance", wp_flag: "Guidance",
  write: "Guidance", modes: "Guidance", mtargeting: "Guidance", time_go: "Guidance",
  tip: "Guidance", zetlagr: "Guidance",
  erq: "Other", etbl: "Other", lconv: "Other", sim_time: "Other",
};

it("assigns every listed stem and uses a listed stem for a component", () => {
  expect(PHYSICS_BY_STEM).toEqual(EXPECTED);
  expect(physicsOf("SBEL1")).toBe("Position");
  expect(physicsOf("SBEL2")).toBe("Position");
  expect(physicsOf("sbeg1")).toBe("Position");
  expect(physicsOf("SBEG1")).toBe("Position");
  expect(physicsOf("FSPV3")).toBe("Load");
  expect(physicsOf("realp1")).toBe("Aero");
  expect(physicsOf("no_such")).toBe("Other");
});

it("nests modules under physics, loose names first, input order kept", () => {
  expect(
    groupCurves(["mach", "alt", "SBEL1", "foo", "hbe"], { alt: "newton", SBEL1: "newton", mach: "environment" }),
  ).toEqual([
    { physics: "Position", loose: ["hbe"], modules: [{ module: "newton", names: ["alt", "SBEL1"] }] },
    { physics: "Velocity", loose: [], modules: [{ module: "environment", names: ["mach"] }] },
    { physics: "Other", loose: ["foo"], modules: [] },
  ]);
});

it("orders known modules first and unknown modules A to Z", () => {
  expect(
    groupCurves(
      ["psix", "gamma", "alphax"],
      { psix: "targeting", gamma: "trajectory", alphax: "kinematics" },
    ),
  ).toEqual([
    {
      physics: "Attitude",
      loose: [],
      modules: [
        { module: "kinematics", names: ["alphax"] },
        { module: "targeting", names: ["psix"] },
        { module: "trajectory", names: ["gamma"] },
      ],
    },
  ]);
});

it("keeps the first vehicle module and fills columns only that vehicle lacked", () => {
  expect(
    mergedColumnModules([
      { columns: ["time", "alt", "SBEL1"], modules: { alt: "newton", SBEL1: "newton" } },
      { columns: ["alt", "mach"], modules: { alt: "kinematics", mach: "environment" } },
    ]),
  ).toEqual({ alt: "newton", SBEL1: "newton", mach: "environment" });
  expect(mergedColumnModules([{ columns: ["alt"] }])).toEqual({});
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd workbench/web && npm test -- src/plotGroups.test.ts`

Expected: FAIL. `plotGroups` does not exist, or `PHYSICS_BY_STEM` / `groupCurves` / `mergedColumnModules` is not exported.

- [ ] **Step 3: Write the minimal implementation**

Create `workbench/web/src/plotGroups.ts`:

```typescript
export type PhysicsName =
  | "Position"
  | "Velocity"
  | "Attitude"
  | "Load"
  | "Rate"
  | "Aero"
  | "Propulsion"
  | "Guidance"
  | "Other";

export const PHYSICS_ORDER: readonly PhysicsName[] = [
  "Position",
  "Velocity",
  "Attitude",
  "Load",
  "Rate",
  "Aero",
  "Propulsion",
  "Guidance",
  "Other",
];

export const MODULE_ORDER: readonly string[] = [
  "environment",
  "kinematics",
  "newton",
  "aerodynamics",
  "forces",
  "euler",
  "attitude",
  "propulsion",
  "guidance",
  "control",
  "actuator",
  "seeker",
];

export const PHYSICS_BY_STEM: Record<string, PhysicsName> = {
  alt: "Position", hbe: "Position", latx: "Position", lonx: "Position", dbi: "Position",
  SBEL: "Position", SAEL: "Position", sbeg: "Position", SBEG: "Position",
  range_go: "Position", wp_grdrange: "Position", dwbh: "Position", STBG: "Position",
  dvbe: "Velocity", dvbi: "Velocity", dvba: "Velocity", mach: "Velocity", vmach: "Velocity",
  VBEL: "Velocity", VBEB: "Velocity", vbeg: "Velocity", VBEG: "Velocity",
  velocityx: "Velocity", altd: "Velocity",
  alphax: "Attitude", alphaix: "Attitude", alppx: "Attitude", alp: "Attitude",
  betax: "Attitude", betaix: "Attitude", phix: "Attitude", phipx: "Attitude",
  phibdx: "Attitude", phiblx: "Attitude", psix: "Attitude", psibd: "Attitude",
  psibdx: "Attitude", psiblx: "Attitude", psivdx: "Attitude", psivgx: "Attitude",
  psivlx: "Attitude", thtbdx: "Attitude", thtblx: "Attitude", thtvdx: "Attitude",
  thtvgx: "Attitude", thtvlx: "Attitude", gamma: "Attitude", psisbx: "Attitude",
  thtsbx: "Attitude",
  alx: "Load", anx: "Load", ayx: "Load", avx: "Load", FSPV: "Load", gmax: "Load", gminx: "Load",
  ppx: "Rate", qqx: "Rate", rrx: "Rate", qq: "Rate", omegax: "Rate", omega_rpm: "Rate",
  cl_ov_cd: "Aero", cla: "Aero", dma: "Aero", dmde: "Aero", pdynmc: "Aero", stmarg: "Aero",
  wnp: "Aero", wny: "Aero", zetp: "Aero", zety: "Aero", realp1: "Aero", realp2: "Aero",
  realy1: "Aero", realy2: "Aero", tpsp_ratio: "Aero",
  thrust: "Propulsion", thrst_stoch: "Propulsion", mass: "Propulsion", fmasse: "Propulsion",
  fmassr: "Propulsion", throttle: "Propulsion", fidle: "Propulsion", idle: "Propulsion",
  mprop: "Propulsion", power: "Propulsion", tav: "Propulsion", mil: "Propulsion",
  max: "Propulsion", cg: "Propulsion", xcg: "Propulsion", phis: "Propulsion", phisd: "Propulsion",
  alcomx: "Guidance", ancomx: "Guidance", altcom: "Guidance", phicx: "Guidance",
  phimvx: "Guidance", psivgcx: "Guidance", psivlcx: "Guidance", thtvgcx: "Guidance",
  delax: "Guidance", delex: "Guidance", delrx: "Guidance", delacx: "Guidance",
  delecx: "Guidance", delrcx: "Guidance", nl_gain: "Guidance", wp_flag: "Guidance",
  write: "Guidance", modes: "Guidance", mtargeting: "Guidance", time_go: "Guidance",
  tip: "Guidance", zetlagr: "Guidance",
  erq: "Other", etbl: "Other", lconv: "Other", sim_time: "Other",
};

const COMPONENT = /^(.+)([123])$/;

export function physicsOf(name: string): PhysicsName {
  const match = COMPONENT.exec(name);
  const stem = match != null && match[1] in PHYSICS_BY_STEM ? match[1] : name;
  return PHYSICS_BY_STEM[stem] ?? "Other";
}

export type CurveModuleGroup = { module: string; names: string[] };

export type CurveGroup = {
  physics: PhysicsName;
  loose: string[];
  modules: CurveModuleGroup[];
};

function moduleRank(name: string): [number, string] {
  const index = MODULE_ORDER.indexOf(name);
  if (index === -1) return [MODULE_ORDER.length, name];
  return [index, ""];
}

export function groupCurves(names: string[], modules: Record<string, string>): CurveGroup[] {
  const loose = new Map<PhysicsName, string[]>();
  const byModule = new Map<PhysicsName, Map<string, string[]>>();
  for (const name of names) {
    const physics = physicsOf(name);
    const module = modules[name];
    if (module == null || module === "") {
      const bucket = loose.get(physics) ?? [];
      bucket.push(name);
      loose.set(physics, bucket);
      continue;
    }
    const groups = byModule.get(physics) ?? new Map<string, string[]>();
    const bucket = groups.get(module) ?? [];
    bucket.push(name);
    groups.set(module, bucket);
    byModule.set(physics, groups);
  }
  const result: CurveGroup[] = [];
  for (const physics of PHYSICS_ORDER) {
    const modulesFor = byModule.get(physics);
    if ((loose.get(physics)?.length ?? 0) === 0 && (modulesFor == null || modulesFor.size === 0)) {
      continue;
    }
    const moduleNames = modulesFor == null ? [] : [...modulesFor.keys()];
    moduleNames.sort((a, b) => {
      const ra = moduleRank(a);
      const rb = moduleRank(b);
      if (ra[0] !== rb[0]) return ra[0] - rb[0];
      return ra[1] < rb[1] ? -1 : ra[1] > rb[1] ? 1 : 0;
    });
    result.push({
      physics,
      loose: loose.get(physics) ?? [],
      modules: moduleNames.map((module) => ({ module, names: modulesFor?.get(module) ?? [] })),
    });
  }
  return result;
}

export function mergedColumnModules(
  sources: { columns: string[]; modules?: Record<string, string> }[],
): Record<string, string> {
  const merged: Record<string, string> = {};
  for (const source of sources) {
    const table = source.modules ?? {};
    for (const name of source.columns) {
      if (name === "time" || merged[name]) continue;
      const module = table[name];
      if (module) merged[name] = module;
    }
  }
  return merged;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd workbench/web && npm test -- src/plotGroups.test.ts`

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

Only if the user asked for commits in this session:

```bash
git add workbench/web/src/plotGroups.ts workbench/web/src/plotGroups.test.ts
git commit -m "$(cat <<'EOF'
Group plotted column names by physics and module.

EOF
)"
```

---

### Task 2: Module map on the run result

**Files:**
- Modify: `Python/src/cadac/io/plot.py` (after `_VEC_COMPONENT`, before `flagged_plot_columns`)
- Modify: `Python/src/cadac/cli.py` (`RunResult` at lines 107–110; import line 7; track build at lines 333–344)
- Create: `Python/tests/unit/test_column_modules.py`

**Interfaces:**
- Consumes: `Field.module` on `StateStore`. `_VEC_COMPONENT` in `cadac.io.plot` (`SBEG1` → `sbeg`, `VBEG1` → `vbeg`, `FSPV1` → `FSPV`).
- Produces:
  - `column_modules(store, columns) -> dict[str, str]` in `cadac.io.plot`. Skips `time`. Omits a column whose stem is not in the store.
  - `RunResult.column_modules: dict` default `{}`. Slot-0 plot columns (`csv_columns` in `run_scenario`).
  - Each track dict gains `"modules": dict[str, str]` for that track's column list. Existing keys `name`, `columns`, `rows` stay.

- [ ] **Step 1: Write the failing test**

Create `Python/tests/unit/test_column_modules.py`:

```python
import json
import shutil
from pathlib import Path

from cadac.cli import run_scenario
from cadac.io.jsonc import loads
from cadac.io.plot import column_modules
from cadac.kernel.state import Field, StateStore


def test_column_modules_uses_vector_stem_and_skips_time():
    store = StateStore()
    store.define(Field("SBEL", (0.0, 0.0, 0.0), "vec", "state", "newton", ("plot",)))
    store.define(Field("alt", 1.0, "real", "out", "newton", ("plot",)))
    store.define(Field("sbeg", (0.0, 0.0, 0.0), "vec", "state", "newton", ("plot",)))
    store.define(Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces", ("plot",)))
    assert column_modules(store, ["time", "alt", "SBEL1", "SBEG2", "FSPV3", "missing"]) == {
        "alt": "newton",
        "SBEL1": "newton",
        "SBEG2": "newton",
        "FSPV3": "forces",
    }


def test_run_scenario_attaches_column_modules(tmp_path: Path):
    src = Path(__file__).resolve().parents[2] / "cases" / "hyper3"
    for path in src.glob("*.jsonc"):
        shutil.copy2(path, tmp_path / path.name)
    scenario = tmp_path / "input_climb.jsonc"
    data = loads(scenario.read_text(encoding="utf-8"))
    data["end_time"] = 0.05
    scenario.write_text(json.dumps(data) + "\n", encoding="utf-8")
    result = run_scenario(scenario)
    assert result.column_modules["alt"] == "newton"
    assert result.column_modules["SBEG1"] == "newton"
    assert result.column_modules["FSPV1"] == "forces"
    assert "time" not in result.column_modules
    assert result.tracks[0]["modules"]["alt"] == "newton"
    assert result.tracks[0]["modules"]["mach"] == "environment"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_column_modules.py -v`

Expected: FAIL. `column_modules` is not importable, then `RunResult` has no `column_modules` / tracks lack `"modules"`.

- [ ] **Step 3: Write the minimal implementation**

In `Python/src/cadac/io/plot.py`, after `_VEC_COMPONENT`:

```python
def column_stem(store, column):
    spec = _VEC_COMPONENT.get(column)
    if spec is not None:
        name = spec[0]
        if name in store:
            return name
        return None
    if column and column[-1] in "123":
        name = column[:-1]
        if name in store and store.field(name).type == "vec":
            return name
    if column in store:
        return column
    return None


def column_modules(store, columns):
    modules = {}
    for column in columns:
        if column == "time":
            continue
        stem = column_stem(store, column)
        if stem is None:
            continue
        modules[column] = store.field(stem).module
    return modules
```

In `Python/src/cadac/cli.py`, extend the plot import:

```python
from cadac.io.plot import PLOT_COLUMNS, column_modules, flagged_plot_columns, plot_row, write_plot_csv
```

Extend `RunResult`:

```python
@dataclass
class RunResult:
    plot_rows: list[dict]
    tracks: list[dict] = field(default_factory=list)
    column_modules: dict = field(default_factory=dict)
```

Where `run_scenario` builds `tracks` (the loop that appends `name` / `columns` / `rows`), set modules from that vehicle's store and the track's column list:

```python
columns = list(track_rows[slot][0].keys())
tracks.append(
    {
        "name": labels[slot],
        "columns": columns,
        "rows": track_rows[slot],
        "modules": column_modules(vehicles[slot].store, columns),
    }
)
```

On the `return RunResult(...)` line, pass slot-0 modules. `csv_columns` is already computed earlier in `run_scenario`:

```python
return RunResult(
    plot_rows=plot_rows,
    tracks=tracks,
    column_modules=column_modules(vehicles[0].store, csv_columns) if vehicles else {},
)
```

Do not change `write_plot_csv` or `PLOT_COLUMNS`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_column_modules.py tests/unit/test_plot_slot0.py -v`

Expected: PASS. `test_plot_slot0.py` still passes (`RunResult` default and track callers that build dicts themselves are unchanged).

- [ ] **Step 5: Commit**

Only if the user asked for commits in this session:

```bash
git add Python/src/cadac/io/plot.py Python/src/cadac/cli.py Python/tests/unit/test_column_modules.py
git commit -m "$(cat <<'EOF'
Report the CADAC module of each plotted column.

EOF
)"
```

---

### Task 3: Carry modules through the run API

**Files:**
- Modify: `workbench/api/cadac_web/runs.py` (`run_case` return around lines 75–80; `run_status` around lines 186–189)
- Modify: `workbench/api/tests/test_run_api.py`
- Modify: `workbench/web/src/store.ts` (`VehiclePlot` and `PlotData`, lines 10–20)
- Modify: `workbench/web/src/run.ts` (`GetRunBody` and the `PlotData` built around lines 79–83)
- Modify: `workbench/web/src/run.test.ts` (the "done ok keeps a track" test)

**Interfaces:**
- Consumes: `RunResult.column_modules` and track `"modules"` from Task 2.
- Produces: done-run JSON `"modules": { column: module }` plus each `vehicles[]` object may include `"modules"`. `PlotData.modules?: Record<string, string>` and `VehiclePlot.modules?: Record<string, string>`. `startRun` copies both onto `lastPlot`. Absent `modules` leaves the field off the plot object.

- [ ] **Step 1: Write the failing tests**

Add to `workbench/api/tests/test_run_api.py`:

```python
def test_run_status_includes_column_modules(monkeypatch):
    def spy(path):
        return types.SimpleNamespace(
            plot_rows=[{"time": 0.0, "alt": 1.0}],
            tracks=[
                {
                    "name": "Missile",
                    "columns": ["time", "alt", "mach"],
                    "rows": [{"time": 0.0, "alt": 1.0, "mach": 0.8}],
                    "modules": {"alt": "newton", "mach": "environment"},
                }
            ],
            column_modules={"alt": "newton"},
        )

    monkeypatch.setattr("cadac_web.runs.run_scenario", spy)
    client = TestClient(app)
    started = client.post("/run", json={"program": "hyper3", "stem": "input_climb", "end_time": 0.05})
    body = _wait_run(client, started.json()["runId"])
    assert body["modules"]["alt"] == "newton"
    assert body["vehicles"][0]["modules"]["mach"] == "environment"
```

In the existing `test_run_hyper3_short`, after the `rows` assertions, add:

```python
    assert body["modules"]["alt"] == "newton"
    assert body["vehicles"][0]["modules"]["alt"] == "newton"
```

In `workbench/web/src/run.test.ts`, extend the "done ok keeps a track for every vehicle" fetch body with `modules: { alt: "newton" }` and give the first vehicle `modules: { SBEL1: "newton", SBEL2: "newton", SBEL3: "newton" }`. Assert:

```typescript
  expect(store.getState().lastPlot?.modules).toEqual({ alt: "newton" });
  expect(store.getState().lastPlot?.vehicles?.[0].modules?.SBEL1).toBe("newton");
```

The existing `lastPlot?.vehicles` equality must include that `modules` field, because `startRun` passes vehicles through.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd workbench/api && python -m pytest tests/test_run_api.py::test_run_status_includes_column_modules tests/test_run_api.py::test_run_hyper3_short -v`

If that interpreter has no FastAPI, use `workbench/api/.venv/bin/python` instead of `python`.

Expected: FAIL. Response has no `modules`, or HYPER3 `modules["alt"]` is missing.

Run: `cd workbench/web && npm test -- src/run.test.ts`

Expected: FAIL. `lastPlot.modules` is undefined.

- [ ] **Step 3: Write the minimal implementation**

In `run_case`'s success dict (`workbench/api/cadac_web/runs.py`):

```python
        return {
            "ok": True,
            "columns": columns,
            "rows": rows,
            "vehicles": list(getattr(result, "tracks", None) or []),
            "modules": dict(getattr(result, "column_modules", None) or {}),
        }
```

In `run_status`, next to the existing `columns` / `rows` / `vehicles` copies:

```python
            payload["modules"] = result.get("modules", {})
```

In `workbench/web/src/store.ts`:

```typescript
export type VehiclePlot = {
  name: string;
  columns: string[];
  rows: Record<string, number>[];
  modules?: Record<string, string>;
};

export type PlotData = {
  columns: string[];
  rows: Record<string, number>[];
  vehicles?: VehiclePlot[];
  modules?: Record<string, string>;
};
```

In `workbench/web/src/run.ts`, add `modules?: Record<string, string>` to `GetRunBody`. When building `plot`:

```typescript
        const plot: PlotData = {
          columns: st.columns,
          rows: st.rows,
          ...(Array.isArray(st.vehicles) ? { vehicles: st.vehicles } : {}),
          ...(st.modules != null && typeof st.modules === "object" ? { modules: st.modules } : {}),
        };
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd workbench/api && python -m pytest tests/test_run_api.py::test_run_status_includes_column_modules tests/test_run_api.py::test_run_hyper3_short tests/test_run_api.py::test_run_returns_every_vehicle_track -v`

Expected: PASS. The spy test that omits `column_modules` still returns `"modules": {}`.

Run: `cd workbench/web && npm test -- src/run.test.ts`

Expected: PASS.

- [ ] **Step 5: Commit**

Only if the user asked for commits in this session:

```bash
git add workbench/api/cadac_web/runs.py workbench/api/tests/test_run_api.py workbench/web/src/store.ts workbench/web/src/run.ts workbench/web/src/run.test.ts
git commit -m "$(cat <<'EOF'
Pass plotted-column modules through the run API.

EOF
)"
```

---

### Task 4: Render grouped checkboxes

**Files:**
- Modify: `workbench/web/src/ResultsPane.tsx` (fieldset at lines 435–447; imports at the top)
- Modify: `workbench/web/src/ResultsPane.test.ts`
- Modify: `UPDATES.md` (new top entry `0.187.0`)

**Interfaces:**
- Consumes: `groupCurves`, `mergedColumnModules`, `CurveGroup` from `./plotGroups` (Task 1). `PlotData.modules` and `VehiclePlot.modules` (Task 3). Existing `pickable` (union of source columns, `time` removed, first-seen order) and `toggleSelectedColumn`.
- Produces: the checklist UI. No new exports. `README.md` stays as it is.

- [ ] **Step 1: Write the failing test**

Add to `workbench/web/src/ResultsPane.test.ts`:

```typescript
it("groups curve checkboxes by physics then module", () => {
  store.setState({
    program: "aim5",
    lastPlot: {
      columns: ["time", "alt", "mach", "foo", "SBEL1"],
      rows: [{ time: 0, alt: 1, mach: 0.5, foo: 1, SBEL1: 0 }],
      modules: { alt: "newton", mach: "environment", SBEL1: "newton" },
    },
    selectedColumns: ["alt"],
  });
  const host = mount();
  expect([...host.querySelectorAll("h3")].map((el) => el.textContent)).toEqual([
    "Position",
    "Velocity",
    "Other",
  ]);
  expect([...host.querySelectorAll("h4")].map((el) => el.textContent)).toEqual([
    "newton",
    "environment",
  ]);
  expect([...host.querySelectorAll("label")].map((el) => el.textContent)).toEqual([
    "alt",
    "SBEL1",
    "mach",
    "foo",
  ]);
  const alt = [...host.querySelectorAll("label")].find((el) => el.textContent === "alt");
  expect(alt?.querySelector("input")?.checked).toBe(true);
});

it("uses the first vehicle module when vehicles disagree", () => {
  store.setState({
    program: "aim5",
    lastPlot: {
      columns: ["time", "alt"],
      rows: [{ time: 0, alt: 1 }],
      vehicles: [
        {
          name: "Aim",
          columns: ["time", "alt"],
          rows: [{ time: 0, alt: 1 }],
          modules: { alt: "newton" },
        },
        {
          name: "Aircraft",
          columns: ["time", "alt", "mach"],
          rows: [{ time: 0, alt: 2, mach: 0.8 }],
          modules: { alt: "kinematics", mach: "environment" },
        },
      ],
    },
    selectedColumns: [],
  });
  const host = mount();
  expect([...host.querySelectorAll("h4")].map((el) => el.textContent)).toEqual([
    "newton",
    "environment",
  ]);
});

it("shows physics headings when the plot has no module map", () => {
  store.setState({
    program: "aim5",
    lastPlot: {
      columns: ["time", "alt", "foo"],
      rows: [{ time: 0, alt: 1, foo: 2 }],
    },
    selectedColumns: [],
  });
  const host = mount();
  expect([...host.querySelectorAll("h3")].map((el) => el.textContent)).toEqual(["Position", "Other"]);
  expect(host.querySelectorAll("h4")).toHaveLength(0);
  expect([...host.querySelectorAll("label")].map((el) => el.textContent)).toEqual(["alt", "foo"]);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd workbench/web && npm test -- src/ResultsPane.test.ts`

Expected: FAIL. The new tests find no `h3` physics headings. Existing tests in that file still pass.

- [ ] **Step 3: Write the minimal implementation**

In `ResultsPane.tsx`, import `groupCurves` and `mergedColumnModules` from `./plotGroups`.

After `pickable` is computed, add:

```tsx
  const columnModules = useMemo(() => {
    if (lastPlot == null) return {};
    if (lastPlot.vehicles != null && lastPlot.vehicles.length > 0) {
      return mergedColumnModules(lastPlot.vehicles);
    }
    return mergedColumnModules([{ columns: lastPlot.columns, modules: lastPlot.modules }]);
  }, [lastPlot]);
  const groups = useMemo(
    () => groupCurves(pickable, columnModules),
    [columnModules, pickable],
  );
```

Replace the checkbox map inside the existing fieldset (keep the `<legend>columns vs time</legend>`) with:

```tsx
        {groups.map((group) => (
          <div key={group.physics}>
            <h3 className="mb-1 mt-2 text-xs font-semibold text-slate-600 dark:text-slate-300">
              {group.physics}
            </h3>
            {group.loose.map((name) => (
              <label key={name} className="flex items-center gap-2 pl-2">
                <input
                  type="checkbox"
                  checked={selected.includes(name)}
                  onChange={() => toggleSelectedColumn(name)}
                />
                <FieldLabel label={name} hint={columnHint(program, name)} />
              </label>
            ))}
            {group.modules.map((mod) => (
              <div key={mod.module}>
                <h4 className="mb-1 mt-1 pl-2 text-xs font-medium text-slate-500 dark:text-slate-400">
                  {mod.module}
                </h4>
                {mod.names.map((name) => (
                  <label key={name} className="flex items-center gap-2 pl-4">
                    <input
                      type="checkbox"
                      checked={selected.includes(name)}
                      onChange={() => toggleSelectedColumn(name)}
                    />
                    <FieldLabel label={name} hint={columnHint(program, name)} />
                  </label>
                ))}
              </div>
            ))}
          </div>
        ))}
```

`key` on a label stays the column name. One name appears once.

Add this block at the top of `UPDATES.md`, above `0.186.1`:

```markdown
## 0.187.0 - Results curves grouped by physics and module
- The Results checklist groups each plotted name under Position, Velocity, Attitude, Load, Rate, Aero, Propulsion, Guidance, or Other, then under the CADAC module that defined it.
- The run payload includes `modules` on the slot-0 plot and on each vehicle track. A component uses the vector stem's module. The first vehicle that has the column supplies the module. A name with no module sits under its physics heading.
- `plot.csv` is unchanged.
- Tests: `test_column_modules.py`, `test_run_api.py`, `plotGroups.test.ts`, `run.test.ts`, `ResultsPane.test.ts`.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd workbench/web && npm test -- src/ResultsPane.test.ts src/plotGroups.test.ts src/run.test.ts`

Expected: PASS, including the older Results pane tests (chart captions, ground track, trajectory).

- [ ] **Step 5: Commit**

Only if the user asked for commits in this session:

```bash
git add workbench/web/src/ResultsPane.tsx workbench/web/src/ResultsPane.test.ts UPDATES.md
git commit -m "$(cat <<'EOF'
Group Results curve checkboxes by physics and module.

EOF
)"
```
