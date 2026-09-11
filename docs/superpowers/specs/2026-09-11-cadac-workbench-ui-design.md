# CADAC workbench, AID web, aero handshake — design

Date: 2026-09-11
Status: approved (chat + spec review). Plans:

1. `docs/superpowers/plans/2026-09-11-cadac-catalog.md`
2. `docs/superpowers/plans/2026-09-11-cadac-workbench.md`
3. `docs/superpowers/plans/2026-09-11-cadac-handshake.md`
4. `docs/superpowers/plans/2026-09-11-aid-web.md`
5. `docs/superpowers/plans/2026-09-11-cadac-roundtrip.md`

## Purpose

A local browser workbench for the Python `cadac` library: load every Zipfel CADAC++ demo as JSONC, edit the full scenario (modules, timing, vehicles, events, decks), run, and plot. Define or refine a flying body in sibling aero apps and write CADAC `aero_deck.jsonc` tables.

Three localhost apps. CADAC orchestrates. Siblings are not embedded.

| App | Repo | Role |
|---|---|---|
| CADAC workbench | this repo, `workbench/` | Case library, scenario forms, run, plot, handshake hub |
| MISDC | `/home/valentin/Projects/FlightSimulation/MDT/MISDC2026` | Slender bodies (missiles/rockets); Fortran MDT |
| AID web | `/home/valentin/Projects/FlightSimulation/USAF_DATCOM/AircraftIntuitiveDesign` | Aircraft; Digital DATCOM / Tornado / AVL / flow5 |

## Non-goals

- Embedding MISDC or AID in an iframe or npm package.
- Editing any file under `CADAC_Simulations/` (C++ stays the Zipfel dump).
- Porting HYPER6 `RADAR0` / `SAT3` (still unregistered; those cases translate but Run fails with the existing `cadac` error).
- Inventing F-16-style `PLANE6` tables (`cx_vs_elev_alpha`, …) from AID polars.
- Replacing harvested e2e goldens or changing `run_scenario` numerics.
- Pixel-parity with MATLAB `AID.m` beyond the existing PySide layout being ported.
- Auth, remote deploy, Monte Carlo UI, 3D CADAC trajectory.

## Process

One spec, five sequential implementation plans. Each plan uses **subagent-driven-development**: Grok non-fast implementer per atomic task, TDD, then a Grok non-fast reviewer. No production code without a failing test. No commits unless the user asks.

1. Catalog — every scenario `.asc` → JSONC under `Python/cases/<program>/`
2. CADAC workbench — library, forms, JSONC drawer, run, plots
3. Handshake contract + mapper + MISDC routes
4. AID web port (full PySide feature set in the browser)
5. CADAC round-trip UI (Launch / preview / confirm / Import file)

Plans 4 and 5 may be split internally; they stay this spec’s scope.

## Architecture

```
BodyNavigation/workbench/
  start.sh kill.sh          CADAC API :8001 + Vite :5174 (.run/ gitignored)
  api/                      FastAPI wrapping cadac + cadac.aero_map
  web/                      Vite + React 18 + TypeScript + Tailwind (darkMode like MISDC)

MISDC2026/                  existing :8000 / :5173; add handshake client routes
AircraftIntuitiveDesign/
  Python/aid                unchanged engine
  Python/aid_gui            PySide remains until web parity; not the CADAC sibling
  api/                      new FastAPI wrapping aid
  web/                      new Vite/React/Tailwind (MISDC chrome)
```

Default ports: MISDC 8000/5173, CADAC 8001/5174, AID 8002/5175. BodyNavigation `workbench/start.sh` boots CADAC only; it may offer flags to start siblings. Two (or three) localhost processes, no auth.

`cadac` remains the simulation kernel. The workbench API does not reimplement EOM.

## Catalog

Source trees (do not modify):

- `CADAC_Simulations/<PROGRAM>_*/<PROGRAM>/` via `cadac_cpp.extract_cpp.PROGRAM_DIRS`
- `CADAC_Simulations/HYPER6 Input Problems for Sec 10_4/` → `Python/cases/hyper6/`
- `CADAC_Simulations/AGM6_250217/Additional input Files/` → `Python/cases/agm6/`

Skip: `readme.asc`, `documentation.asc`, `doc.asc`, `input_copy.asc`.

Scenario vs deck: if `_parse_scenario_asc` yields `vehicles` or `modules`, write `{stem}.jsonc` with `translate_scenario_asc(..., family=...)`. Else treat as a table deck (`deck_asc_to_jsonc`) into the same program folder. Never overwrite an existing `Python/cases/**/*.jsonc` (committed e2e cases stay hand-checked). Only missing stems are written. Re-run is idempotent.

Family stamp (same as current translate tests):

| Program | family |
|---|---|
| AIM5 | `aim5` |
| CRUISE5 | `cruise5` |
| MAGSIX | `magsix` |
| ROCKET6 | `rocket6` |
| SAM6 | `sam6` |
| SRAAM6 | `sraam6` |
| AGM6 | `agm6` |
| others | omit |

CLI: `python -m cadac.io.catalog` (`Python/src/cadac/io/catalog.py`) writes JSONC and `Python/cases/catalog-report.json` `{translated, skipped, failed: [{path, error}]}`. Failed files stay out of the library list with their error. Re-run is safe.

Open file in the UI: `.jsonc` load; `.asc` translate to a temp JSONC then load. Not added to the catalog unless Save As into `Python/cases/`.

## CADAC UI

MISDC-shaped chrome.

**Start:** twelve program folders (HYPER3 … AGM6). Open a folder, pick a case, enter the editor. New (blank scenario for a chosen program/type), Open file, theme toggle like MISDC.

**Editor top bar:** loaded case name, Save, Run / Cancel. Launch MISDC / Launch AID live on each vehicle row (see routing).

**Left nav:** Overview, Modules, Vehicles, Timing, Events, Decks, Results.

**Center:** forms are the source of truth. Fields match the JSONC: `title`, `options`, `modules[]` (name + phases), `timing`, `end_time`, `family`, `iseed`, `vehicles[]` (`type`, `name`, `family`, deck paths, `params`, `events`). Unknown `params` keys still appear (CADAC vehicles have open param dicts). Blank number → JSON `null` / omit; commit on blur.

**Right:** Results column picker after a successful run (see Plots). Empty until Run succeeds.

**Bottom drawer:** generated JSONC, dirty/revision like MISDC namelist. Failed parse does not replace the last good scenario. Run disabled on parse error or in-flight.

Forms write JSONC; Save writes the file. `cadac.io.scenario` validation is the schema (unknown `options` key is an error).

### Launch routing

Key is `(scenario family or program, vehicle type)`. Buttons sit on the vehicle row.

| Program / family | type | Sibling |
|---|---|---|
| AIM5 / `aim5` | `AIM5` | MISDC |
| SRAAM6 / AGM6 / SAM6 | `MISSILE6` | MISDC |
| SAM6 | `ROCKET5` | MISDC |
| ROCKET6 / `rocket6` | `HYPER6` | MISDC |
| FALCON5 | `PLANE` | AID |
| FALCON6 | `PLANE6` | AID |
| AIM5 / SAM6 / AGM6 | `AIRCRAFT3` | AID |
| CRUISE5 / `cruise5` | `CRUISE3` | AID |
| HYPER3 / HYPER5 / HYPER6 | `CRUISE3` / `HYPER5` / `HYPER6` | none (GHAME deck) |
| any | `ROTOR`, `TARGET3`, `SATELLITE3`, `RADAR0`, `GROUND0` | none |

User can still Import file for any vehicle. FALCON6 Launch AID is allowed for geometry; Confirm will not replace F-16 table names (merge-only).

## Plots

`run_scenario` plot CSV (vehicle slot 0). Right pane: checklist of numeric columns, overlay vs `time`. Defaults if present: `alt`, `mach`, `dvbe`, `alphax` (else first four numeric columns after `time`). Extra chart: `latx` vs `lonx` when both exist. Last good plot stays until a new run completes. CSV remains on disk; no spreadsheet tab.

## Handshake

CADAC `POST /handshake/sessions` → `{id, vehicle, family, type, callback}`. Launch opens

`http://127.0.0.1:5173/?cadacSession=<id>` (MISDC) or `:5175` (AID).

Sibling loads optional starter geometry if the session includes a project path; otherwise New. User edits and Run/Analyze as today. On success the sibling `POST`s CADAC `/handshake/sessions/{id}/complete` with:

```json
{
  "source": "misdc",
  "solver": "mdt",
  "axes": { "mach": [0.5, 0.8], "alpha": [-2, 0, 4], "beta": [0] },
  "tables": { "cn": [[...]], "cm": [[...]], "ca": [[...] ] },
  "ref": { "sref": 1.0, "lref": 1.0, "xcg": 0.5 }
}
```

AID uses `source: "aid"`, `solver: "datcom"|"tornado"|"avl"|"flow5"`, tables `cl` / `cd` / `cm` (and derivatives when the solver provides them) vs `alpha` × `mach`.

File Import is the same JSON (or MISDC `for006` / AID Analyze JSON) without a session.

Sessions expire (default 1 h). Analyze/Run failure does not complete the session. If the sibling is down, Launch says start that app; CADAC keeps working.

CADAC API allows CORS (or Vite proxy) from `http://127.0.0.1:5173` and `:5175` so the sibling can POST complete. CADAC web proxies `/` API to `:8001` like MISDC does to `:8000`.

## Mapper (`cadac.aero_map`)

Family schema = every `look_up` name that vehicle’s `aero.py` calls (required). Auto-map fills what the payload can produce. Preview rows: **mapped** | **merged** (kept from the current `aero_deck.jsonc`) | **missing**.

Confirm writes only when every required name is mapped or merged. Default: merge gaps from the current deck. Unmapped required with no template → Confirm disabled (no “require complete” override in v1 — that would ship a deck `look_up` cannot read).

Documented aliases (MDT static rows → missile tables), Mach × α grids:

| CADAC (SRAAM6 / AGM6) | MDT |
|---|---|
| `cn0_vs_mach_alpha` | `cn` |
| `clm0_vs_mach_alpha` | `cm` |
| `ca0_vs_mach` | `ca` at α=0 (or α-averaged if α=0 absent) |

AIM5: `cl_aim_vs_alpha_mach` ← MDT `cl` if present else `cn`; `cd_aim_on_vs_alpha_mach` and `cd_aim_off_vs_alpha_mach` ← MDT `cd` if present else `ca` (same grid copied to on and off; preview marks copied). SAM6 3-D `(mach, beta, alpha)`: use MDT beta=0 slice if beta axis missing; remaining β tables merge from template. ROCKET6 `*slv{n}_*` maps onto the active `slv` suffix; other stages merge from template.

FALCON5: AID `CL`/`CD` vs α, Mach → `cl_30MAC_vs_mach_alphax` / `cd_30MAC_vs_mach_alphax` (and 35/40 copies of the same polar unless AID `XCG`/%MAC distinguishes them). FALCON6: do not synthesize `cx_vs_elev_alpha` etc. Preview is merge-only unless the import is already a CADAC F-16 deck.

CRUISE5 `cd0_vs_mach` / `cl0_vs_mach` / `cla_vs_mach`: derive from AID CL/CD vs α at each Mach (`cd0` at α=0, `cla` from dCL/dα near 0). `ckk` / `cla0` merge from template if not derived.

GHAME HYPER tables: mapper does not run unless the user Import-file a compatible deck.

Coefficients are not invented. NaN/Inf from DATCOM stay missing.

## AID web

Replace PySide as the CADAC sibling. Wrap existing `aid` (no second engine). React screens match current `aid_gui`: File New/Load/Save/Recent; tabs Wing/HT/VT/Control/Body/Aero/`+`; Analyze DATCOM/Tornado/AVL/flow5; Results geometry/stability/aerodynamics; 3D (three.js or R3F, not PyVista). Bundled `Python/models/*.jsonc` on Examples. Analyze returns the same coefficient dicts the mapper consumes.

`aid_gui` PySide stays installable until the web app covers those Analyze paths; handshake is web-only.

## Error handling

| Failure | UI |
|---|---|
| Invalid JSONC / unknown option / missing type | Parse error on form+drawer; Run disabled; last good scenario kept |
| Catalog translate fail | `catalog-report.json`; case omitted from library |
| Unregistered type | `cadac` error in log; no success plot |
| Run throw / timeout | Toast + log; do not plot partial CSV as success |
| Sibling down | Launch message; CADAC continues |
| Handshake timeout / Analyze fail | No deck write; Import still works |
| Required table missing and no template | Preview red; Confirm disabled |
| MDT / DATCOM / AVL / flow5 binary missing | Sibling 400; CADAC shows session error |

## Testing

- Catalog: skip list; HYPER6 §10.4 eight files exist as JSONC; `translate_scenario_asc` on a golden subset; report lists failures without aborting the rest.
- API: catalog index, reject bad JSONC, save, short `run_scenario`, cancel, handshake create/complete/timeout.
- Mapper: MDT fixture → SRAAM/AGM required names; AID DATCOM fixture → FALCON5 `cl_30MAC`/`cd_30MAC`; FALCON6 payload without F-16 tables → Confirm blocked if no template merge; missing CN → missing row.
- Web: Vitest form ↔ JSONC (modules, vehicles, events, dirty/revision). No Playwright matrix.
- MISDC: new handshake routes only; existing MDT e2e unchanged.
- AID: existing `aid` tests stay the engine oracle; new API Analyze DATCOM returns mapper-shaped tables.
- CI: unit + mapper + one short run (e.g. HYPER3 0.1 s). Not the full Zipfel library.

## Implementation notes

- Do not change C++ CADAC or harvested goldens.
- Workbench depends on installed `cadac` (`Python/`).
- `.superpowers/brainstorm/` is gitignored; `.superpowers/sdd/` already is.
- After each plan: `UPDATES.md` bump; README only if layout/ports/handshake change.
