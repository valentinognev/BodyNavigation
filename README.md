# Modeling and Simulation of Aerospace Vehicles (Zipfel)

## Idea
Workspace for Peter H. Zipfel's CADAC++ simulations (3/5/6-DOF aerospace vehicles) and a unified Python library that re-implements them in Python style, not as a line-for-line C++ transcription.

Mandatory reading: `UPDATES.md`.

## Architecture
**C++ source of truth:** `CADAC_Simulations/` — twelve standalone CADAC++ programs. Each copies executive, `Variable` tables, `Matrix`/`Datadeck`, atmosphere, and I/O. Hierarchy is `Cadac` → `Round3`/`Flat3`/`Round6`/`Flat6` → vehicle (Cruise, Plane, Missile, …). Multi-vehicle runs use a `combus` packet bus.

| Folder | DOF / earth | Vehicles |
|---|---|---|
| HYPER3 | 3DOF round | Cruise |
| AIM5 | 5DOF flat | Aim + Aircraft |
| CRUISE5 | 5DOF round | Cruise + Satellite + Target |
| FALCON5 | 5DOF flat | Plane |
| HYPER5 | 5DOF round | Hyper + Satellite + Target |
| FALCON6 | 6DOF flat | Plane |
| HYPER6 | 6DOF round (+ 3DOF) | Hyper + Radar + Satellite + Ground |
| MAGSIX | 6DOF rotor | Rotor |
| ROCKET6 | 6DOF round | Rocket |
| SAM6 | 6DOF flat (+ 3DOF) | Missile + Aircraft + Rocket + Radar |
| SRAAM6 | 6DOF flat (+ 3DOF) | Missile + Target |
| AGM6 | 6DOF flat (+ 3DOF) | Missile + Aircraft + Target |

Python `vehicles/` is grouped by kernel: round3 (hyper3, cruise5, hyper5), flat3 (falcon5, aim5), flat6 (falcon6, sraam6, agm6, sam6), round6 (hyper6, rocket6), planar (magsix).

**Python target:** `Python/` — installable package `cadac`. Runtime JSONC (scenarios + decks); `cadac.io.translate.deck_asc_to_jsonc` converts CADAC `.asc` decks. Named numpy state; CADAC `integrate` / `look_up` / atmospheres (`atmosphere76` vacuum + T=186.946 when geometric alt ≥ 84.852 km, matching C++). `cadac.stoch` is glibc `rand()` / `gauss` / `uniform` / Dryden white noise (`run_scenario` seeds `cfg.iseed`; default 0; AGM6 test-case ASC iseed 12345). Round6 Dryden (`mturb=1`) burns 15 GPS+startrack MARKOV `gauss` then white noise. Round6/INS Euler treats `|tbd13| >= 1-1e-14` as C++ `|tbd13|>=1` (vertical-launch ulps). Round6 kinematics/euler/newton and ROCKET6 INS/LTG use C++ `Matrix::operator*` (row-major ijk, `cadac_matmul`); `cad_tgi84` is `TGD*TDI` ijk; euler `IBBB.inverse()` is adjoint/det (`cadac_inverse`); ROCKET6 GPS/startrack use the same multiply/inverse (8×8 GPS Kalman inverse is recursive like C++, slow). ROCKET6 `mins=1` `init_ins` draws 18 `def_ins` gauss (RTL) then 9 unit Cholesky `gauss` (`iseed` 1234); do not zero ESBI. SAM6 `mins=1` `init_ins` draws ASpec `gauss` (g++ RTL) then 9 unit Cholesky `gauss`; `ins()` walka uses RTL `uniform`. Harvest `nmonte==0` still draws Dryden `rand()` and MARKOV `gauss`, then zeros MARKOV stores. SAM6 deck GAUSS/MARKOV skip `gauss()` when `nmonte==0`. `run_loop` skips module names missing on a vehicle and seeds combus from store `com_names` before the first execute; plot CSV records vehicle slot 0 only. MAGSIX Rotor also appends the C++ post-loop last-integration plot row (`sim_time=end_time+int_step`); `Rotor::plot_data` ignores `merge` so that row is not `time=-1`. `Flat3Newton.initialize` matches C++ `init_newton`: does not write `alt` (stays 0 until first `newton()`); FALCON5 first-step propulsion/control therefore see `alt=0`.

**Mapper (0.170):** `cadac.aero_map` — `AeroPayload` (MISDC `for006` / AID lists) → family `look_up` tables (`mapped` / `merged` from current `aero_deck` / `missing`). Confirm only if every required name is mapped or merged. FALCON6 is merge-only (do not invent `cx_vs_elev_alpha`, …). GHAME HYPER: empty schema (Import-file of an already CADAC deck only). Non-finite coefficients omitted. ROCKET6 copies leftover SLV stages from the template; SAM6 wraps a 2-D polar as one β plane.

**Catalog (0.170):** `cadac.io.catalog` walks `PROGRAM_DIRS` plus `CADAC_Simulations/HYPER6 Input Problems for Sec 10_4/` → `Python/cases/hyper6/` and `AGM6_250217/Additional input Files/` → `Python/cases/agm6/`. Skip stems `readme`/`documentation`/`doc`/`input_copy`. Scenario vs deck via `parse_scenario_asc` (`modules`/`vehicles`) else `deck_asc_to_jsonc`. Family stamps AIM5 `aim5` … AGM6 `agm6`; others omit. Never overwrites existing `Python/cases/**/*.jsonc`. CLI: `cd Python && PYTHONPATH=src:tools python -m cadac.io.catalog` writes missing JSONC and `Python/cases/catalog-report.json` `{translated, skipped, failed}`. HYPER5 Demo 4.5 stays in `failed` (Latin-1 `°` in TITLE). Re-run is idempotent. Do not edit CADAC `.asc`.

**Workbench (0.170):** `workbench/` Vite :5174 + FastAPI :8001 wrapping `cadac` (no EOM copy). `./start.sh` / `./kill.sh` at repo root wrap `workbench/start.sh` / `kill.sh` (start kills any previous instance then starts API+Vite; creates `workbench/api/.venv` and `pip install -e .` if `cadac` is missing; kill is stop-only; PIDs in `workbench/.run/`). Catalog lists `Python/cases` scenarios only (not aero/prop decks). Forms + JSONC drawer are the source of truth; Run posts the in-memory scenario (does not mutate library files); `plot.csv` stays vehicle slot 0. Time charts, the `latx`/`lonx` ground track, and the 3D path each draw every vehicle that has that quantity, on one scale, in its own color (`Missile 1` when names repeat). Defaults are `alt`/`mach`/`dvbe`/`alphax`. The 3D path uses `latx`/`lonx`/`alt`, otherwise `SBEL`, otherwise `SAEL` (component 2 east, component 1 north, component 3 down negated). GIF turntable and the draggable WebGL view draw axes, labels, markers, and legend in-canvas so Record WebM matches the still. Open file `.jsonc`/`.asc` is not catalogued (Save/Run disabled until Save As). Handshake sessions exist (`POST /handshake/sessions`, complete, GET preview, confirm writes the vehicle `aero_deck` file next to the case iff `can_confirm`; mapper failure on GET/confirm is HTTP 400 `{ok: false}`; `CADAC_CASES` overrides case root). Launch MISDC/AID UI still plan 5. CORS `127.0.0.1:5173|:5174|:5175`. Sibling MISDC (`?cadacSession=`) and AID-web (`AircraftIntuitiveDesign` `./start-web.sh` API :8002 / Vite :5175) POST complete to `:8001`; `cadac.aero_map` writes `aero_deck.jsonc`. A scenario JSONC may open with a block comment (vehicles, simulation kind, and what the run is for). GET and import return it as `description`; Overview shows it read-only; PUT writes that comment back in front of the scenario.

**Quality kernel (0.169):** `StateStore.set` skips coerce when type and shape already match; `"name" in store` is O(1). `run_loop` binds execute chains once and reuses `SimContext`. `look_up` last-index cache; `cadac_matmul` 3×3 unrolled ijk. Shared `cadac.math.frames` (`cadac_sign`, `skew`, `hypot3`, `quat_to_dcm`, `incidence_angles`). EOM modules inherit `ModuleBase` (copy `Field` on define). Survey: `Python/tools/cadac_quality` (`baseline.json` frozen, `latest.json` after overhaul).

**Parity audit tools:** `Python/tools/cadac_cpp/` — not in the `cadac` runtime package. Pytest `pythonpath` is `["src", "tools"]`. `harvest_table.HARVEST_ROWS` is the 14-row JSONC↔ASC↔golden map later harvest/g++ tasks consume. `schema.InventoryRow` is the frozen inventory row (`program`, `kind`, `name`, `cpp`, `python`, `status`, `note`) with `dump_inventory`/`load_inventory` JSON and `KINDS`/`STATUSES` tuples. `extract_cpp.extract_vehicle_types` parses `set_obj_type` `strcmp(temp,"TYPE")` names (HYPER6 keeps C++ `SAT3`, not Python `SATELLITE3`); `PROGRAM_DIRS` maps the twelve program names to CADAC folders. `kernel_rows()` emits the 19 kernel/I/O rows (`program="kernel"`). `scan_all(root)` walks all twelve C++ programs plus `Python/src/cadac`, appends one `harvest` row per `HARVEST_ROWS` (`ok`→`ported`) and one `e2e` row per JSONC e2e file (`passed`→`ported`, `failed`→`diverged`, `skipped`→`missing`), then sorts by `(program, kind, name)`. CLI `PYTHONPATH=tools python -m cadac_cpp.inventory` writes `Python/tools/cadac_cpp/inventory.json` (HYPER6 `SAT3`/`RADAR0` are `ported` in `_VEHICLE_TYPES`). `build_cadac.build_program` compiles a CADAC program with `Makefile.cadac` and `-include compat.hpp` (`system("pause")` no-op, `_itoa`); binaries under `Python/tools/cadac_cpp/build/` (gitignored). `harvest.harvest_row` snapshots preexisting CADAC I/O (`input.asc`, `doc.asc`, `input_copy.asc`, plot/traj csv), writes a MONTE-off / `y_csv` copy of the mapped ASC, unlinks leftover `plot.csv`/`plot1.csv` (and `plot.asc`/`plot1.asc`) in the C++ cwd, runs the binary with cwd=the CADAC program folder, copies the plot (`plot1.csv` preferred over `plot.csv`; FALCON5 `exit(1)` falls back to `csv_from_plot_asc`), and restores that snapshot in `finally`. CADAC run leftovers (`plot*.asc`/`plot*.csv`, `tabout.asc`, `traj.asc`/`traj.csv`) are gitignored. HYPER3 canary dest is `Python/tests/e2e/goldens/hyper3/plot1.gpp.csv` (does not overwrite committed `plot1.csv`). `harvest_all(skip_failed=True)` builds each `HARVEST_ROWS` program then harvests; returns `{golden: ok|build_failed|run_failed}`. Timeouts: HYPER3 120s, CRUISE5 900s, ROCKET6 600s, others 300s. Do not edit CADAC `.cpp`/`.hpp`.

`VehicleSpec.family` optional. Family set → `_VEHICLE_FAMILIES[(family, type)]` only (no fallthrough to `_VEHICLE_TYPES`). Unknown-type sentinel: `"NO_SUCH_TYPE"`. All twelve C++ programs have a JSONC case. HYPER6 `SAT3` and `RADAR0` are registered in `_VEHICLE_TYPES`. Ground0 is not a vehicle; its tracks live on the radar.

| Program | Family | Types | Case |
|---|---|---|---|
| HYPER3 | (none) | `CRUISE3` | `Python/cases/hyper3/` climb |
| FALCON5 | (none) | `PLANE` | `Python/cases/falcon5/` turning to IP |
| FALCON6 | (none) | `PLANE6` | `Python/cases/falcon6/` gamma |
| HYPER5 | (none) | `HYPER5` / `TARGET3` / `SATELLITE3` | `Python/cases/hyper5/` Demo 4.7 |
| HYPER6 | (none) | `HYPER6` / `SAT3` / `RADAR0` | `Python/cases/hyper6/` climb + eight §10.4 |
| AIM5 | `aim5` | `AIM5` (also global) / `AIRCRAFT3` | `Python/cases/aim5/` hori |
| CRUISE5 | `cruise5` | `CRUISE3` / `TARGET3` / `SATELLITE3` | `Python/cases/cruise5/` input_1 |
| MAGSIX | `magsix` | `ROTOR` (also global) | `Python/cases/magsix/` attitude + trajectory |
| ROCKET6 | `rocket6` | `HYPER6` (SLV; global `HYPER6` is still Hyper6) | `Python/cases/rocket6/` insertion (`mair` 0, `iseed` 1234 = harvest ASC) |
| SAM6 | `sam6` | `MISSILE6` / `AIRCRAFT3` / `ROCKET5` / `RADAR0` | `Python/cases/sam6/` autopilot |
| SRAAM6 | `sraam6` | `MISSILE6` / `TARGET3` | `Python/cases/sraam6/` 1v1 |
| AGM6 | `agm6` | `MISSILE6` / `TARGET3` / `AIRCRAFT3` | `Python/cases/agm6/` free flight + test case. MISSILE6 uses `Agm6Kinematics` (incidence `VBAB=VBEB-TBL·VAEL`, writes `phip`) not stock `Flat6Kinematics`. |

Translate writes scenario-level and per-vehicle `"family"`; parses `WEATHER_DECK`, `SAM_DECK`/`SRBM_DECK`; `GAUSS`/`RAYL` store the mean; `MARKOV` stores 0; glued IF (`IF time >.25`). E2E tests skip if their golden CSV is absent.

**Design / plans (agents):** kernel through handshake executed; AID-web implemented in AircraftIntuitiveDesign (`feature/flow5-v1` `0afb344`); round-trip UI not executed.
1. `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
2. `docs/superpowers/plans/2026-09-04-cadac-kernel-hyper3.md` (done)
3. `docs/superpowers/plans/2026-09-04-cadac-falcon5.md` (done)
4. `docs/superpowers/plans/2026-09-04-cadac-falcon6.md` (done)
5. `docs/superpowers/plans/2026-09-06-cadac-hyper5.md` (done)
6. `docs/superpowers/plans/2026-09-06-cadac-hyper6.md` (done)
7. `docs/superpowers/plans/2026-09-06-cadac-aim5.md` (done) — `input_hori.asc`; introduces family dispatch
8. `docs/superpowers/plans/2026-09-06-cadac-cruise5.md` (done) — `input_1.asc`; family `cruise5`
9. `docs/superpowers/plans/2026-09-06-cadac-magsix.md` (done) — Rotor EOM; `input.asc` attitude
10. `docs/superpowers/plans/2026-09-06-cadac-rocket6.md` (done) — family `rocket6` + type `HYPER6` SLV
11. `docs/superpowers/plans/2026-09-06-cadac-sam6.md` (done) — Flat0 + family `sam6`
12. `docs/superpowers/plans/2026-09-06-cadac-sraam6.md` (done) — `input_1v1.asc`; family `sraam6`
13. `docs/superpowers/plans/2026-09-06-cadac-agm6.md` (done) — free flight; family `agm6`
14. `docs/superpowers/specs/2026-09-08-cadac-python-quality-overhaul-design.md` + `docs/superpowers/plans/2026-09-08-cadac-python-quality-overhaul.md` (done) — parity-safe kernel/EOM quality; e2e 134 passed, 1 skipped (SAM6 RF golden absent)
15. `docs/superpowers/specs/2026-09-11-cadac-workbench-ui-design.md` (approved) — CADAC web workbench + AID web sibling + MISDC/AID aero handshake. Plans: `docs/superpowers/plans/2026-09-11-cadac-catalog.md` (done, merged `22662a1`) → `cadac-workbench.md` (done, merged `5a751dd`) → `cadac-handshake.md` (done) → `aid-web.md` (done, AID `feature/flow5-v1` `0afb344`) → `cadac-roundtrip.md` (plan 5 not executed)

Implementation: Grok subagents + Grok reviewers, TDD, isolated worktree. Family API landed with AIM5; later Task 1 is idempotent.

## Reading order for agents
1. Read this `README.md` (mandatory if present).
2. Read `UPDATES.md` (mandatory) for the change history and current state before working.
3. Do not transcribe C++ arrays/pointers; map CADAC concepts onto Python types.
