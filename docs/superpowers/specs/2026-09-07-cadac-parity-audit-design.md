# CADAC C++ → Python parity audit — design

Date: 2026-09-07
Status: approved (chat). Approach C: hybrid inventory, then Linux g++ goldens for existing JSONC cases. Do not port leftover gaps.

Parent: `docs/superpowers/specs/2026-09-04-cadac-python-design.md`

## Goal

Prove what transferred in the CADAC++ → `cadac` Python migration and what did not, then numerically compare C++ vs Python on the existing representative JSONC cases.

1. Inventory every C++ vehicle type, module, and integer mode against Python.
2. Harvest C++ `plot.csv` goldens on Linux (g++) for those JSONC cases.
3. Run the existing pytest e2e suite against those goldens.
4. Record leftover C++ abilities as gaps. Do not implement them in this work.

## Non-goals

- Porting stubbed or missing modes, vehicles, or I/O.
- E2E on the other ~100 C++ `input*.asc` files.
- Monte Carlo live `rand()`, CADAC Studio `.asc` plot merge, screen/`tabout`/`doc` I/O (inventory only).
- Rewriting CADAC++ for Linux beyond a thin compat shim.
- Changing CSV tolerances to hide numeric divergence.
- SAM6 RF e2e (`tests/e2e/test_sam6_rf.py`) — no JSONC case; leave skip-without-golden.

## Architecture

Two sequential phases, one audit project.

```
C++ sources ──► inventory scanner + mismatch review ──► inventory.json
Python src

C++ sources ──► Linux shim + g++ ──► run matching input.asc ──► goldens/
Python JSONC ──► run_scenario ──► pytest e2e vs goldens
```

Phase A must finish (matrix written) before Phase B harvests families other than the HYPER3 canary. HYPER3 g++ output is checked against the already-committed golden first.

## Components

### Inventory scanner

`Python/tools/cadac_cpp/inventory.py`

Walks all twelve C++ program directories and `Python/src/cadac/`. Writes `Python/tools/cadac_cpp/inventory.json`.

C++ extraction:

- Vehicle types from `set_obj_type` / class names (`CRUISE3`, `HYPER6`, `SATELLITE3`, `RADAR0`, …).
- Modules from `def_*` / `init_*` / execution functions and from MODULES lists in the representative `input*.asc`.
- Integer modes from `switch` / `if` on `mguid`, `mguidance`, `mguide`, `maut`, `mauty`, `mcontrol`, `mprop`, `mseek`, `mseeker`, `mins`, `maero`, `mact`, `mtvc`, `mair`, `matmo`, `mturb`, `mwind`, `mnav`, `mterm`, `mtrack`, `mtarget`, `skr_dyn`, `skr_type`, `minit`, `mroll`, `mrcs_force`, `mrcs_moment`, `mgps`, `mstar`, `mtargeting`, `tgt_option`, `acft_option`, `guid_mid`, `guid_term`. Count only branches that appear in executable `if`/`switch`, not comments.
- Kernel/I/O: events, `look_up`, `integrate`, combus, atmospheres, `GAUSS`/`MARKOV`/`RAYL`, `markov_noise`, options (`scrn`, `events`, `plot`, `doc`, `csv`, `tabout`, `merge`, `comscrn`, `traj`), Monte Carlo `nmonte`.

Python extraction:

- `_VEHICLE_TYPES` and `_VEHICLE_FAMILIES` in `Python/src/cadac/cli.py`.
- Module classes attached to each vehicle.
- Implemented vs `ValueError` vs no-op for each mode.
- Translate: `WEATHER_DECK`, `SAM_DECK`/`SRBM_DECK`, `family`, glued IF, stochastic means.

Each matrix row:

| Field | Meaning |
|---|---|
| `program` | HYPER3, AIM5, … |
| `kind` | `vehicle` \| `module` \| `mode` \| `kernel` \| `e2e` \| `harvest` |
| `name` | type, module, or `maut=24` |
| `cpp` | C++ file / symbol |
| `python` | module path or `null` |
| `status` | `ported` \| `stubbed` \| `missing` \| `deferred` \| `diverged` |
| `note` | human pass: equivalent, slice limit, or drop-out |

Status:

- `ported` — Python implements the C++ behavior for that row.
- `stubbed` — Python raises `ValueError` (or equivalent) for that mode/type.
- `missing` — C++ ability has no Python class, branch, or registry entry.
- `deferred` — design non-goal (Studio merge, live `rand()`, scrn/tabout/doc).
- `diverged` — Python implements the row, but harvested C++ vs Python e2e exceeds CSV tolerances.

A name match is not automatic `ported`. The human pass records conceptual-port equivalence vs a real drop-out.

### Linux C++ runner

`Python/tools/cadac_cpp/` (not under `CADAC_Simulations/`):

- `compat.hpp` — stub `system("pause")` and other Windows-only bits; do not rewrite vehicle math.
- Per-program Makefile compiling that program's `.cpp` with `g++ -std=c++17`.
- Binaries in `Python/tools/cadac_cpp/build/<program>/`.
- `harvest.py` — for each harvest row: backup `input.asc`, copy the mapped ASC onto `input.asc`, run with cwd = the C++ program folder (decks resolve), copy the produced plot CSV to the golden path, restore `input.asc`.

MONTE: harvest with `nmonte==0` / stored means. If the ASC enables MONTE, harvest from a one-off copy with MONTE off. Do not keep live `rand()` goldens.

If g++ fails after the compat pass, skip that family's harvest, add a `kind=harvest` row with `status=missing` and the compiler error in `note`, and leave the pytest skip-without-golden in place. Do not port CADAC++ to Linux beyond the shim.

### E2E harness

Existing files under `Python/tests/e2e/` stay the comparison. Harvesting a golden is what turns a skip into a real test.

Rules (already in those tests):

- `rtol=1e-5`, `atol=max(1e-6, 5e-6*|golden|)`.
- Skip CADAC sentinel `time=-1`.
- Compare columns present on both sides. HYPER3 already compares the full plot grid.
- Python plot records vehicle slot 0 only.

Numeric fail = migration bug. Add a `kind=e2e` row with `status=diverged`. Do not loosen tolerances. Do not port extra modes to make a different ASC pass.

## Harvest table

JSONC `title` is the C++ ASC basename. Copy that file onto `input.asc` in the C++ program directory.

| Python JSONC | C++ directory | C++ ASC | Golden |
|---|---|---|---|
| `Python/cases/hyper3/input_climb.jsonc` | `CADAC_Simulations/HYPER3_250114/HYPER3` | `input_climb.asc` | `tests/e2e/goldens/hyper3/plot1.csv` |
| `Python/cases/falcon5/input_turning_to_IP.jsonc` | `CADAC_Simulations/FALCON5_250116/FALCON5` | `input_turning_to_IP.asc` | `tests/e2e/goldens/falcon5/plot.csv` |
| `Python/cases/falcon6/input_gamma.jsonc` | `CADAC_Simulations/FALCON6_250201/FALCON6` | `input_gamma.asc` | `tests/e2e/goldens/falcon6/plot.csv` |
| `Python/cases/hyper5/input.jsonc` | `CADAC_Simulations/HYPER5_250113/HYPER5` | `input_Demo_4_7_pro_nav.asc` | `tests/e2e/goldens/hyper5/plot.csv` |
| `Python/cases/hyper6/input_climb.jsonc` | `CADAC_Simulations/HYPER6_250125/HYPER6` | `input_climb.asc` | `tests/e2e/goldens/hyper6/plot.csv` |
| `Python/cases/aim5/input_hori.jsonc` | `CADAC_Simulations/AIM5_250114/AIM5` | `input_hori.asc` | `tests/e2e/goldens/aim5/plot.csv` |
| `Python/cases/cruise5/input_1.jsonc` | `CADAC_Simulations/CRUISE5_250115/CRUISE5` | `input_1.asc` | `tests/e2e/goldens/cruise5/plot.csv` |
| `Python/cases/magsix/input.jsonc` | `CADAC_Simulations/MAGSIX_231111/MAGSIX` | `input_attitudeMR1.asc` | `tests/e2e/goldens/magsix/plot.csv` |
| `Python/cases/magsix/input_trajectoryMR1.jsonc` | `CADAC_Simulations/MAGSIX_231111/MAGSIX` | `input_trajectoryMR1.asc` | `tests/e2e/goldens/magsix/trajectory/plot.csv` |
| `Python/cases/rocket6/input.jsonc` | `CADAC_Simulations/ROCKET6_250122/ROCKET6` | `input_insertion.asc` | `tests/e2e/goldens/rocket6/plot.csv` |
| `Python/cases/sam6/input_SAM_autopilot.jsonc` | `CADAC_Simulations/SAM6_250217/SAM6` | `input_SAM_autopilot.asc` | `tests/e2e/goldens/sam6/plot.csv` |
| `Python/cases/sraam6/input_1v1.jsonc` | `CADAC_Simulations/SRAAM6_250130/SRAAM6` | `input_1v1.asc` | `tests/e2e/goldens/sraam6/plot.csv` |
| `Python/cases/agm6/input_freeflight.jsonc` | `CADAC_Simulations/AGM6_250217/AGM6` | `input_3_1 AGM6 Free Flight.asc` | `tests/e2e/goldens/agm6/plot.csv` |
| `Python/cases/agm6/input_testcase.jsonc` | `CADAC_Simulations/AGM6_250217/AGM6` | `input_2_1 AGM6 Test Case.asc` | `tests/e2e/goldens/agm6/test_case_plot.csv` |

HYPER3 golden already exists. Canary: rebuild HYPER3 on g++, compare new `plot1.csv` to `tests/e2e/goldens/hyper3/plot1.csv` with the same CSV tolerances, then harvest the other rows.

C++ plot output name is `plot1.csv` or `plot.csv` depending on merge/vehicle count. `harvest.py` copies whichever file CADAC wrote onto the golden path above.

## C++ programs in scope (inventory)

| Folder | Python family / types |
|---|---|
| `HYPER3_250114` | (none) `CRUISE3` |
| `FALCON5_250116` | (none) `PLANE` |
| `FALCON6_250201` | (none) `PLANE6` |
| `HYPER5_250113` | (none) `HYPER5` / `TARGET3` / `SATELLITE3` |
| `HYPER6_250125` | (none) `HYPER6`; C++ also `SATELLITE3` / `RADAR0` / Ground0 |
| `AIM5_250114` | `aim5` `AIM5` / `AIRCRAFT3` |
| `CRUISE5_250115` | `cruise5` `CRUISE3` / `TARGET3` / `SATELLITE3` |
| `MAGSIX_231111` | `magsix` `ROTOR` |
| `ROCKET6_250122` | `rocket6` `HYPER6` (SLV) |
| `SAM6_250217` | `sam6` `MISSILE6` / `AIRCRAFT3` / `ROCKET5` / `RADAR0` |
| `SRAAM6_250130` | `sraam6` `MISSILE6` / `TARGET3` |
| `AGM6_250217` | `agm6` `MISSILE6` / `TARGET3` / `AIRCRAFT3` |

Known starting gaps (must appear in the matrix, not assumed fixed):

- HYPER6 Radar / Satellite / Ground0 are not registered.
- Many integer modes raise `ValueError` (“unknown maut/mguid/…”).
- Translate stores GAUSS/RAYL means and MARKOV as 0; no live `rand()`.
- Plot CSV is vehicle slot 0 only.
- Most e2e goldens are absent until harvest.

## Error handling

- Inventory mismatch → matrix row, never an auto-fix.
- g++ failure after compat → skip harvest for that program; record the error.
- Python vs C++ numeric fail → pytest fails; `kind=e2e` row `status=diverged`. Module rows stay `ported` if the code exists.
- Missing golden after skip → existing pytest skip remains valid.

## Testing (this work)

- Unit: inventory parser on a fixture snippet of C++ `switch`/`def_*` and a fake Python registry.
- Unit: harvest mapping table (JSONC title → ASC path → golden path).
- Canary: HYPER3 g++ plot vs checked-in golden.
- E2E: `pytest Python/tests/e2e/` after harvest; families without a harvested golden still skip.

## Deliverables

- This spec.
- Implementation plan: `docs/superpowers/plans/2026-09-07-cadac-parity-audit.md`.
- `Python/tools/cadac_cpp/` (scanner, compat, Makefiles, harvest).
- `Python/tools/cadac_cpp/inventory.json`.
- Goldens under `Python/tests/e2e/goldens/`.
- `UPDATES.md` entry; `README.md` only if the architecture paragraph needs an audit note.

No new markdown gap report. The matrix is `inventory.json`. Summarize counts in `UPDATES.md`.
