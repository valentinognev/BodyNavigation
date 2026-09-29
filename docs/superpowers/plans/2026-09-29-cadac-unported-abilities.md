# CADAC Unported Abilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Executor model:** Grok non-fast only (e.g. `grok-4.7-high` / `cursor-grok-4.6-high`). No Fast, Claude, GPT, Composer, or `inherit` unless the user approves that run. Same rule for every implementer and reviewer subagent.

**Goal:** Port every ability in the 2026-09-29 gap catalog into Python `cadac`: current-C++ gaps first (Bands A–C), then legacy abilities that exist only in older Fortran / older C++ and were never carried into `CADAC_Simulations/` (Band D), so Python can run the legacy input examples.

**Architecture:** One sequential priority track across programs (not parallel sub-plans): case-blocking raises/KeyErrors → copy/adapt sibling laws for unused modes → new HYPER6/CRUISE5 sensor modules → CADAC I/O options → legacy-only abilities (old ROCKET6 actuator; new `sraam5` / `rocket3` families; Fortran CRUISE5/SRAAM/AIM5/FALCON6/GHAME modes). Trust this catalog over `inventory.json` (several rows are falsely `ported`). Do not re-audit the tree. Current C++ under `CADAC_Simulations/` is read-only. Older sources are read-only under `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_{FTN,C++}/`.

**Tech Stack:** Python ≥ 3.11, numpy, pytest. Work in `Python/` with `PYTHONPATH=src:tools`. Unit tests under `Python/tests/unit/`. E2E under `Python/tests/e2e/` (match existing `test_*` + `goldens/` pattern). Deck translation via `cadac.io.translate` / `load_scenario`.

**Spec:** Gap catalog in the user request that commissioned this plan (2026-09-29 five-way source comparison + Band D legacy catalog). This plan is the executable decomposition of that catalog. Prefer cited C++/Fortran files over inventory.

## Global Constraints

- Python package is `Python/src/cadac`. Never edit `CADAC_Simulations/` or the Books `CADAC4_Enhance` trees.
- TDD: write the failing unit test first, watch it fail, minimal source-faithful port, watch it pass. No production code without a failing test.
- Family dispatch today: `cruise5`, `aim5`, `sam6`, `sraam6`, `agm6`, `magsix`, `rocket6`. HYPER3 / HYPER5 / HYPER6 / FALCON5 / FALCON6 use family `None`. Band D **adds** families `sraam5` and `rocket3` (do not fold into `sraam6` / `rocket6`).
- CRUISE5 `mguidance` 33, 40, and 66 are already ported (uncommitted as of plan writing). Do **not** re-implement them. `mguidance` 70 remains (Band B).
- FALCON5 `mguidance` 6 and 60 are **not** C++ branches (Python raises; C++ would zero commands). Optional note only — do not port as features.
- Programs with **zero gaps vs current C++** (Bands A–C invent no work): **HYPER3**, **AIM5**, **SRAAM6**. Band D still adds legacy abilities onto those programs (and new vehicle families).
- Do not rename CADAC type tokens or family strings. Band D tasks that need a legacy deck **add** a new JSONC under `Python/cases/<family>/` via `translate_scenario_asc` / `deck_asc_to_jsonc` — never overwrite an existing JSONC; never copy by editing `CADAC_Simulations`.
- Band D every task: (1) unit test asserting Fortran/old-C++ equation or mode dispatch on planted state; (2) named legacy `*.asc` → destination JSONC (or “covered by Task N deck”); (3) e2e under `Python/tests/e2e/` — old-C++ → `legacy-binary` golden; Fortran → `python-golden` after unit lock (optional `gfortran` harvest only if a Makefile exists — none found under SRAAM5/ROCKET3); (4) Done when unit passes, JSONC loads with `load_scenario`, e2e command named.
- Commit steps run only with explicit user approval (agent-permissions). Plan text still shows the intended commit message for when approved.
- Port formulas line-for-line in behavior; use existing `cadac_matmul` / `cadac_inverse` / frame helpers / `cadac.stoch` where applicable. ROCKET6 vs HYPER6 INS gauss draw order may differ — adapt ROCKET6, do not assume identical sequences. FALCON6 Dryden: transcribe Fortran `G2TURB`, do not copy AGM6’s gauss sequence blindly.

## Priority rationale (overview)

1. **Shipped-case blockers** — A `Python/cases/...` file already sets the mode; `execute` raises or KeyErrors. Highest user-visible value; unlocks e2e later.
2. **Small unused mode branches** — C++ law or a sibling vehicle’s Python law already exists; copy/adapt, not new physics.
3. **Whole missing modules** — seeker, datalink, GPS, startrack, RCS, intercept. Larger; block several HYPER6/CRUISE5 cases (those blockers sit in band 1 when a case needs them).
4. **Shared executive outputs** — MONTE, markov_noise, tabout, traj, doc, stat, scrn, comscrn, merge. No equation changes; CADAC I/O only. One track after current-C++ physics gaps.
5. **Legacy abilities absent from current C++** — older Fortran / older C++ features never carried into `CADAC_Simulations/`. New Python families (`sraam5`, `rocket3`) and mode ports so legacy decks run. After Band C (or in parallel only if A–C physics for that family already green).

## File structure map

| Area | Touch | Role |
|---|---|---|
| `vehicles/flat3/falcon5/{guidance,control}.py` | Modify | Wire missing `mguidance`/`mcontrol` dispatch using existing helpers |
| `vehicles/round3/cruise5/{guidance,control,seeker}.py` | Modify | Arc guidance, control allowlist, seeker 1→3; Band D: terrain, scene seeker, GPS, INS, MTURN/MAUT/MROLL, MAIR |
| `vehicles/round3/hyper5/{guidance,control,intercept}.py` | Modify | Guidance mode wiring, mcontrol 46, wp_alt kill |
| `vehicles/flat6/falcon6/control.py` | Modify | `mautp` 2/3/5; port `control_altitude` / call `control_normal_accel` |
| `eom/flat6.py` + optional `falcon6/environment.py` | Modify / Create | Wind, mfreeze env; FALCON6-only `mguid==6` trcode hook; Band D Dryden + tabular weather |
| `vehicles/flat6/agm6/guidance.py` | Modify | term digit 5, mid digit 2, combined mid\|term |
| `vehicles/flat6/sam6/{control,tvc,forces,rcs,guidance}.py` | Modify | maut=4, mtvc, RCS, mid digit 3 |
| `vehicles/round6/hyper6/` + `vehicle.py` | Create/Modify | GPS, startrack, seeker, datalink, RCS, intercept; prop/INS/control; Band D tabular MATMO/MWIND |
| `vehicles/round6/rocket6/{tvc,rcs,actuator}.py` + `eom/round6.py` | Modify / Create | mtvc 1/3, RCS, matmo=2, mwind=1; Band D fin actuator from old C++ |
| `eom/rotor.py` | Modify | MAGSIX mwind 1/2 |
| `cli.py`, `io/translate.py`, `io/scenario.py`, `stoch.py`, new `io/` writers | Modify/Create | MONTE loop + CADAC output files; Band D Fortran→JSONC translation for new cases |
| `vehicles/flat5/sraam5/` (or `flat3`-style pseudo-5DOF path agreed in Task 53) | Create | New family `sraam5` — MODULE.FOR A1–D2, S1/S2/S4, C1/C2, G1/G2/G4 |
| `vehicles/flat6/sraam6/{seeker,intercept,target}.py` | Modify | Band D AI radar S2, SHAZAM, DTCT/DBTC, MINIT LAR/CIRCLE |
| `vehicles/flat3/aim5/` aircraft/target init | Modify | Band D MTARG=1 polar init |
| `vehicles/round3/hyper3/` environment | Modify | Band D tabular atmosphere MAIR=1 (GHAME3 digits) |
| `vehicles/round3/rocket3/` | Create | New family `rocket3` — Fortran A1/A2/A3/G2/D1 multi-stage ascent |
| `Python/cases/{sraam5,rocket3,...}/` | Create (additive) | Translated legacy ASC → JSONC; never overwrite existing |

---

## Band A — Shipped-case blockers

### Task 1: FALCON5 mcontrol 16 and 36 (altitude-hold pairs)

**Why first:** `Python/cases/falcon5/input_mcontrol_16.jsonc` and `input_mcontrol_36.jsonc` set these modes today; `control()` raises.

**Files:**
- Modify: `Python/src/cadac/vehicles/flat3/falcon5/control.py` (`execute` allowlist + branches)
- Test: `Python/tests/unit/test_falcon5_mcontrol.py` (create or extend)
- C++: `CADAC_Simulations/FALCON5_250116/FALCON5/plane_modules.cpp` — `mcontrol==16` / `==36` compose `control_heading` / bank on `phicx` + `control_altitude` + `control_load`
- Python helpers already present: `control_heading`, `control_bank`, `control_altitude`, `control_load` (same file). Mirror HYPER5/`Cruise5` wiring for 16/36.

**Interfaces:**
- Consumes: existing Falcon5 control helpers.
- Produces: `mcontrol in {16, 36}` runs without raise; writes `phimvx`, `alphax`, `ancomx`, `TBG`.

- [ ] **Step 1: Write the failing test**

```python
def test_falcon5_mcontrol_16_sets_altitude_hold_alphax():
    veh = _falcon5()
    veh.store.set("mcontrol", 16)
    veh.store.set("psivlcx", 10.0)
    veh.store.set("altcom", 5000.0)
    _exec(veh, "control")
    assert veh.store.get("alphax") != 0.0 or veh.store.get("ancomx") != 0.0

def test_falcon5_mcontrol_36_banks_then_altitude_hold():
    veh = _falcon5()
    veh.store.set("mcontrol", 36)
    veh.store.set("phicx", 15.0)
    veh.store.set("altcom", 5000.0)
    _exec(veh, "control")
    assert "TBG" in veh.store
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_falcon5_mcontrol.py::test_falcon5_mcontrol_16_sets_altitude_hold_alphax tests/unit/test_falcon5_mcontrol.py::test_falcon5_mcontrol_36_banks_then_altitude_hold -v`

Expected: FAIL with `unknown mcontrol 16` / `36`.

- [ ] **Step 3: Write minimal implementation**

Extend `execute` allowlist; add `if mcontrol == 16:` / `36:` matching C++ (heading+alt / bank phicx+alt). Reuse existing helpers; do not invent new loops.

- [ ] **Step 4: Run tests to verify they pass**

Same pytest command → PASS.

- [ ] **Step 5: Commit (only with user approval)**

```bash
git add Python/src/cadac/vehicles/flat3/falcon5/control.py Python/tests/unit/test_falcon5_mcontrol.py
git commit -m "feat(falcon5): port mcontrol 16 and 36 altitude-hold pairs"
```

**Done when:** both tests pass; cases that set `mcontrol` 16/36 no longer raise in `control`.

---

### Task 2: CRUISE5 mseeker 1 and 3

**Why:** `input.jsonc`, `input_1_2`, `input_1_3` events enable seeker; stub raises if `mseeker≠0`. Blocks the already-ported `mguidance` 66 pro-nav path those cases switch to.

**Files:**
- Modify: `Python/src/cadac/vehicles/round3/cruise5/seeker.py`
- Test: `Python/tests/unit/test_cruise5_seeker.py`
- C++: `Cruise::seeker` / `seeker_grnd_ranges` in `CADAC_Simulations/CRUISE5_250115/CRUISE5/cruise_modules.cpp`
- Sibling pattern: `Python/src/cadac/vehicles/round3/hyper5/seeker.py` (`Hyper5Seeker` — acquire within `acq_range` → `mseeker=3`; tracking writes `WOEB`, `closing_speed`, `UTBB`, angles)

**Interfaces:**
- Consumes: combus target packets (`sbeg`), cruise kinematics.
- Produces: `mseeker` 1→3 transition; mode 3 fields used by `guidance_pronav` / intercept.

- [ ] **Step 1: Write the failing test**

```python
def test_cruise5_mseeker_1_acquires_within_acq_range():
    veh, combus = _cruise5_with_target(range_m=500.0)
    veh.store.set("mseeker", 1)
    veh.store.set("acq_range", 1000.0)
    _exec(veh, "seeker", combus=combus)
    assert veh.store.get("mseeker") == 3
    assert veh.store.get("acquisition") == 1

def test_cruise5_mseeker_3_writes_woeb_and_closing_speed():
    veh, combus = _cruise5_with_target(range_m=500.0)
    veh.store.set("mseeker", 3)
    veh.store.set("acquisition", 1)
    _exec(veh, "seeker", combus=combus)
    assert "WOEB" in veh.store
    assert "closing_speed" in veh.store
```

- [ ] **Step 2: Run — expect FAIL** `unknown mseeker`

- [ ] **Step 3: Port** `seeker_grnd_ranges` + acquire/track branches from C++ / HYPER5 pattern into `Cruise5Seeker`.

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit (approval)** `feat(cruise5): port mseeker acquire and track`

**Done when:** both tests pass; seeker no longer stubs nonzero modes.

---

### Task 3: FALCON6 maut pitch digit 3 (`control_normal_accel`)

**Why:** `input_4WP` uses codes like `maut` 33; whitelist `(1, 24, 30, 40)` raises.

**Files:**
- Modify: `Python/src/cadac/vehicles/flat6/falcon6/control.py`
- Test: `Python/tests/unit/test_falcon6_maut.py`
- C++: `Plane::control` `mautp==3` + `Plane::control_normal_accel` in `CADAC_Simulations/FALCON6_250201/FALCON6/control.cpp`
- Sibling math: `Hyper6Control.control_normal_accel` / CRUISE load loop as reference, but transcribe FALCON6 C++ gains/state names.

**Interfaces:**
- Produces: `control_normal_accel(vehicle, ancomx, int_step) -> delecx`; `mautp==3` path allowed (e.g. 33).

- [ ] **Step 1: Failing test**

```python
def test_falcon6_maut_33_calls_normal_accel():
    veh = _falcon6()
    veh.store.set("maut", 33)
    veh.store.set("ancomx", 1.0)
    _exec(veh, "control")
    assert "delecx" in veh.store or veh.store.get("delecx") is not None
```

(Adjust assert to the store field name C++ writes for elevator command — `delecx` / actuator inputs already used by Falcon6.)

- [ ] **Step 2: FAIL** `unknown maut 33`

- [ ] **Step 3:** Widen whitelist to digit decode (keep rejecting truly unknown digits); add `control_normal_accel`; `if mautp == 3: delecx = ...`

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit** `feat(falcon6): port mautp=3 normal-accel autopilot`

**Done when:** `maut` with pitch digit 3 executes without raise.

---

### Task 4: FALCON6 maut pitch digit 5 (altitude hold → normal accel)

**Why:** `input_3waypoint`, `input_4WP`, `input_alt_head`, `input_altitude_steps` use 35/45. Shares new `control_altitude`; one task covers both call paths.

**Files:**
- Modify: `Python/src/cadac/vehicles/flat6/falcon6/control.py`
- Test: same `test_falcon6_maut.py`
- C++: `mautp==5` → `ancomx=control_altitude(altcom)` then `control_normal_accel` (`control.cpp`)
- Depends on: Task 3 (`control_normal_accel`).

**Interfaces:**
- Produces: `control_altitude(vehicle, altcom) -> ancomx`; `mautp==5` for 35 and 45.

- [ ] **Step 1: Failing tests** for `maut` 35 and 45 (both must leave finite `delecx` / updated `ancomx`).

- [ ] **Step 2: FAIL** `unknown maut 35` (or 45).

- [ ] **Step 3:** Port `control_altitude` from FALCON6 C++; wire `mautp==5`.

- [ ] **Step 4: PASS** both paths.

- [ ] **Step 5: Commit** `feat(falcon6): port mautp=5 altitude-hold autopilot`

**Done when:** 35 and 45 pass unit tests.

---

### Task 5: AGM6 mguid terminal digit 5 (`guidance_term_pronav`)

**Why:** `input_4_7`, `input_5_1`, `input_5_2` set term digit 5.

**Files:**
- Modify: `Python/src/cadac/vehicles/flat6/agm6/guidance.py`
- Test: `Python/tests/unit/test_agm6_guidance_term_pronav.py`
- C++: `Missile::guidance_term_pronav` in `CADAC_Simulations/AGM6_250217/AGM6/guidance.cpp` (`guid_term==5`)
- Sibling: SAM6 `guidance_term_pronav` exists but AGM6 C++ is kinematic LOS-rate **without** compensation — do not call `guidance_term_comp`.

**Interfaces:**
- Produces: `guidance_term_pronav(self, vehicle) -> acbx`; dispatch when `guid_term == 5` (alone or with mid — if mid also set, Task 8 handles combine; for pure `mguid==5` handle here).

- [ ] **Step 1: Failing test** `test_agm6_mguid_5_term_pronav_sets_ancomx` with `mguid=5`.

- [ ] **Step 2: FAIL** `unknown mguid 5`

- [ ] **Step 3:** Add method from C++; `elif guid_term == 5:` (order with mid per C++: mid may run first when both digits set — for `mguid==5` only term runs).

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit** `feat(agm6): port guidance_term_pronav for mguid digit 5`

**Done when:** `mguid=5` no longer raises; `alcomx`/`ancomx` set.

---

### Task 6: HYPER6 mprop 3 and 4 (rocket LTG / constant thrust)

**Why:** `input_TV`, `input`, `input_intercept`, `input_head-on_auto`.

**Files:**
- Modify: `Python/src/cadac/vehicles/round6/hyper6/propulsion.py`
- Test: `Python/tests/unit/test_hyper6_mprop.py`
- C++: HYPER6 propulsion in `CADAC_Simulations/HYPER6_250125/HYPER6/` (mprop 3/4, exo MOI)
- Sibling: `Python/src/cadac/vehicles/round6/rocket6/propulsion.py` — adapt ROCKET6 LTG/constant-thrust branches; do not assume identical deck field names without checking HYPER6 C++.

**Interfaces:**
- Produces: `mprop in {0,1,2,3,4}` allowed; 3/4 update thrust / mass / MOI per C++.

- [ ] **Step 1:** `test_hyper6_mprop_3_ltg_thrust` and `test_hyper6_mprop_4_constant_thrust` — FAIL today.
- [ ] **Step 2–4:** Port; PASS.
- [ ] **Step 5: Commit** `feat(hyper6): port mprop 3 and 4 rocket propulsion`

**Done when:** both modes execute without `unknown mprop`.

---

### Task 7: HYPER6 mins=1 INS

**Why:** Cape / arc / head-on cases.

**Files:**
- Modify: `Python/src/cadac/vehicles/round6/hyper6/ins.py`
- Test: `Python/tests/unit/test_hyper6_ins_mins1.py`
- C++: HYPER6 `ins.cpp`; sibling `Rocket6Ins` (`mins==1`) — **adapt**; verify HYPER6 gauss draw order against HYPER6 C++ (do not blindly copy ROCKET6 sequence).

**Interfaces:**
- Produces: instrumented INS with GPS/star update hooks when those modules exist (Tasks 8–9); until then, mins=1 must at least not raise and write INS outputs C++ writes with updates off.

- [ ] **Step 1:** `test_hyper6_mins_1_initialize_and_execute` — FAIL `unknown mins 1`.
- [ ] **Step 3:** Port `init_ins` / `ins` for mins=1.
- [ ] **Step 5: Commit** `feat(hyper6): port mins=1 instrumented INS`

**Done when:** `mins=1` initialize+execute pass.

---

### Task 8: HYPER6 GPS module (mgps 0–3)

**Why:** `input_Cape_ascent_GPS`, head-on. Module name currently skipped (not on `Hyper6.modules`).

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/gps.py` (`Hyper6Gps`)
- Modify: `Python/src/cadac/vehicles/round6/hyper6/vehicle.py` — insert GPS in C++ module order
- Test: `Python/tests/unit/test_hyper6_gps.py`
- C++: HYPER6 `gps.cpp`; sibling `Rocket6Gps` — adapt Kalman/SV init; same `cadac_matmul`/`cadac_inverse` rules.

**Interfaces:**
- Consumes: INS state; `mgps` data fields from deck.
- Produces: module name `"gps"`; `SXH`/`VXH`/`CXH` etc. for INS updates.

- [ ] **Step 1:** `test_hyper6_has_gps_module` + `test_hyper6_mgps_2_one_step` — FAIL (no module / AttributeError).
- [ ] **Step 3:** Create module; register on vehicle.
- [ ] **Step 5: Commit** `feat(hyper6): add GPS module`

**Done when:** scenario module list `"gps"` runs; mgps 0–3 do not raise.

---

### Task 9: HYPER6 Startrack (mstar 0–3)

**Why:** head-on cases.

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/startrack.py`
- Modify: `vehicle.py` module list
- Test: `Python/tests/unit/test_hyper6_startrack.py`
- C++: HYPER6 `startrack.cpp`; sibling `Rocket6Startrack`.

- [ ] **Step 1–5:** Same TDD pattern as GPS. Commit `feat(hyper6): add startrack module`

**Done when:** `mstar` 0–3 execute; module on vehicle.

---

### Task 10: HYPER6 RF seeker (mseek 2–5)

**Why:** intercept, head-on, §10.2.2.

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/seeker.py` (vehicle RF seeker — distinct from `Hyper6RadarSeeker`)
- Modify: `vehicle.py`
- Test: `Python/tests/unit/test_hyper6_seeker.py`
- C++: HYPER6 seeker module in `CADAC_Simulations/HYPER6_250125/HYPER6/`

**Interfaces:**
- Produces: `mseek`, LOS/body angles, fields `guidance_pronav` already reads (`mseek` KeyError today).

- [ ] **Step 1:** `test_hyper6_mseek_3_tracking_fields` — FAIL.
- [ ] **Step 3:** Port seeker; register.
- [ ] **Step 5: Commit** `feat(hyper6): add RF seeker module`

**Done when:** seeker fields exist; mseek 2–5 do not raise.

---

### Task 11: HYPER6 datalink (mnav / STCII)

**Why:** intercept/head-on module lists; unblocks mguide 8.

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/datalink.py`
- Modify: `vehicle.py`
- Test: `Python/tests/unit/test_hyper6_datalink.py`
- C++: `CADAC_Simulations/HYPER6_250125/HYPER6/datalink.cpp` — `Hyper::datalink` reads radar combus tracks into `STCII`/`VTCII`, sets `mnav=3` on change.

**Interfaces:**
- Consumes: radar combus packets (`STCIIx`/`VTCIIx`).
- Produces: `mnav`, `STCII`, `VTCII` on Hyper vehicle store.

- [ ] **Step 1:** `test_hyper6_datalink_sets_mnav_on_track_change`
- [ ] **Step 3–5:** Port + commit `feat(hyper6): add datalink module`

**Done when:** datalink execute writes `STCII`/`mnav`.

---

### Task 12: HYPER6 RCS (mrcs_moment type 1/2, mrcs_force 1)

**Why:** TV, intercept, head-on.

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/rcs.py`
- Modify: `vehicle.py`, ensure `Hyper6Forces` already adds `FMRCS` (it does)
- Test: `Python/tests/unit/test_hyper6_rcs.py`
- C++: HYPER6 `rcs.cpp`; sibling ROCKET6/SAM6 RCS for structure.

- [ ] **Step 1–5:** TDD port. Commit `feat(hyper6): add RCS module`

**Done when:** nonzero mrcs modes write `FMRCS` without raise.

---

### Task 13: HYPER6 intercept (miss / closest approach)

**Why:** `input_intercept` lists intercept module.

**Files:**
- Create: `Python/src/cadac/vehicles/round6/hyper6/intercept.py`
- Modify: `vehicle.py`
- Test: `Python/tests/unit/test_hyper6_intercept.py`
- C++: HYPER6 intercept; sibling `Rocket6Intercept`.

- [ ] **Step 1–5:** Commit `feat(hyper6): add intercept module`

**Done when:** module runs; miss/hit diagnostics defined.

---

### Task 14: HYPER6 maut=1 roll-only

**Why:** §10.1.2 / 10.1.3 cases.

**Files:**
- Modify: `Python/src/cadac/vehicles/round6/hyper6/control.py` (gate currently rejects `maut` 1)
- Test: `Python/tests/unit/test_hyper6_maut.py`
- C++: HYPER6 control — roll-only when `maut==1`

- [ ] **Step 1:** `test_hyper6_maut_1_roll_only` — FAIL `unknown maut 1`
- [ ] **Step 3:** Allow `mauty==0 and mautp==1` (or exact C++ decode); run roll path only.
- [ ] **Step 5: Commit** `feat(hyper6): allow maut=1 roll-only`

**Done when:** maut 1 executes.

---

### Task 15: HYPER6 mautp=2 pitch-rate SAS

**Why:** `10_1_4` maut=22; `10_2_2` maut=2. `control_pitch_rate` exists; gate rejects `mautp=2`.

**Files:**
- Modify: `hyper6/control.py`
- Test: `test_hyper6_maut.py`
- C++: `mautp==2` → `control_pitch_rate`

- [ ] **Step 1:** `test_hyper6_maut_22_pitch_rate` and `test_hyper6_maut_2_pitch_rate`
- [ ] **Step 3:** Extend `pitch_ok` to include 2; call existing `control_pitch_rate`.
- [ ] **Step 5: Commit** `feat(hyper6): enable mautp=2 pitch-rate SAS`

**Done when:** maut 2 and 22 pass.

---

### Task 16: HYPER6 mguide=6 pro-nav field wiring

**Why:** Law present; KeyError `mseek` without seeker/datalink fields. Cases: intercept, head-on. **Depends on Tasks 10–11.**

**Files:**
- Modify: `Python/src/cadac/vehicles/round6/hyper6/guidance.py` only if dispatch still assumes missing names; primarily verify seeker/datalink fields satisfy `guidance_pronav`
- Test: `Python/tests/unit/test_hyper6_mguide6.py`

- [ ] **Step 1:** Full one-step with mguide=6 + seeker tracking fields — must not KeyError.
- [ ] **Step 3:** Fill any remaining missing stores; do not rewrite the law.
- [ ] **Step 5: Commit** `fix(hyper6): mguide=6 runs with seeker fields`

**Done when:** mguide=6 execute completes with seeker on vehicle.

---

### Task 17: HYPER6 mguide=8 glideslope (STCII)

**Why:** `input_intercept`. Law present; KeyError `STCII`. **Depends on Task 11.**

**Files:**
- Modify: `guidance.py` only if needed; datalink supplies `STCII`
- Test: `Python/tests/unit/test_hyper6_mguide8.py`

- [ ] **Step 1–5:** Commit `fix(hyper6): mguide=8 runs with datalink STCII`

**Done when:** mguide=8 no KeyError.

---

### Task 18: ROCKET6 mrcs_moment=23 (Schmitt incidence)

**Why:** `input_ballistic` sets 23.

**Files:**
- Modify: `Python/src/cadac/vehicles/round6/rocket6/rcs.py`
- Test: `Python/tests/unit/test_rocket6_rcs.py`
- C++: ROCKET6 RCS `mrcs_moment==23` Schmitt on incidence

- [ ] **Step 1:** `test_rocket6_mrcs_moment_23_schmitt` — FAIL allowlist.
- [ ] **Step 3:** Port type/mode 23 branch; allow in set.
- [ ] **Step 5: Commit** `feat(rocket6): port mrcs_moment=23 Schmitt incidence`

**Done when:** mode 23 writes `FMRCS` without raise.

---

## Band B — Small unused mode branches (copy/adapt)

### Task 19: FALCON5 mguidance 43 (point lateral + line pitch)

**Files:** `falcon5/guidance.py`; C++ `plane_modules.cpp` mguidance 43; helpers `guidance_point` / `guidance_line` already exist.
**Test:** `test_falcon5_mguidance.py::test_mguidance_43_point_lateral_line_pitch`
**Done when:** mode 43 sets `phicx`/`thtvgcx` (or C++ equivalents) without raise.
**Commit:** `feat(falcon5): port mguidance 43`

---

### Task 20: FALCON5 mcontrol 0, 3, 4, 6, 40

One task per mode (reviewer isolation), same file `falcon5/control.py`, same helpers as HYPER5 (which already implements these). Suggested order and tests:

| Subtask | Mode | C++ behavior | Test name |
|---|---|---|---|
| 20a | 0 | zero bank/alpha | `test_falcon5_mcontrol_0_zeros_commands` |
| 20b | 3 | bank + open-loop `alphacx` | `test_falcon5_mcontrol_3_bank_openloop_alpha` |
| 20c | 4 | load-factor AP on `ancomx` | `test_falcon5_mcontrol_4_load` |
| 20d | 6 | altitude hold → load | `test_falcon5_mcontrol_6_altitude_hold` |
| 20e | 40 | lateral accel → bank | `test_falcon5_mcontrol_40_lateral` |

Implement as **five sequential mini-tasks** (each: fail test → wire branch → pass → commit). Copy HYPER5 `control.py` dispatch text, adjust flat3 state names (`psivlcx` vs `psivgcx`) from FALCON5 C++.

**Note:** Do not port mguidance 6/60.

---

### Task 21: CRUISE5 mguidance 70 (`guidance_arc`)

**Files:** `cruise5/guidance.py` — add `guidance_arc` + dispatch; C++ `Cruise::guidance_arc`; pattern `Hyper5Guidance.guidance_arc`.
**Test:** `test_cruise5_mguidance.py::test_mguidance_70_arc_lateral_bank`
**Pairs with:** mcontrol 36 (Task 22).
**Commit:** `feat(cruise5): port mguidance 70 guidance_arc`
**Done when:** mode 70 returns bank command; no raise.

---

### Task 22: CRUISE5 mcontrol 3, 4, 6, 16, 36, 40

Allowlist today `(0, 1, 10, 11, 44, 46)`. Helpers already in `cruise5/control.py`. Five/six mini-tasks like FALCON5 Task 20; C++ `cruise_modules.cpp`. Prefer copying cruise C++ order (same as HYPER5 control wiring).

**Tests:** `test_cruise5_mcontrol_{3,4,6,16,36,40}`
**Done when:** each mode in allowlist and executes.
**Note:** No shipped cruise5 case; 36 pairs with guidance 70.

---

### Task 23: HYPER5 mguidance 30, 33, 40, 43

**Files:** `hyper5/guidance.py` — laws `guidance_line` / `guidance_point` exist; wire modes like FALCON5/CRUISE5.
**C++:** `hyper_modules.cpp`
**Tests:** `test_hyper5_mguidance_{30,33,40,43}` — one failing test per mode (four mini-tasks).
**Done when:** each mode dispatches without raise.

---

### Task 24: HYPER5 mcontrol 46

**Files:** `hyper5/control.py` — `_ALLOWED_MCONTROL` omits 46; CRUISE5/FALCON5 already have 46 (lateral + altitude + load).
**Test:** `test_hyper5_mcontrol_46`
**Done when:** 46 in allowlist and matches CRUISE5 composition.

---

### Task 25: HYPER5 intercept terminal for mguidance 33 and 43

**Files:** `hyper5/intercept.py` — add `alt<=wp_alt` kill when `mguidance in (33, 43)` per C++ `hyper_modules.cpp` ~2225.
**Test:** `test_hyper5_intercept_wp_alt_terminates_33`
**Done when:** health/combus cleared when `alt<=wp_alt` for those modes.

---

### Task 26: FALCON6 maut pitch digit 2 (`control_pitch_rate`)

**Files:** `falcon6/control.py` — method exists, not called; whitelist rejects.
**C++:** `mautp==2`
**Test:** `test_falcon6_maut_pitch_digit_2`
**Done when:** e.g. maut 32 calls `control_pitch_rate`.

---

### Task 27: FALCON6 mwind 1 and 2

**Files:** Prefer shared approach with MAGSIX: implement wind laws in a way both can call, or port into `Flat6Environment` carefully.
**C++:** `FALCON6_250201/FALCON6/environment.cpp` `mwind==1/2`; MAGSIX same laws in `environment.cpp`.
**Test:** `test_flat6_mwind_1_constant`, `test_flat6_mwind_2_shear`
**Constraint:** SAM6/SRAAM6/AGM6 must keep `mwind==0` behavior unchanged (same numerical results when mwind=0).
**Done when:** mwind 1/2 set VAES/wind without raise; mwind=0 regression still passes.

---

### Task 28: FALCON6 mguid==6 environment termination (trcode 2/3)

**Files:** Prefer **falcon6-only hook** (subclass or vehicle-local environment), not blind edit of shared `Flat6Environment` that would change SAM6/SRAAM6/AGM6.
**Confirm:** C++ lives in FALCON6 `environment.cpp` (`mguid==6` → low `vmach`/`pdynmc`). Round6 already has analogous block for HYPER — Flat6 shared base currently lacks it.
**Test:** `test_falcon6_env_mguid6_sets_trcode_2_on_low_mach`
**Done when:** FALCON6-only; other Flat6 families unchanged.

---

### Task 29: FALCON6 mfreeze environment (vmach/pdynmc)

**Files:** falcon6 env hook or `Flat6Environment` freeze block matching C++ `mfreeze_environ` / `vmachf` / `pdynmcf`. Round6 already freezes env; Flat6 newton/propulsion freeze exist.
**Test:** `test_falcon6_mfreeze_holds_vmach_pdynmc`
**Done when:** with `mfreeze=1`, env outputs stick; `mfreeze=0` clears.

---

### Task 30: AGM6 mguid mid digit 2 (`guidance_mid_line`)

**Files:** `agm6/guidance.py`; C++ `guidance_mid_line`.
**Test:** `test_agm6_mguid_mid_2`
**Done when:** `guid_mid==2` runs without raise.

---

### Task 31: AGM6 combined mid|term codes (26, 36, 46)

**Files:** `agm6/guidance.py` — C++ runs **both** mid and term branches; Python only exact 30/40/6.
**Test:** `test_agm6_mguid_26_runs_mid_and_term` (and 36/46)
**Depends on:** Tasks 5 and 30.
**Done when:** 26/36/46 execute both laws; cases that only switch 30↔6 / 40↔6 stay green.

---

### Task 32: SAM6 maut=4 (rate+accel + RCS command copies)

**Files:** `sam6/control.py` (currently raises on 4); C++ SAM6 control.
**Test:** `test_sam6_maut_4`
**Done when:** maut=4 runs and copies RCS commands per C++.

---

### Task 33: SAM6 mtvc 1/2/3

**Files:** `sam6/tvc.py`, `sam6/forces.py`; sibling ROCKET6 `mtvc==2`.
**Test:** `test_sam6_mtvc_2` (and 1/3 if C++ distinct)
**Done when:** mtvc≠0 no longer raises; forces accept TVC.

---

### Task 34: SAM6 mrcs_moment / mrcs_force nonzero

**Files:** `sam6/rcs.py`; C++ SAM6 rcs.
**Test:** `test_sam6_rcs_nonzero`
**Done when:** nonzero modes write `FMRCS`.

---

### Task 35: SAM6 mguide mid digit 3 (`guidance_mid_pronav` to radar IP)

**Priority:** lowest SAM6 (C++ comment: unused). Still a real branch.
**Files:** `sam6/guidance.py`
**Test:** `test_sam6_guid_mid_3`
**Done when:** mid digit 3 does not raise.

---

### Task 36: HYPER6 mguide=7 AGL (STBIK)

**Files:** Ensure AGL law gets `STBIK` (define/write from kinematics or guidance init per C++). No shipped JSONC.
**Test:** `test_hyper6_mguide_7_agl`
**Done when:** no KeyError `STBIK`.

---

### Task 37: HYPER6 mair constant wind (…1) and shear (…2)

**Files:** `Python/src/cadac/eom/round6.py` — today allows mair 0, 12, 100 only.
**C++:** HYPER6 environment wind digits.
**Test:** `test_round6_mair_wind_digit_1`, `test_round6_mair_shear_digit_2`
**Constraint:** extend existing weather/mair decode; do not fork a second atmosphere.
**Done when:** wind digits run; existing 0/12/100 unchanged.

---

### Task 38: ROCKET6 mtvc=1 (no dynamics) and mtvc=3 (2nd-order + online gain)

**Files:** `rocket6/tvc.py` (today 0 or 2 only); C++ ROCKET6 TVC.
**Tests:** `test_rocket6_mtvc_1`, `test_rocket6_mtvc_3` — two mini-tasks.
**Done when:** both modes pass.

---

### Task 39: ROCKET6 mrcs_force 1 and 2; mrcs_moment type 1 (codes 1x)

**Files:** `rocket6/rcs.py`
**Tests:** separate mini-tasks per force mode and moment type 1.
**Done when:** force≠0 and moment type 1 no longer raise.

---

### Task 40: ROCKET6 matmo=2 tabular atmosphere (mair=2xx)

**Files:** `eom/round6.py` — extend allowed mair set; reuse weather-deck path used by other codes.
**Test:** `test_round6_matmo_2_tabular`
**Done when:** mair 2xx selects tabular atmos without raise.

---

### Task 41: ROCKET6 mwind=1 constant wind alone

**Files:** `eom/round6.py` mair decode allowlist.
**Test:** `test_round6_mwind_1_alone`
**Done when:** constant-wind-only mair accepted.

---

### Task 42: MAGSIX mwind 1 and 2

**Files:** `Python/src/cadac/eom/rotor.py` `RotorEnvironment`
**C++:** `MAGSIX_231111/MAGSIX/environment.cpp`
**Share approach with Task 27** (same wind laws). May extract a tiny helper under `cadac/env/` if both Flat6 and Rotor need it — only if it reduces duplication without new physics.
**Tests:** `test_magsix_mwind_1`, `test_magsix_mwind_2`
**Done when:** both modes run; mwind=0 unchanged.

---

## Band C — Shared executive outputs (after physics)

All options are parsed into `RunConfig.options` (see `OPTION_KEYS` in `cadac/io/scenario.py`) and currently ignored by `run_scenario`. MONTE is skipped entirely by `parse_scenario_asc` (`else: i += 1`). C++ reference: each program’s `execution.cpp`.

### Task 43: Parse MONTE + outer Monte Carlo loop

**Files:**
- Modify: `Python/src/cadac/io/translate.py` (`parse_scenario_asc` — `MONTE n iseed`), `scenario.py` (`nmonte` on `RunConfig`), `cli.py` `run_scenario`
- Test: `Python/tests/unit/test_monte_loop.py`
- Cases that exercise: AGM6 `MONTE 1`, SRAAM6 `MONTE 30`

**Interfaces:**
- Produces: `RunConfig.nmonte`, reseed per run; `nmonte` outer iterations calling existing single-run body.

- [ ] **Step 1:** `test_parse_monte_sets_nmonte_and_iseed` + `test_run_scenario_monte_2_invokes_body_twice` (mock or tiny end_time).
- [ ] **Step 3:** Parse MONTE; loop in `run_scenario`.
- [ ] **Step 5: Commit** `feat(cli): honor MONTE nmonte outer loop`

**Done when:** MONTE parsed; loop runs `nmonte` times with seeding per C++ semantics (document iseed bump if C++ does it).

---

### Task 44: Per-step `markov_noise` before module loop

**Files:** `cli.py` / run_loop; `cadac/stoch.py` (comments already describe markov_noise / nmonte==0 draw-then-zero)
**C++:** execution.cpp markov update before modules (five MC programs).
**Test:** `test_markov_noise_called_each_step`
**Done when:** MARKOV deck values update each step; nmonte==0 still draw-then-zero per README.

---

### Task 45: y_stat → stati.asc / stat.asc

**Files:** new writer under `cadac/io/` (e.g. `stat.py`); hook in `run_scenario` when `options["stat"]`
**Test:** `test_y_stat_writes_stati_asc`
**Cases:** SRAAM6 sweep/range.
**Done when:** enabling `stat` creates CADAC-shaped stat files without changing plot.csv equations.

---

### Task 46: y_tabout → tabout.asc

**Files:** `cadac/io/tabout.py` + cli hook; C++ tabout dump.
**Test:** `test_y_tabout_writes_tabout_asc`
**Cases:** AIM5, FALCON5.
**Done when:** option produces `tabout.asc`.

---

### Task 47: y_traj → traj.asc (combus history)

**Files:** traj writer + cli; C++ traj.
**Test:** `test_y_traj_writes_traj_asc`
**Cases:** CRUISE5, AIM5.
**Done when:** `traj.asc` written when `traj` option true.

---

### Task 48: y_doc → doc.asc + document_input

**Files:** doc writer matching C++ `document_input` / doc dump.
**Test:** `test_y_doc_writes_doc_asc`
**Done when:** `doc` option emits `doc.asc`.

---

### Task 49: y_comscrn combus screen dump

**Files:** comscrn writer; C++ comscrn.
**Test:** `test_y_comscrn_writes_dump`
**Case:** SAM6 `input_SRBM_Ballistic`.
**Done when:** option dumps combus text artifact.

---

### Task 50: y_merge ploti/stati merge

**Files:** merge writer; **do not** change slot-0 `plot.csv` selection (documented separately). Merge is an additional CADAC output.
**Test:** `test_y_merge_writes_ploti_stati`
**Cases:** MAGSIX multi, AIM5 multi, SRAAM6.
**Done when:** merge option produces merge artifacts; `plot.csv` behavior unchanged.

---

### Task 51: y_scrn timed console tables

**Files:** scrn formatter hooked to timing / field `scrn` outputs.
**Test:** `test_y_scrn_emits_timed_table` (capture stdout or buffer).
**Done when:** `scrn` option prints CADAC-like tables on schedule; default off stays quiet in tests.

---

## Band D — Legacy abilities absent from current C++

Abilities that exist in older Fortran (`Simulations_FTN/`) or older C++ (`Simulations_C++/`) but were **not** carried into current `CADAC_Simulations/`, and are therefore also absent from Python. Implement in `cadac` so legacy input examples run. Do **not** invent work for Fortran comments that have no live `CALL` (MGUID=2 LAG / MAUT=2,4,5). Features still present in current C++ remain Tasks 1–51 only.

**Band D test contract (every task):**
1. **Unit** — `Python/tests/unit/…` asserts the Fortran/old-C++ equation or mode dispatch on a small planted state (not “does not raise”).
2. **Legacy input** — named `*.asc` under Books trees → new JSONC under `Python/cases/<family>/` via `translate_scenario_asc` / `deck_asc_to_jsonc` (or “covered by Task N deck”). Never overwrite existing JSONC; never edit `CADAC_Simulations`.
3. **E2E** — `Python/tests/e2e/` matching `test_rocket6_insertion.py` / `test_cruise5_input1.py` (load JSONC → `run_scenario` → compare golden). Label: `legacy-binary` (old C++ harvest) or `python-golden` (Fortran behavior lock after unit formula lock). Optional `gfortran` harvest only if a Makefile exists (none under SRAAM5/ROCKET3 as of plan writing).
4. **Done when** — unit PASS; `load_scenario(jsonc)` succeeds; named e2e pytest command exists.

**Legacy roots (read-only):**
- Fortran: `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_FTN/`
- Old C++: `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_C++/`

---

### Task 52: ROCKET6 fin actuator (old C++ drop)

**Why:** Current `ROCKET6_250122` has no `actuator.cpp`; old `Simulations_C++/ROCKET6/actuator.cpp` has `Hyper::actuator` / `actuator_0th` / `actuator_scnd`. Python `rocket6` module list has no actuator (sibling HYPER6/FALCON6 do).

**Files:**
- Create: `Python/src/cadac/vehicles/round6/rocket6/actuator.py` (`Rocket6Actuator`)
- Modify: `Python/src/cadac/vehicles/round6/rocket6/vehicle.py` — insert `"actuator"` in old-C++ module order (after control/rcs, before tvc)
- Test: `Python/tests/unit/test_rocket6_actuator.py`
- E2E: `Python/tests/e2e/test_rocket6_actuator.py`
- Source: `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_C++/ROCKET6/actuator.cpp`
- Pattern: `Python/src/cadac/vehicles/round6/hyper6/actuator.py` for file shape only — **transcribe rocket mix**, do not keep elevon mix.

**Interfaces:**
- Consumes: `delacx`, `delecx`, `delrcx`, `mact=|morder|mvehicle|`, `dlimx`, `dlimx_min`, `ddlimx`, `wnact`, `zetact`
- Produces: `delax`, `delex`, `delrx`, `delx1..4`; `morder=0` position limit; `morder=2` 2nd-order + rate limit; `mvehicle=2` → 4-fin mix:
  - `delcx1=-delacx+delecx-delrcx`, … (as in old cpp); recover `delax=(-delx1-delx2+delx3+delx4)/4` etc.

**Legacy input:**
- Source ASC: `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_C++/ROCKET6/input.asc` (MODULES lists `actuator`; `mact` defaults 0 → morder=0,mvehicle=0 until set — set `mact 22` in translated deck for mvehicle=2 + morder=2, or `mact 2` for morder=0 mvehicle=2)
- Dest JSONC: `Python/cases/rocket6/input_actuator_legacy.jsonc` (new; do not overwrite `input.jsonc` / `input_insertion.jsonc`)

**E2E (`legacy-binary`):**
- Cwd: `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_C++/ROCKET6`
- Build/run old binary the same way `Python/tools/cadac_cpp` harvest works, but against that old tree; input file `input.asc` (or the ASC that sets `mact`/`dlimx` after Task 52 adds those fields to a copy used only for harvest)
- Golden: `Python/tests/e2e/goldens/rocket6/actuator_legacy.plot.csv`

- [ ] **Step 1: Failing unit test**

```python
def test_rocket6_mact_02_four_fin_mix_position_limit():
    veh = _rocket6()
    veh.store.set("mact", 2)  # morder=0, mvehicle=2
    veh.store.set("dlimx", 20.0)
    veh.store.set("delacx", 5.0)
    veh.store.set("delecx", 2.0)
    veh.store.set("delrcx", 1.0)
    _exec(veh, "actuator")
    # Unlimited 4-fin mix recovers control deflections exactly:
    # delcx1=-4, delcx2=-2, delcx3=6, delcx4=8 → delax=5, delex=2, delrx=1
    assert abs(veh.store.get("delax") - 5.0) < 1e-9
    assert abs(veh.store.get("delex") - 2.0) < 1e-9
    assert abs(veh.store.get("delrx") - 1.0) < 1e-9

def test_rocket6_mact_22_second_order_rate_limit_path():
    veh = _rocket6()
    veh.store.set("mact", 22)
    veh.store.set("dlimx", 20.0)
    veh.store.set("ddlimx", 100.0)
    veh.store.set("wnact", 50.0)
    veh.store.set("zetact", 0.7)
    veh.store.set("delacx", 1.0)
    veh.store.set("delecx", 0.0)
    veh.store.set("delrcx", 0.0)
    _exec(veh, "actuator")
    assert "delax" in veh.store
```

- [ ] **Step 2:** `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_rocket6_actuator.py -v` → FAIL (no module).
- [ ] **Step 3:** Implement from old `actuator.cpp`; register on vehicle; translate deck → JSONC.
- [ ] **Step 4:** Unit PASS; `load_scenario("cases/rocket6/input_actuator_legacy.jsonc")` succeeds.
- [ ] **Step 5: E2E** harvest old binary → golden; `pytest tests/e2e/test_rocket6_actuator.py -v` (`legacy-binary`).
- [ ] **Step 6: Commit (approval)** `feat(rocket6): port legacy fin actuator from old C++`

**Done when:** units pass; JSONC loads; e2e command named and green (or skips only if golden absent with clear skip message, same as other e2e).

---

### Task 53: SRAAM5 vehicle skeleton + family + deck

**Why:** Entire pseudo-5DOF program; no current SRAAM5. Successor SRAAM6 is 6DOF — do **not** pretend it is SRAAM6.

**Files:**
- Create: `Python/src/cadac/vehicles/flat5/sraam5/` (or package path chosen to mirror pseudo-5DOF; preferred: `vehicles/flat5/sraam5/` with `vehicle.py` stub modules that `define` only)
- Modify: vehicle registry / `family_for` so family `"sraam5"` maps type(s) from deck translation
- Test: `Python/tests/unit/test_sraam5_skeleton.py`
- E2E: `Python/tests/e2e/test_sraam5_inlar1.py` (`python-golden` — stub run may only check `load_scenario` + zero-step until later tasks fill modules; after Task 64, golden locks trajectory)

**Legacy input (owns primary SRAAM5 deck):**
- Source: `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_FTN/SRAAM5/inlar1.asc`
- Dest: `Python/cases/sraam5/inlar1.jsonc`

**Interfaces:**
- Produces: family `sraam5`; vehicle constructible; module names matching MODULES in `inlar1.asc` (`target`, `environment`, `seeker`, `ai_radar`/`s2`, `ins`, `guidance`, `control`, `aerodynamics`, `propulsion`, `forces`, `newton`, `rotations`) registered as stubs that define fields without raising.

- [ ] **Step 1:** `test_sraam5_family_registers_and_load_scenario` — FAIL.
- [ ] **Step 3:** Skeleton + translate `inlar1.asc` → JSONC (extend translator for Fortran-style MODULES if needed; keep changes minimal).
- [ ] **Step 4–5:** Unit PASS; `load_scenario`; e2e skeleton (may skip golden until modules exist — document).
- [ ] **Commit:** `feat(sraam5): add vehicle family skeleton and inlar1 JSONC`

**Done when:** JSONC loads; family dispatches; stubs do not crash `define`.

---

### Tasks 54–64: SRAAM5 modules (one module per task)

Each task: Fortran `MODULE.FOR` subroutine → Python module; unit asserts a Fortran equation; deck covered by Task 53 `inlar1.jsonc` unless noted; e2e `python-golden` updates allowed only after that module’s unit locks the formula.

| Task | Module | Fortran | Unit focus | Commit |
|---:|---|---|---|---|
| 54 | Target G1 | `G1` / MTARG styles in SRAAM5 | planted shooter/target geometry from `MTARG=21` fields (`RHL`, `HT1E`, …) matches Fortran init math | `feat(sraam5): port G1 target` |
| 55 | Environment G2 | `G2` | ISO/air density vs altitude sample | `feat(sraam5): port G2 environment` |
| 56 | Aero A1 | `A1` | table lookup / force coeff at planted Mach/α | `feat(sraam5): port A1 aerodynamics` |
| 57 | Propulsion A2 | `A2` | thrust/mass vs time table row | `feat(sraam5): port A2 propulsion` |
| 58 | Forces A3 | `A3` / `A3TRA` | `A3TRA` body/vel DCM for MTURN paths | `feat(sraam5): port A3 forces` |
| 59 | Newton D1 | `D1` | one-step SBEL/VBEL update vs Fortran kinematics slice | `feat(sraam5): port D1 newton` |
| 60 | Rotations D2 | `D2` | skid-to-turn WBVB from ALP/BETD (MTURN=0 default) | `feat(sraam5): port D2 rotations` |
| 61 | Seeker S1 | `S1`/`S1KIN` | MSEEK 2→3/4 transition fields | `feat(sraam5): port S1 seeker` |
| 62 | INS S4 | `S4` | mins=1 error injection writes INS states | `feat(sraam5): port S4 INS` |
| 63 | Guidance C1 | `C1`/`C1MID`/`C1TERM` | MGUID mid/term commands | `feat(sraam5): port C1 guidance` |
| 64 | Control C2 baseline | `C2` MAUT=44 path | accel autopilot sets pitch/yaw commands for MAUT=44 | `feat(sraam5): port C2 MAUT=44 baseline` |

**Per-task checklist (54–64):**
- [ ] Failing unit in `Python/tests/unit/test_sraam5_<module>.py`
- [ ] Implement minimal Fortran-faithful code
- [ ] Unit PASS; `load_scenario(cases/sraam5/inlar1.jsonc)` still works
- [ ] E2E: `pytest tests/e2e/test_sraam5_inlar1.py -v` (`python-golden` — refresh golden only when this module’s unit locks new outputs)
- [ ] Commit (approval) with message above

**Done when (each):** unit formula assertion green; JSONC still loads; e2e command named.

---

### Task 65: SRAAM5 MTURN=1 bank-to-turn (α+φ kinematics)

**Files:** Modify `sraam5` control + rotations (`C2`/`D2`); Test: `test_sraam5_mturn.py`
**Fortran:** `MODULE.FOR` C2 / A3TRA / D2 — MTURN=1 uses ALP/ALPD/PHD (not BETD).

**Legacy input:** No shipped ASC sets `MTURN=1` (HEAD documents it; decks omit → default 0). **Derive** from Task 53 deck:
- Dest: `Python/cases/sraam5/inlar1_mturn1.jsonc` (copy of `inlar1.jsonc` with `"mturn": 1`)

**Unit:** planted MTURN=1 path writes bank kinematics fields (PHD/PHIB) per D2 BTT branch, not sideslip rate.

**E2E:** `tests/e2e/test_sraam5_mturn1.py` (`python-golden`).

**Done when:** unit PASS; derived JSONC loads; e2e named.
**Commit:** `feat(sraam5): port MTURN=1 bank-to-turn`

---

### Task 66: SRAAM5 MAUTP/MAUTL=1 α/β hold (ALPHAC/BETAC)

**Files:** `sraam5` control; Test: `test_sraam5_maut_hold.py`
**Fortran:** C2 — MAUTP=1 alpha hold; MAUTL=1 beta hold when MTURN=0.

**Legacy input:** No ASC sets MAUT=11. **Derive:**
- Dest: `Python/cases/sraam5/inlar1_maut11.jsonc` (`maut=11`, `alphac`/`betac` set)

**Unit:** with MTURN=0, MAUT=11, planted ALPHAC/BETAC → autopilot holds α/β (error drives toward commands).

**E2E:** `test_sraam5_maut11.py` (`python-golden`).
**Commit:** `feat(sraam5): port MAUTP/MAUTL=1 alpha/beta hold`

---

### Task 67: SRAAM5 and SRAAM6 AI acquisition radar S2 (NTAG, MNAV 2→3)

**Why:** Current SRAAM6 `mnav=3` reads truth; Fortran S2 applies bias/noise and timed NTAG→MNAV 2→3.

**Files:**
- Create/Modify: `sraam5` AI radar module; Modify `Python/src/cadac/vehicles/flat6/sraam6/` seeker/nav path
- Test: `test_sraam5_s2_ai_radar.py`, `test_sraam6_s2_ai_radar.py`
- Fortran: SRAAM5/SRAAM6 `MODULE.FOR` S2

**Legacy input:** No ASC sets `NTAG≠0`. **Derive** from SRAAM5 `inlar1` / SRAAM6 `inlar1`:
- `Python/cases/sraam5/inlar1_ntag.jsonc` (`ntag=1`, timing fields)
- `Python/cases/sraam6/inlar1_ntag.jsonc` from `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_FTN/SRAAM6/inlar1.asc` (+ NTAG) — do not overwrite existing `input_1v1.jsonc`

**Unit:** NTAG=1 starts AI; after DTIM-style interval MNAV becomes 2 then 3; update applies bias/noise stores (assert nonzero vs truth when noise planted).

**E2E:** `test_sraam5_ai_radar.py`, `test_sraam6_ai_radar.py` (`python-golden`).
**Commit:** `feat(sraam): port AI acquisition radar S2`

---

### Task 68: SRAAM5 and SRAAM6 SHAZAM warhead G4SHAZ (MTERM>0)

**Files:** intercept modules on `sraam5` and `sraam6`; Test: `test_sraam_shazam.py`
**Fortran:** G4 / G4SHAZ — AFATL-TR-86-32 geometry YSS/ZSS/DYRB. Current C++ intercept prints aspect only.

**Legacy input (owns SHAZAM deck):**
- Source: `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_FTN/SRAAM6/INLAR3WM.ASC` (`MTERM=1`)
- Dest: `Python/cases/sraam6/inlar3wm.jsonc`
- SRAAM5: derive `Python/cases/sraam5/inlar1_mterm1.jsonc` (`mterm=1`) — covered companion

**Unit:** planted SBTL/VBT1L → G4SHAZ writes YSS/ZSS/DYRB matching Fortran formulas.

**E2E:** `test_sraam6_shazam.py` (`python-golden`).
**Commit:** `feat(sraam): port SHAZAM G4SHAZ warhead block`

---

### Task 69: SRAAM5 and SRAAM6 intercept-plane split (DTCT, DBTC at MSEEK=5)

**Files:** intercept; Test: `test_sraam_intercept_plane.py`
**Fortran:** G4 — nav miss `DTCT` and G&C miss `DBTC` when MSEEK=5.

**Legacy input:** Covered by Task 53 / Task 68 decks; **derive** event or field `mseek=5` near intercept:
- `Python/cases/sraam5/inlar1_mseek5.jsonc` (owns this mode’s JSONC)

**Unit:** at MSEEK=5, both `dtct` and `dbtc` written; assert distinct formulas (not the same scalar).

**E2E:** `test_sraam5_intercept_plane.py` (`python-golden`).
**Commit:** `feat(sraam): port DTCT/DBTC intercept-plane miss split`

---

### Task 70: SRAAM6 MINIT LAR/CIRCLE auto-geometry (codes 12–15, 21–24)

**Files:** SRAAM6 target/init (Fortran G1I / D1I MINIT table); Test: `test_sraam6_minit_lar.py`
**Fortran:** MINIT `_12`…`_15` LAR, `_21`…`_24` CIRCLE (see MODULE comments).

**Legacy inputs (exact filenames on disk):**
| Source ASC | MINIT | Dest JSONC |
|---|---:|---|
| `…/Simulations_FTN/SRAAM6/inlar1.asc` | 21 | `Python/cases/sraam6/inlar1.jsonc` (Task 70 owns; do not overwrite `input_1v1.jsonc`) |
| `…/Simulations_FTN/SRAAM6/INLAR3.ASC` | 13 | `Python/cases/sraam6/inlar3.jsonc` |
| `…/Simulations_FTN/SRAAM6/INCRC3.ASC` | 23 | `Python/cases/sraam6/incrc3.jsonc` |
| `…/Simulations_FTN/SRAAM6/INLAR4.ASC` | 14 | `Python/cases/sraam6/inlar4.jsonc` (extra coverage) |

Catalog names `inlar1`, `INLAR3`, `INCRC3` — all found. (`inlar1` is lowercase on disk.)

**Unit:** for MINIT=21 and MINIT=13, init writes shooter/target geometry matching Fortran LAR/CIRCLE trig (planted input scalars → expected SBEL/aspect).

**E2E:** `test_sraam6_minit_inlar1.py` (`python-golden`); other decks load-only or share golden pattern.
**Commit:** `feat(sraam6): port MINIT LAR/CIRCLE auto-geometry`

---

### Task 71: AIM5 MTARG=1 polar target init (HTE, DHTB, AZTLX)

**Files:** AIM5 aircraft/target init; Test: `test_aim5_mtarg_polar.py`
**Fortran:** AIM5 `MODULE.FOR` G1I — `STBL` from `HTE`, `DHTB`, `AZTLX`. Current AIM5 is Cartesian only.

**Legacy input:**
- Source: `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_FTN/AIM5/INLENV.ASC`
- Dest: `Python/cases/aim5/inlenv.jsonc` (new; do not overwrite `input_hori.jsonc`)

**Unit:**

```python
def test_aim5_mtarg_1_polar_init_stbl():
    # DUMH=SBEL(3)+HTE; THTTL0=atan2(DUMH,DHTB); DBT1=hypot; MATCAR(STBL,…)
    veh = _aim5_aircraft_or_target()
    veh.store.set("mtarg", 1)
    veh.store.set("hte", 9000.0)
    veh.store.set("dhtb", 5000.0)
    veh.store.set("aztlx", 10.0)
    # SBEL z = -10000 as in INLENV
    _init_target(veh)
    stbl = veh.store.get("STBL")  # or ST1EL relative — match Fortran name used in port
    assert stbl is not None
    # assert components vs hand-computed MATCAR
```

**E2E:** `test_aim5_inlenv.py` (`python-golden`).
**Commit:** `feat(aim5): port MTARG=1 polar target initialization`

---

### Task 72: CRUISE5 stochastic terrain + look-down/look-forward TF/OA

**Files:** `cruise5/guidance.py` (+ terrain helper); Test: `test_cruise5_terrain_tf.py`
**Fortran:** D1TER, C1 — MGUIDP=1 look-down, =2 look-fwd; `OCCDEN`/`SIGOBS`/`RAHEAD`.

**Legacy input (owns CRUISE5 legacy primary deck for Band D):**
- Source: `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_FTN/CRUISE5/INPUT.ASC` (`MGUID=32` → MGUIDP=2; sets `RAHEAD`/`OCCDEN`/`SIGOBS`)
- Dest: `Python/cases/cruise5/input_ftn.jsonc` (new name; do not overwrite `input.jsonc` / `input_1.jsonc`)

**Unit:** MGUIDP=2 with planted terrain DB → pitch command / `ELMAX` path; obstacle draw uses `OCCDEN`/`SIGOBS` (assert Rayleigh/occurrence call sites via planted seed).

**E2E:** `test_cruise5_input_ftn_terrain.py` (`python-golden`).
**Commit:** `feat(cruise5): port stochastic terrain TF/OA`

---

### Task 73: CRUISE5 scene-matching imaging seeker (MSEEK 1→4, NMAP/NFIX)

**Files:** `cruise5/seeker.py`; Test: `test_cruise5_scene_seeker.py`
**Fortran:** S1/S1LOS/S1EPCH. Current seeker is LOS acquire/track only (Task 2).

**Legacy input:** Covered by Task 72 deck (`MSEEK=1`, `NMAP`, `NFIXM` in `INPUT.ASC`).

**Unit:** MSEEK advances 1→…→4 with NFIX increments; epoch/map fields written.

**E2E:** extend `test_cruise5_input_ftn_terrain.py` or `test_cruise5_scene_seeker.py` (`python-golden`).
**Commit:** `feat(cruise5): port scene-matching imaging seeker`

---

### Task 74: CRUISE5 GPS (MGPS 0/1/2)

**Files:** Create `cruise5/gps.py` or sensor module; register on vehicle; Test: `test_cruise5_gps.py`
**Fortran:** S2. Current Cruise has no GPS.

**Legacy input:**
- Source: `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_FTN/CRUISE5/INGPS.ASC` (also covered partially by Task 72 `INPUT.ASC` MGPS events)
- Dest: `Python/cases/cruise5/ingps.jsonc` (Task 74 owns)

**Unit:** MGPS=1 enables; MGPS=2 update writes GPS measurement stores then resets per S4 handshake fields.

**E2E:** `test_cruise5_ingps.py` (`python-golden`).
**Commit:** `feat(cruise5): port GPS MGPS 0/1/2`

---

### Task 75: CRUISE5 INS + Doppler-aided INS (MINS 0/1/2) + HBGM

**Files:** Create `cruise5/ins.py`; Test: `test_cruise5_ins.py`
**Fortran:** S4/S4GYRO/S4ACCL/S4ALT — terrain altimeter `HBGM`.

**Legacy input:** Covered by Task 72 / 74 decks (`MINS=1` in `INPUT.ASC` / `INGPS.ASC`).

**Unit:** MINS=1 writes INS errors; MINS=2 Doppler-aid path; `HBGM` from altimeter branch.

**E2E:** shared `ingps` / `input_ftn` golden (`python-golden`).
**Commit:** `feat(cruise5): port INS/Doppler INS and HBGM`

---

### Task 76: CRUISE5 skid-to-turn / sideslip hold (MTURN=0, MAUTL=1, BETAC)

**Files:** `cruise5/control.py` (+ kinematics if needed); Test: `test_cruise5_mturn0_sideslip.py`
**Fortran:** C2.

**Legacy input:** All FTN cruise decks set `MTURN=1`. **Derive:**
- Dest: `Python/cases/cruise5/input_ftn_mturn0.jsonc` from Task 72 with `mturn=0`, `maut=13`→ force `mautl=1` (e.g. `maut=11`), `betac` set

**Unit:** MTURN=0 + MAUTL=1 holds BETAC (sideslip error → yaw command).

**E2E:** `test_cruise5_mturn0.py` (`python-golden`).
**Commit:** `feat(cruise5): port skid-to-turn sideslip hold`

---

### Task 77: CRUISE5 body-rate autopilots + aero/TVC mix (MAUTL/P=6, APTVC)

**Files:** `cruise5/control.py`; Test: `test_cruise5_maut6_aptvc.py`
**Fortran:** C2 — live `MAUTP.EQ.6` and `MAUTL.EQ.6` (BTT path); `APTVC` mixes aero/TVC poles.

**Legacy input:** No ASC sets MAUT=66. **Derive:**
- `Python/cases/cruise5/input_ftn_maut66.jsonc` (`maut=66`, `aptvc` in (0,1))

**Unit:** MAUTP=6 rate law; APTVC=0.5 splits DELQ/ETA per Fortran `POLEA`/`POLEP` formulas.

**E2E:** `test_cruise5_maut66.py` (`python-golden`).
**Commit:** `feat(cruise5): port MAUT=6 body-rate and APTVC mix`

---

### Task 78: CRUISE5 inverted guidance (MROLL=1)

**Files:** `cruise5/guidance.py`; Test: `test_cruise5_mroll1.py`
**Fortran:** C1 — `IF(MROLL.EQ.1) PHIBVC=3.1412-PHIBVC`.

**Legacy input:** No ASC sets MROLL digit. **Derive** from Task 72:
- `Python/cases/cruise5/input_ftn_mroll1.jsonc` (`mguid` with hundreds digit 1, e.g. 132)

**Unit:** MROLL=1 inverts bank command vs MROLL=0 for same geometry.

**E2E:** `test_cruise5_mroll1.py` (`python-golden`).
**Commit:** `feat(cruise5): port inverted guidance MROLL=1`

---

### Task 79: CRUISE5 atmosphere/wind MAIR=`|MATM|MWIND|`

**Files:** cruise5 environment (round3 env path); Test: `test_cruise5_mair.py`
**Fortran:** G2 — MATM 0/1 tabular; MWIND 0/1/2/3. Current CRUISE5 round3 is ISO-62 only.

**Legacy input:** Task 72 `INPUT.ASC` sets `MAIR=INT(1)` in an event (MATM/MWIND decode: `MATM=INT(MAIR/10)`, `MWIND=MAIR-MATM*10` → MAIR=1 means MATM=0,MWIND=1 constant wind). Covered by `input_ftn.jsonc`.

**Unit:** MAIR=1 applies constant wind from `DVAEL`/`PSIWLX`; MAIR=10 selects tabular atmos when weather deck present (plant mini deck).

**E2E:** terrain/GPS e2e already exercises wind event (`python-golden`).
**Commit:** `feat(cruise5): port MAIR atmosphere/wind options`

---

### Task 80: FALCON6 Dryden turbulence (MAIR MTURB=1)

**Files:** `falcon6/environment.py` (or Flat6 env hook — FALCON6-only); Test: `test_falcon6_dryden.py`
**Fortran:** G2/G2TURB. Current FALCON6 `mwind` 0/1/2 only. Reuse `cadac.stoch`; **transcribe FALCON6 `G2TURB`**, do not copy AGM6 sequence.

**Legacy input:** `inpitch.asc` does **not** set MAIR/MTURB. **Derive:**
- Source base: `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_FTN/FALCON6/inpitch.asc`
- Dest: `Python/cases/falcon6/inpitch_mturb1.jsonc` with `mair=100` (`MTURB=1,MWIND=0,MATMO=0`) plus `turb_length`/`turb_sigma` fields from HEAD

**Unit:** MTURB=1 burns Dryden white-noise draws and adds `VTAG` gust components matching G2TURB formulas at planted `DVBA`.

**E2E:** `test_falcon6_dryden.py` (`python-golden`).
**Commit:** `feat(falcon6): port Dryden turbulence MTURB=1`

---

### Task 81: FALCON6 tabular weather (MWIND=3, MATMO=3)

**Files:** same env module; Test: `test_falcon6_weather_tabular.py`
**Fortran:** G2 WEATHER deck look-ups.

**Legacy input:** No ASC ships WEATHER+MAIR=33. **Derive:**
- `Python/cases/falcon6/inpitch_mair33.jsonc` (`mair=33`) + minimal `weather_deck` JSONC beside it (table altitudes/speeds/dirs/atm columns per MODULE comment format)

**Unit:** MATMO=3 density/temp from weather table; MWIND=3 horizontal wind from table at altitude.

**E2E:** `test_falcon6_weather_tabular.py` (`python-golden`).
**Commit:** `feat(falcon6): port tabular weather MWIND=3 MATMO=3`

---

### Task 82: HYPER3 tabular atmosphere (GHAME3 MAIR=1)

**Files:** HYPER3 / round3 environment; Test: `test_hyper3_mair1_weather.py`
**Fortran:** GHAME3 G2 — MAIR=0 ISO, MAIR=1 weather deck (atmosphere only). Current HYPER3 is ISO-62 only.

**Legacy input:** `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_FTN/GHAME3/INPUT.ASC` does **not** set `MAIR=1` (defaults ISO). **Nearest real ASC:** that `INPUT.ASC`. **Derive:**
- Dest: `Python/cases/hyper3/input_mair1.jsonc` from existing `input.jsonc` / translated `INPUT.ASC` with `mair=1` + weather table (reuse ROCKET6 weather-deck reader pattern / `weather_deck_Wallops` shape adapted to GHAME3 WINDS columns)

**Unit:** MAIR=1 look_up RHX/CTMP/WPRES vs altitude matches TABLE interpolation on planted deck.

**E2E:** `test_hyper3_mair1.py` (`python-golden`).
**Commit:** `feat(hyper3): port GHAME3 tabular atmosphere MAIR=1`

---

### Task 83: HYPER6 tabular atmosphere/wind (GHAME6 MATMO=3, MWIND=3)

**Files:** `eom/round6.py` / HYPER6 environment allowlist; Test: `test_hyper6_matmo3_mwind3.py`
**Fortran:** GHAME6 G2. Current HYPER6 has matmo 0/1 and shear wind, no WEATHER deck path like ROCKET6. **Reuse ROCKET6 deck reader**; transcribe GHAME6 digit packing `MAIR=|MTURB|MWIND|MATMO|`.

**Legacy input:** `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_FTN/GHAME6/inclimb.asc` has no MAIR=x33. **Nearest:** that `inclimb.asc`. **Derive:**
- Dest: `Python/cases/hyper6/inclimb_mair033.jsonc` (`mair=33` or packed form matching Python round6 decode — document digit order vs ROCKET6; **must match GHAME6 Fortran decode** `MTURB=INT(MAIR/100)`, `MWIND=INT((MAIR-MTURB*100)/10)`, `MATMO=…`)

**Unit:** MATMO=3 + MWIND=3 read weather deck; assert density and wind vs table.

**E2E:** `test_hyper6_weather_tabular.py` (`python-golden`).
**Commit:** `feat(hyper6): port GHAME6 tabular atmosphere/wind`

---

### Task 84: ROCKET3 vehicle skeleton + family + deck

**Why:** Entire 3DOF open-loop multi-stage ascent; no current ROCKET3. Do **not** fold into ROCKET6.

**Files:** Create `Python/src/cadac/vehicles/round3/rocket3/`; register family `rocket3`; Test: `test_rocket3_skeleton.py`
**Legacy input (owns ROCKET3 deck):**
- Source: `/home/valentin/Books/CADAC/CADAC4_Enhance/Simulations_FTN/ROCKET3/INLAUNCH.ASC`
- Dest: `Python/cases/rocket3/inlaunch.jsonc`

**E2E:** `test_rocket3_inlaunch.py` (`python-golden` stub→full as modules land).
**Commit:** `feat(rocket3): add vehicle family skeleton and inlaunch JSONC`

---

### Tasks 85–89: ROCKET3 modules (one module per task)

| Task | Module | Fortran | Unit focus | Commit |
|---:|---|---|---|---|
| 85 | Environment G2 | `G2` | atmosphere at launch alt | `feat(rocket3): port G2 environment` |
| 86 | Aero A1 | `A1` MAERO=`|MAERT|MAERV|` | stage 11/12/13 coeff select | `feat(rocket3): port A1 aerodynamics` |
| 87 | Propulsion A2 | `A2` MPROP 0/1/2 | burning fuel flow / mass | `feat(rocket3): port A2 propulsion` |
| 88 | Forces A3 | `A3` | force sum in geographic frame | `feat(rocket3): port A3 forces` |
| 89 | Newton D1 + α schedule | `D1` | open-loop `ALPHAX` events (−5.5° stage) integrate trajectory step | `feat(rocket3): port D1 newton and alpha schedule` |

**Per-task:** unit → implement → `load_scenario(cases/rocket3/inlaunch.jsonc)` → e2e `python-golden` → commit. Deck covered by Task 84.

**Done when (each):** Fortran formula unit green; JSONC loads; e2e command named.

---

## Band D deck ownership summary

| Legacy ASC (Books) | Dest JSONC | Owning task |
|---|---|---:|
| `Simulations_C++/ROCKET6/input.asc` | `cases/rocket6/input_actuator_legacy.jsonc` | 52 |
| `Simulations_FTN/SRAAM5/inlar1.asc` | `cases/sraam5/inlar1.jsonc` | 53 |
| derived MTURN/MAUT/NTAG/MTERM/MSEEK | `cases/sraam5/inlar1_*.jsonc` | 65–69 |
| `Simulations_FTN/SRAAM6/INLAR3WM.ASC` | `cases/sraam6/inlar3wm.jsonc` | 68 |
| `Simulations_FTN/SRAAM6/inlar1.asc` (+NTAG derive) | `cases/sraam6/inlar1.jsonc` / `inlar1_ntag.jsonc` | 67, 70 |
| `Simulations_FTN/SRAAM6/INLAR3.ASC` | `cases/sraam6/inlar3.jsonc` | 70 |
| `Simulations_FTN/SRAAM6/INCRC3.ASC` | `cases/sraam6/incrc3.jsonc` | 70 |
| `Simulations_FTN/AIM5/INLENV.ASC` | `cases/aim5/inlenv.jsonc` | 71 |
| `Simulations_FTN/CRUISE5/INPUT.ASC` | `cases/cruise5/input_ftn.jsonc` | 72 |
| `Simulations_FTN/CRUISE5/INGPS.ASC` | `cases/cruise5/ingps.jsonc` | 74 |
| derived cruise MTURN0/MAUT66/MROLL1 | `cases/cruise5/input_ftn_*.jsonc` | 76–78 |
| `Simulations_FTN/FALCON6/inpitch.asc` (+derive) | `cases/falcon6/inpitch_mturb1.jsonc`, `inpitch_mair33.jsonc` | 80–81 |
| `Simulations_FTN/GHAME3/INPUT.ASC` (+derive MAIR=1) | `cases/hyper3/input_mair1.jsonc` | 82 |
| `Simulations_FTN/GHAME6/inclimb.asc` (+derive) | `cases/hyper6/inclimb_mair033.jsonc` | 83 |
| `Simulations_FTN/ROCKET3/INLAUNCH.ASC` | `cases/rocket3/inlaunch.jsonc` | 84 |

**Catalog name notes:** `inlar1` / `INLAR3` / `INCRC3` exist under SRAAM6 FTN (mixed case). SRAAM5 has `inlar1.asc`, `inlar3.asc`, `INCIRC3.ASC` (not `INCRC3`). No FTN Makefile for SRAAM5/ROCKET3 → no blocking Fortran binary harvest.

---

## Coverage table

| Program | Gap count (catalog) | Tasks (this plan) | Case-blocking / legacy decks |
|---|---:|---|---|
| HYPER3 | 0 current-C++ + 1 Band D | **82** | legacy `input_mair1.jsonc` (from GHAME3 `INPUT.ASC`) |
| AIM5 | 0 current-C++ + 1 Band D | **71** | legacy `inlenv.jsonc` (`INLENV.ASC`) |
| SRAAM6 | 0 current-C++ + 4 Band D | **67–70** | `inlar1`/`inlar3`/`incrc3`/`inlar3wm` JSONC |
| FALCON5 | 8 (+ note 6/60 non-gaps) | 1, 19, 20a–e | **1** (mcontrol 16/36) |
| CRUISE5 | 9 current (excl. 33/40/66) + 8 Band D | 2, 21, 22×6, **72–79** | **2** mseeker; Band D `input_ftn`/`ingps` + derives |
| HYPER5 | 6 | 23×4, 24, 25 | none shipped |
| FALCON6 | 6 current + 2 Band D | 3, 4, 26–29, **80–81** | **3, 4** mautp; Band D `inpitch_mturb1` / `inpitch_mair33` |
| AGM6 | 3 | 5, 30, 31 | **5** (term digit 5) |
| SAM6 | 4 | 32–35 | none shipped |
| HYPER6 | 14 current + 1 Band D | 6–17, 36–37, **83** | **6–17**; Band D `inclimb_mair033.jsonc` |
| ROCKET6 | 7 current + 1 Band D | 18, 38–41, **52** | **18**; Band D `input_actuator_legacy.jsonc` |
| MAGSIX | 2 | 42 | none shipped |
| Kernel | 9 | 43–51 | MONTE/stat/tabout/traj/doc/comscrn/merge/scrn |
| **SRAAM5 (new)** | 1 program + MTURN + α/β (+ shared 67–69) | **53–66** (+67–69 shared) | `inlar1.jsonc` + derives |
| **ROCKET3 (new)** | 1 program (multi-module) | **84–89** | `inlaunch.jsonc` |

**Gap vs task notes:** FALCON5 Task 20 and CRUISE5 Task 22 and HYPER5 Task 23 are multi-mini-task bundles. Kernel Band C is nine I/O abilities → Tasks 43–51. Band D is Tasks **52–89** (38 tasks): 1 old-C++ actuator + SRAAM5 skeleton/modules/modes + shared missile legacy + AIM5 polar + CRUISE5 Fortran slice + FALCON6 turb/weather + HYPER3/6 tabular + ROCKET3 family.

---

## Self-review (plan author)

1. **Spec coverage (Bands A–C):** Every current-C++ catalog bullet maps to Tasks 1–51; CRUISE5 33/40/66 excluded; FALCON5 6/60 noted non-port; HYPER3/AIM5/SRAAM6 zero *current-C++* gap stated, with Band D extensions called out in the coverage table.
2. **Spec coverage (Band D):** Catalog items map as: ROCKET6 actuator→52; SRAAM5 program→53–64; MTURN→65; α/β hold→66; AI S2→67; SHAZAM→68; DTCT/DBTC→69; MINIT LAR/CIRCLE→70; AIM5 MTARG→71; CRUISE5 terrain/seeker/GPS/INS/MTURN0/MAUT6/MROLL/MAIR→72–79; FALCON6 Dryden/weather→80–81; GHAME3→HYPER3→82; GHAME6→HYPER6→83; ROCKET3→84–89. No tasks for dead Fortran comments (MGUID=2 LAG / MAUT=2,4,5).
3. **Placeholders:** No TBD/TODO; each Band D task names unit test file, legacy ASC→JSONC (or covered-by), e2e label (`legacy-binary` vs `python-golden`), and Done when.
4. **Dependencies:** FALCON6 altitude (4) after normal accel (3); AGM6 combined (31) after 5+30; HYPER6 mguide 6/8 after seeker/datalink; Band C after current physics; Band D SRAAM5 modes after skeleton/modules; CRUISE5 Band D after Task 2 seeker baseline where they share the seeker module; ROCKET3 modules after skeleton 84.
5. **inventory.json:** Not used as truth; do not flip inventory rows as a substitute for unit tests.
6. **Deck name audit:** Catalog `inlar1`/`INLAR3`/`INCRC3` found under SRAAM6 FTN; AIM5 `INLENV.ASC` found; GHAME3/GHAME6/FALCON6 mode decks often omit the switch — tasks derive JSONC from nearest real ASC and say so.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-29-cadac-unported-abilities.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh Grok non-fast subagent per task, review between tasks (`superpowers:subagent-driven-development`).
2. **Inline Execution** — this session with `superpowers:executing-plans` and checkpoints.

Which approach?
