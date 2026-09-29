# Suite Red Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Python unit suite green for the eight remaining failures after the traj/comscrn flatten fix (commit `43723c3`), without reopening flatten work.

**Architecture:** Eight independent test/contract mismatches left by the unported-abilities wave. Each task is a one-behavior fix: either update a stale expectation to the ported contract, or narrow a false-positive lint / move a define-time plant so the existing unit’s stated contract holds. No shared code path across tasks except the final `UPDATES.md` bump.

**Tech Stack:** Python 3, pytest (`cd Python && PYTHONPATH=src:tools python -m pytest <files> -q`).

**Spec:** User brief for this plan (eight named failures + chosen directions); supporting context in `docs/superpowers/plans/2026-09-29-cadac-unported-abilities.md` (Tasks 15, 36, 53–69) and root `UPDATES.md` `0.191.0` / `0.191.1`.

## Global Constraints

- Do not reopen traj/comscrn flatten (`_flatten_packet_values`, `_is_vector_entry`). Out of scope (accepted flatten minors): duplicated `_is_vector_entry`, `len==3` non-string heuristic, missing negative flatten tests.
- Do not edit `CADAC_Simulations/`. Do not edit `README.md`. Do not edit `docs/superpowers/plans/2026-09-29-cadac-unported-abilities.md`.
- Confirm each failure is independent of flatten helpers (none of the eight tests or their production targets import `_flatten_packet_values` / `_is_vector_entry`).
- TDD: watch RED on the named test, minimal fix, watch GREEN. Command form: `cd Python && PYTHONPATH=src:tools python -m pytest <files> -q`.
- Commit steps are included but run **only if the user asks** for a commit in that session.
- Final docs task bumps root `UPDATES.md` bug-fix subsubver under `0.191.x` → next free is **`0.191.2`** (after `0.191.1`).

## File structure

- `Python/tests/unit/test_cadac_quality_metrics.py` — exempt `look_up` with `>=`; narrow names-list membership lint.
- `Python/tests/unit/test_cruise5_satellite.py`, `test_cruise5_target.py` — expect `Cruise5Environment`.
- `Python/tests/unit/test_hyper6_control_accel.py` — rewrite unported maut 32/42 test to ported pitch-digit-2 contract; plant `dlde` in helper.
- `Python/src/cadac/vehicles/round6/hyper6/seeker.py` — stop defining `STBIK`/`VTBIK` at define; create on execute.
- `Python/cases/rocket3/inlaunch.jsonc`, `Python/cases/sraam5/inlar1.jsonc`, `inlar1_maut11.jsonc`, `inlar1_mturn1.jsonc`, `inlar1_ntag.jsonc` — leading overview comments (≥40 chars).
- `Python/tests/unit/test_sraam5_skeleton.py` + `Python/cases/sraam5/inlar1.jsonc` — include intentional `intercept` module.
- `UPDATES.md` — `0.191.2`.

---

### Task 1: Exempt `look_up` in quality baseline (like `store_get`)

**Files:**
- Modify: `Python/tests/unit/test_cadac_quality_metrics.py` (`test_scan_matches_checked_in_baseline`)
- Test: same

**Interfaces:**
- Consumes: `scan_cadac`, `load_metrics`, checked-in `Python/tools/cadac_quality/baseline.json` (`look_up` baseline 229; live scan 253).
- Produces: `look_up` treated as a rising counter (parity ports), not a hard equality.

**Why this (not remove calls):** Extra `look_up` sites are real table lookups from the abilities wave; rewriting production to hit 229 would fight the port. Same pattern as existing `store_get` / `store_set` exemptions.

- [ ] **Step 1: Confirm RED (test already fails; do not change production)**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_cadac_quality_metrics.py::test_scan_matches_checked_in_baseline -q`

Expected: FAIL `AssertionError: look_up` with `assert 253 == 229`.

- [ ] **Step 2: Exempt `look_up` with `>=`**

In `test_scan_matches_checked_in_baseline`, add `"look_up"` to the `continue` key set (with `store_get` / `store_set`), and after the loop add:

```python
assert data["look_up"] >= base["look_up"]
```

Do **not** rewrite `baseline.json`.

- [ ] **Step 3: Confirm GREEN**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_cadac_quality_metrics.py::test_scan_matches_checked_in_baseline -q`

Expected: PASS

- [ ] **Step 4: Commit (only if the user asks)**

```bash
git add Python/tests/unit/test_cadac_quality_metrics.py
git commit -m "test(quality): exempt look_up baseline like store_get"
```

---

### Task 2: Allow `for name in store.names()`; keep membership banned

**Files:**
- Modify: `Python/tests/unit/test_cadac_quality_metrics.py` (`test_src_membership_does_not_use_names_list`)
- Test: same
- Do **not** rewrite `Python/src/cadac/io/tabout.py:32` or `Python/src/cadac/io/stat.py:20` (those are legitimate iteration).

**Interfaces:**
- Consumes: existing `_FOR_IN_NAMES` for `for … in names`.
- Produces: new regexes that distinguish iteration over `store.names()` from membership (`x in store.names()` / `x not in store.names()`).

**Why this (not rewrite tabout/stat):** Hits are `for name in store.names():` — iteration, not membership. The lint substring `"in store.names()"` misreads `for`.

- [ ] **Step 1: Confirm RED**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_cadac_quality_metrics.py::test_src_membership_does_not_use_names_list -q`

Expected: FAIL with hits at `tabout.py:32` and `stat.py:20` (`for name in store.names():`).

- [ ] **Step 2: Narrow the lint**

Replace the body of `test_src_membership_does_not_use_names_list` with:

```python
_FOR_IN_STORE_NAMES = re.compile(r"\bfor\s+\w+\s+in\s+store\.names\(\)")
_MEMBERSHIP_STORE_NAMES = re.compile(r"\b(?:not\s+)?in\s+store\.names\(\)")

def test_src_membership_does_not_use_names_list():
    hits = []
    for path in ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if _MEMBERSHIP_STORE_NAMES.search(line) and not _FOR_IN_STORE_NAMES.search(line):
                hits.append(f"{path}:{i}:{stripped}")
            if "in names" in line and "store.names()" in text:
                if _FOR_IN_NAMES.search(line) or _FOR_IN_STORE_NAMES.search(line):
                    continue
                if _MEMBERSHIP_STORE_NAMES.search(line):
                    continue  # already recorded above
                hits.append(f"{path}:{i}:{stripped}")
    assert hits == [], "use `name in store`:\n" + "\n".join(hits)
```

Place the two new regex constants next to `_FOR_IN_NAMES` at module scope (not nested inside the test). Keep the assertion message `use \`name in store\`:`.

- [ ] **Step 3: Confirm GREEN**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_cadac_quality_metrics.py::test_src_membership_does_not_use_names_list -q`

Expected: PASS (tabout/stat iteration allowed; any real `name in store.names()` still fails).

- [ ] **Step 4: Commit (only if the user asks)**

```bash
git add Python/tests/unit/test_cadac_quality_metrics.py
git commit -m "test(quality): allow iterating store.names(), ban membership only"
```

---

### Task 3: Cruise5 satellite/target expect `Cruise5Environment`

**Files:**
- Modify: `Python/tests/unit/test_cruise5_satellite.py` (`test_constructor_type_health_no_aero_deck`)
- Modify: `Python/tests/unit/test_cruise5_target.py` (`test_constructor_type_health_no_aero_deck`)
- Test: both (same one-line expectation change)

**Interfaces:**
- Consumes: `Cruise5Environment` from `cadac.vehicles.round3.cruise5.environment` (already used by `satellite.py` / `target.py` constructors; `UPDATES.md` `0.191.0`).
- Produces: constructor type-health asserts matching production module lists.

**Why this:** Production already constructs `Cruise5Environment`; stale tests still expect `Round3Environment`.

- [ ] **Step 1: Confirm RED**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_cruise5_satellite.py::test_constructor_type_health_no_aero_deck tests/unit/test_cruise5_target.py::test_constructor_type_health_no_aero_deck -q`

Expected: FAIL at index 0 — `Cruise5Environment != Round3Environment`.

- [ ] **Step 2: Update imports and expectations**

In both files:

```python
from cadac.eom.round3 import Round3Newton  # drop Round3Environment if unused
from cadac.vehicles.round3.cruise5.environment import Cruise5Environment
```

In each `test_constructor_type_health_no_aero_deck` module-type list, replace the first entry `Round3Environment` with `Cruise5Environment`. Leave the rest of the list unchanged (`Cruise5SatelliteForces` / `Cruise5TargetForces` / `Round3Newton` / `Cruise5TargetIntercept` as today).

- [ ] **Step 3: Confirm GREEN**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_cruise5_satellite.py::test_constructor_type_health_no_aero_deck tests/unit/test_cruise5_target.py::test_constructor_type_health_no_aero_deck -q`

Expected: PASS

- [ ] **Step 4: Commit (only if the user asks)**

```bash
git add Python/tests/unit/test_cruise5_satellite.py Python/tests/unit/test_cruise5_target.py
git commit -m "test(cruise5): expect Cruise5Environment on satellite and target"
```

---

### Task 4: HYPER6 maut 32/42 — ported pitch-digit-2 contract

**Files:**
- Modify: `Python/tests/unit/test_hyper6_control_accel.py` (`test_hyper6_unported_pitch_digit_raises` → renamed; `_hyper6_vehicle` plants `dlde`)
- Do **not** change `Python/src/cadac/vehicles/round6/hyper6/control.py` gate (`pitch_ok` already includes `2`; Task 15 intentionally ported `mautp=2`).

**Interfaces:**
- Consumes: `Hyper6Control.control_pitch_rate` (needs store `dlde`); `_hyper6_vehicle` / `_execute_module`.
- Produces: test proving maut 32 (yaw3+pitch2) and 42 (yaw4+pitch2) execute pitch-rate SAS without `unknown maut` / `KeyError`.

**Why this (not re-reject):** Unported-abilities Task 15 enabled `mautp=2`; control gate allows pitch digit 2. Failure is `KeyError: 'dlde'` because the accel helper never planted aero `dlde`, not because digits should stay rejected.

- [ ] **Step 1: Confirm RED on the old raise expectation**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_hyper6_control_accel.py::test_hyper6_unported_pitch_digit_raises -q`

Expected: FAIL `KeyError: 'dlde'` at `state.py` (execution reaches pitch-rate path).

- [ ] **Step 2: Plant `dlde` in `_hyper6_vehicle` and rewrite the test**

In `_hyper6_vehicle`, add among the aero `Field(...)` definitions:

```python
Field("dlde", -12.0, "real", "out", "aerodynamics"),
```

Replace `test_hyper6_unported_pitch_digit_raises` with:

```python
def test_hyper6_maut_32_42_pitch_rate_runs():
    """mautp=2 is ported; 32/42 must run pitch-rate SAS (not raise unknown maut)."""
    ctrl = Hyper6Control()
    for maut in (32, 42):
        veh = _hyper6_vehicle(maut=maut, alcomx=0.5, psivdcomx=0.2, qcomx=4.0)
        _execute_module(veh, "control")
        store = veh.store
        want = ctrl.control_pitch_rate(veh, store.get("qcomx"))
        if abs(want) > DELIMX:
            want = DELIMX * _sign(want)
        assert _approx(store.get("delecx"), want)
        assert store.get("delacx") != 0.0 or store.get("delrcx") != 0.0
```

Reuse existing `_approx` / `_sign` / `DELIMX` already in this file. Keep `pytest` import if still used elsewhere in the file; remove it only if unused after the rewrite.

- [ ] **Step 3: Confirm GREEN**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_hyper6_control_accel.py::test_hyper6_maut_32_42_pitch_rate_runs -q`

Expected: PASS

Also run the rest of the file so the shared helper change did not break siblings:

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_hyper6_control_accel.py -q`

Expected: PASS

- [ ] **Step 4: Commit (only if the user asks)**

```bash
git add Python/tests/unit/test_hyper6_control_accel.py
git commit -m "test(hyper6): assert maut 32/42 pitch-rate path instead of unported raise"
```

---

### Task 5: HYPER6 seeker — create STBIK/VTBIK on execute, not define

**Files:**
- Modify: `Python/src/cadac/vehicles/round6/hyper6/seeker.py` (`define` fields; start of `execute`)
- Test: `Python/tests/unit/test_hyper6_mguide_7_agl.py::test_hyper6_mguide_7_agl` (already states the contract; do not weaken it)

**Interfaces:**
- Consumes: existing `Field`, `_ZEROS3`; guidance `_ensure_stbik_from_kinematics` (fills when absent).
- Produces: after `veh.define()`, `STBIK`/`VTBIK` absent unless seeker has executed; seeker `execute` defines them before first `store.get`.

**Why this (not test-side plant):** Test docstring — “Truth kinematics only — do not plant STBIK/VTBIK (seeker not on vehicle)” / assert absent after define. Seeker define currently plants zeros, so guidance’s kinematics fill early-returns. Production must match that contract.

- [ ] **Step 1: Confirm RED**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_hyper6_mguide_7_agl.py::test_hyper6_mguide_7_agl -q`

Expected: FAIL `assert 'STBIK' not in store` after `veh.define()`.

- [ ] **Step 2: Move STBIK/VTBIK from define to execute**

In `Hyper6Seeker.define`, **delete** these two fields from the `Field(...)` tuple:

```python
Field("STBIK", _ZEROS3, "vec", "out", "seeker"),
Field("VTBIK", _ZEROS3, "vec", "out", "seeker"),
```

At the top of `Hyper6Seeker.execute`, before any `store.get("STBIK")` / `store.get("VTBIK")`:

```python
if "STBIK" not in store:
    store.define(Field("STBIK", _ZEROS3, "vec", "out", "seeker"))
if "VTBIK" not in store:
    store.define(Field("VTBIK", _ZEROS3, "vec", "out", "seeker"))
```

Do not change `guidance._ensure_stbik_from_kinematics` (already defines+sets from STCII−SBIIC when missing).

- [ ] **Step 3: Confirm GREEN (AGL + seeker regression)**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_hyper6_mguide_7_agl.py::test_hyper6_mguide_7_agl tests/unit/test_hyper6_seeker.py -q`

Expected: PASS (`test_mseek_2_to_5_do_not_raise` still sees `STBIK` after execute).

- [ ] **Step 4: Commit (only if the user asks)**

```bash
git add Python/src/cadac/vehicles/round6/hyper6/seeker.py
git commit -m "fix(hyper6): plant STBIK/VTBIK on seeker execute, not define"
```

---

### Task 6: Add overview comments to five catalog scenarios

**Files:**
- Modify (leading `/* … */` only):
  - `Python/cases/rocket3/inlaunch.jsonc`
  - `Python/cases/sraam5/inlar1.jsonc`
  - `Python/cases/sraam5/inlar1_maut11.jsonc`
  - `Python/cases/sraam5/inlar1_mturn1.jsonc`
  - `Python/cases/sraam5/inlar1_ntag.jsonc`
- Test: `Python/tests/unit/test_jsonc.py::test_catalog_scenarios_have_overview_comment`

**Interfaces:**
- Consumes: `leading_comment(text)` — first block/line comment before root value; must be non-`None` and `len(comment) >= 40`.
- Produces: each of the five files starts with a `/* … */` overview (≥40 chars of comment body after strip), matching peer style (e.g. `cases/sraam6/inlar1.jsonc`, `cases/aim5/input.jsonc`).

**Why this:** Test enumerates catalog scenarios (`title` + `vehicles`/`modules`); these five lack a leading comment or are too short.

- [ ] **Step 1: Confirm RED**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_jsonc.py::test_catalog_scenarios_have_overview_comment -q`

Expected: FAIL with missing list exactly:

```text
['rocket3/inlaunch.jsonc', 'sraam5/inlar1.jsonc', 'sraam5/inlar1_maut11.jsonc', 'sraam5/inlar1_mturn1.jsonc', 'sraam5/inlar1_ntag.jsonc']
```

- [ ] **Step 2: Prepend a ≥40-character overview to each file**

Insert as the first bytes of each file (before `{`), adapting titles from each file’s `"title"` field. Examples (each body ≥40 chars):

```text
/* ROCKET3 INLAUNCH: launch to 300 km orbit from IN300.ASC.
   Catalog overview for load_scenario / JSONC comment gate. */
```

```text
/* SRAAM5 INLAR1.ASC: 5-DOF target-centered attack engagement.
   Catalog overview for load_scenario / JSONC comment gate. */
```

```text
/* SRAAM5 INLAR1 maut=11 variant: alpha/beta hold control case.
   Catalog overview for load_scenario / JSONC comment gate. */
```

```text
/* SRAAM5 INLAR1 mturn=1 variant: bank-to-turn kinematics case.
   Catalog overview for load_scenario / JSONC comment gate. */
```

```text
/* SRAAM5 INLAR1 ntag variant: AI acquisition radar NTAG engagement.
   Catalog overview for load_scenario / JSONC comment gate. */
```

Do not change JSON bodies except the leading comment. Keep JSONC parseable.

- [ ] **Step 3: Confirm GREEN**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_jsonc.py::test_catalog_scenarios_have_overview_comment -q`

Expected: PASS (`missing == []`).

- [ ] **Step 4: Commit (only if the user asks)**

```bash
git add Python/cases/rocket3/inlaunch.jsonc Python/cases/sraam5/inlar1.jsonc Python/cases/sraam5/inlar1_maut11.jsonc Python/cases/sraam5/inlar1_mturn1.jsonc Python/cases/sraam5/inlar1_ntag.jsonc
git commit -m "docs(cases): add catalog overview comments to five scenarios"
```

---

### Task 7: SRAAM5 registers `intercept` — update expected list and `inlar1.jsonc`

**Files:**
- Modify: `Python/tests/unit/test_sraam5_skeleton.py` (`EXPECTED_MODULES`)
- Modify: `Python/cases/sraam5/inlar1.jsonc` (modules list — append `intercept` after `rotations`)
- Test: `test_sraam5_family_registers_and_load_scenario`

**Interfaces:**
- Consumes: `Sraam5.modules` already ends with `Sraam5Intercept()` (`name == "intercept"`); sibling decks `inlar1_mseek5.jsonc` / `inlar1_mterm1.jsonc` already list `"intercept"`.
- Produces: `EXPECTED_MODULES` and `cfg.modules` from `inlar1.jsonc` both end with `"intercept"`, matching the vehicle.

**Why this (not unregister):** Intercept is intentional (SHAZAM / DTCT Band D). Skeleton test still expects the pre-intercept module list; `inlar1.jsonc` was never updated when the vehicle gained the module. Updating **only** `EXPECTED_MODULES` would fail the `cfg.modules` assert — both must change together.

- [ ] **Step 1: Confirm RED**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_sraam5_skeleton.py::test_sraam5_family_registers_and_load_scenario -q`

Expected: FAIL — vehicle modules Left contains one more item: `'intercept'` (cfg currently matches the old list without intercept).

- [ ] **Step 2: Add intercept to expected list and scenario**

In `test_sraam5_skeleton.py`:

```python
EXPECTED_MODULES = [
    "target",
    "environment",
    "seeker",
    "ai_radar",
    "ins",
    "guidance",
    "control",
    "aerodynamics",
    "propulsion",
    "forces",
    "newton",
    "rotations",
    "intercept",
]
```

In `cases/sraam5/inlar1.jsonc`, after the `rotations` module object, append (same shape as `inlar1_mterm1.jsonc`):

```json
    {
      "name": "intercept",
      "phases": [
        "def",
        "init",
        "exec"
      ]
    }
```

Keep valid JSON commas. Do not remove `Sraam5Intercept` from `vehicle.py`.

- [ ] **Step 3: Confirm GREEN**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_sraam5_skeleton.py::test_sraam5_family_registers_and_load_scenario -q`

Expected: PASS (both `cfg.modules` and `vehicle.modules` match).

- [ ] **Step 4: Commit (only if the user asks)**

```bash
git add Python/tests/unit/test_sraam5_skeleton.py Python/cases/sraam5/inlar1.jsonc
git commit -m "test(sraam5): expect intercept module on vehicle and inlar1"
```

---

### Task 8: UPDATES.md `0.191.2` + full unit suite green

**Files:**
- Modify: `UPDATES.md` (newest entry on top)
- Verify: full Python unit suite

**Interfaces:**
- Consumes: Tasks 1–7 green on their focused commands.
- Produces: `0.191.2` entry summarizing the eight follow-ups; suite green count.

- [ ] **Step 1: Bump UPDATES**

Insert at top of `UPDATES.md` (below `# Updates`):

```markdown
## 0.191.2 - Suite red follow-ups after unported abilities
- Quality: exempt rising `look_up`; allow `for name in store.names()` while banning membership.
- Cruise5 tests expect `Cruise5Environment`; HYPER6 maut 32/42 assert pitch-rate; seeker plants STBIK/VTBIK on execute.
- Catalog overview comments on five scenarios; SRAAM5 expects `intercept` on vehicle and `inlar1`.
```

- [ ] **Step 2: Full unit suite**

Run: `cd Python && PYTHONPATH=src:tools python -m pytest -q`

Expected: all previously failing eight pass; no new fails. (Prior baseline was 8 failed, 2764 passed, 16 skipped — expect 0 failed.)

- [ ] **Step 3: Commit (only if the user asks)**

```bash
git add UPDATES.md
git commit -m "docs: UPDATES 0.191.2 suite red follow-ups"
```

---

## Self-review

1. **Spec coverage:** All eight failures have a task (1 look_up, 2 membership, 3 Cruise5×2, 4 hyper6 maut, 5 STBIK, 6 jsonc×5, 7 sraam5 intercept) plus UPDATES/suite (8). Flatten minors listed only under Global Constraints.
2. **Placeholder scan:** No TBD/TODO; each step has concrete code or comment text and exact pytest commands.
3. **Type/name consistency:** `Cruise5Environment`, `STBIK`/`VTBIK`, `dlde`, `EXPECTED_MODULES` + `intercept`, `look_up` `>=` match production and peer tests. Task 4 renames the test; Task 8 suite command uses the new name via full collection.
4. **Direction picks (no forks):** look_up exempt; lint narrow; Cruise5Environment; maut 32/42 ported contract; STBIK on execute; five comment paths listed; intercept kept and expected (plus `inlar1.jsonc` so cfg matches).
