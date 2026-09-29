# Traj Flatten Non-Scalar Packet Values Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make CADAC traj (and sibling comscrn) writers emit length-3 packet vectors such as lowercase `sbii` as three float columns without TypeError, matching the existing uppercase-vector column shape.

**Architecture:** Packet writers today treat a variable as a vector only when the name’s first character is uppercase (`_is_vector_name` / C++ `isupper`). Round3 Python publishes inertial position as lowercase `sbii` holding a 3-vector ndarray, so `_flatten_packet_values` falls through to `float(value)` and raises. Detect vectors by name **or** by value shape (indexable length 3, not a string), and use that single predicate for `nvariables`, banner labels, and data flatten so column counts stay consistent. Store-based writers (`stat` / `tabout` / `scrn`) already use `field.type == "vec"` and are out of scope.

**Tech Stack:** Python 3, pytest, numpy arrays / sequences in `Packet.vars`.

**Spec:** Whole-branch review Minor #5 — pre-existing CRUISE5 one-step traj smoke fails in `traj.py` `_flatten_packet_values` on non-scalar packet values (e.g. `sbii`). Sources: `.superpowers/sdd/2026-09-29-cadac-unported-abilities/whole-branch-review.md`, `whole-branch-fix-report.md`. C++ reference (read-only): `CADAC_Simulations/CRUISE5_250115/CRUISE5/global_functions.cpp` `traj_data` expands when `isupper(vector[0])` into three `VEC.get_loc(i,0)` fields; C++ Round3 names the variable `SBII`.

## Global Constraints

- One bug only: traj + comscrn (shared name-only flatten heuristic). Do not reopen weather_deck, column-width formatting, Cruise5Environment packing, or other accepted deferrals.
- Do not edit `CADAC_Simulations/`, `docs/superpowers/plans/2026-09-29-cadac-unported-abilities.md`, or `Python/tests/unit/test_cpp_gap_audit.py`.
- Do not rename Round3 `sbii` → `SBII` or change vehicle packing; fix the writers.
- Vector CADAC shape: three columns (components 0..2); banner labels `stem1_vid` / `stem2_vid` / `stem3_vid` (same path as uppercase names today).
- TDD: failing unit first, watch RED, minimal fix, watch GREEN. Command form: `cd Python && PYTHONPATH=src:tools python -m pytest <files> -q`.
- Commit steps run only if the user asks for a commit in that session.
- Bump root `UPDATES.md` (bug fix → subsubver under current `0.191.x`).

## File structure

- `Python/src/cadac/io/traj.py` — `_is_vector_entry(name, value)`, update `nvariables`, `write_traj_banner`, `_flatten_packet_values`.
- `Python/src/cadac/io/comscrn.py` — same predicate + banner/`_flatten_values` (sibling of traj flatten).
- `Python/tests/unit/test_traj_flatten_vector.py` — new unit proofs for lowercase `sbii` and uppercase `SBII`.
- `Python/tests/unit/test_comscrn_flatten_vector.py` — new unit proofs for the same packet shape via comscrn.
- `Python/tests/unit/test_cruise5_one_step.py` — regression only (smoke must pass after fix); do not expand its assertions beyond health/alt.
- `UPDATES.md` — `0.191.1` (or next subsubver if bumped meantime).

---

### Task 1: Traj flatten detects length-3 packet vectors

**Files:**
- Create: `Python/tests/unit/test_traj_flatten_vector.py`
- Modify: `Python/src/cadac/io/traj.py` (`_is_vector_name` kept; add `_is_vector_entry`; update `nvariables`, `write_traj_banner`, `_flatten_packet_values`)

**Interfaces:**
- Consumes: `Packet` from `cadac.kernel.combus`; existing `_truncate_label`, `_FIELD_WIDTH`, `_ACROSS`.
- Produces:
  - `_is_vector_entry(name: str, value: object) -> bool` — True if `_is_vector_name(name)` **or** value is a non-string sequence with `len(value) == 3`.
  - `_flatten_packet_values(packet, *, skip_time: bool) -> list[float]` — for vector entries, `extend(float(value[i]) for i in range(3))`; scalars unchanged (ints→float).
  - `nvariables(combus)` / `write_traj_banner` count and label using `_is_vector_entry` so banner columns match data columns.

- [x] **Step 1: Write the failing unit test**

Create `Python/tests/unit/test_traj_flatten_vector.py`:

```python
"""traj.asc flatten must expand length-3 packet vectors (incl. lowercase sbii)."""

from io import StringIO

import numpy as np

from cadac.io.traj import (
    _flatten_packet_values,
    nvariables,
    write_traj_banner,
    write_traj_data,
)
from cadac.kernel.combus import Packet


def _packet_sbii():
    return Packet(
        name="c1",
        type="CRUISE3",
        status=1,
        vars={
            "time": 0.0,
            "sbii": np.array([1.0, 2.0, 3.0]),
            "alt": 1000.0,
        },
    )


def test_flatten_expands_lowercase_sbii_to_three_floats():
    values = _flatten_packet_values(_packet_sbii(), skip_time=True)
    assert values == [1.0, 2.0, 3.0, 1000.0]


def test_flatten_still_expands_uppercase_SBII():
    packet = Packet(
        name="m1",
        type="HYPER6",
        status=1,
        vars={"time": 0.0, "SBII": np.array([4.0, 5.0, 6.0])},
    )
    assert _flatten_packet_values(packet, skip_time=True) == [4.0, 5.0, 6.0]


def test_nvariables_and_banner_count_sbii_as_three_columns():
    combus = [_packet_sbii()]
    # names: time, sbii(+2), alt → 1 + 3 + 1 = 5 after shared-time rule on single packet
    assert nvariables(combus) == 5
    stream = StringIO()
    write_traj_banner(stream, "t", combus)
    text = stream.getvalue()
    assert "sbii1_" in text and "sbii2_" in text and "sbii3_" in text


def test_write_traj_data_emits_sbii_components_without_typeerror():
    stream = StringIO()
    write_traj_data(stream, [_packet_sbii()], merge=False)
    text = stream.getvalue()
    assert "1.0" in text and "2.0" in text and "3.0" in text
    assert "1000" in text
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_traj_flatten_vector.py -q
```

Expected: FAIL — `TypeError: only 0-dimensional arrays can be converted to Python scalars` (or AssertionError on nvariables/banner if flatten were patched alone). At least `test_flatten_expands_lowercase_sbii_to_three_floats` and `test_write_traj_data_emits_sbii_components_without_typeerror` must fail.

- [x] **Step 3: Minimal implementation in traj.py**

Replace name-only vector checks used for counting/flattening with a value-aware helper. Keep `_is_vector_name` for the uppercase CADAC rule.

```python
def _is_vector_entry(name: str, value) -> bool:
    if _is_vector_name(name):
        return True
    if isinstance(value, (str, bytes, bytearray)):
        return False
    try:
        return len(value) == 3
    except TypeError:
        return False
```

Update `nvariables`:

```python
def nvariables(combus: list[Packet]) -> int:
    packets = _active_packets(combus)
    if not packets:
        return 0
    total = 0
    for packet in packets:
        names = _packet_names(packet)
        nvec = sum(
            1 for name in names if _is_vector_entry(name, packet.vars[name])
        )
        total += len(names) + 2 * nvec
    return total - (len(packets) - 1)
```

In `write_traj_banner`, change the vector branch from `_is_vector_name(name)` to `_is_vector_entry(name, packet.vars[name])`.

In `_flatten_packet_values`:

```python
def _flatten_packet_values(packet: Packet, *, skip_time: bool) -> list[float]:
    values: list[float] = []
    for name, value in packet.vars.items():
        if skip_time and name == "time":
            continue
        if _is_vector_entry(name, value):
            values.extend(float(value[i]) for i in range(3))
        elif isinstance(value, int) and not isinstance(value, bool):
            values.append(float(value))
        else:
            values.append(float(value))
    return values
```

Do not change field widths, merge `-1.0` endblock, or `packets_from_vehicles`.

- [x] **Step 4: Run test to verify it passes**

Run:

```bash
cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_traj_flatten_vector.py tests/unit/test_y_traj.py -q
```

Expected: PASS (all new traj flatten tests + existing `test_y_traj_writes_traj_asc`).

- [x] **Step 5: Commit (only if the user asks)**

```bash
git add Python/tests/unit/test_traj_flatten_vector.py Python/src/cadac/io/traj.py
git commit -m "$(cat <<'EOF'
fix(traj): expand length-3 packet vectors like sbii

EOF
)"
```

Skip this step unless the user explicitly requested a commit.

---

### Task 2: Comscrn sibling flatten (same packet heuristic)

**Files:**
- Create: `Python/tests/unit/test_comscrn_flatten_vector.py`
- Modify: `Python/src/cadac/io/comscrn.py` (`_is_vector_entry`; update label loop and `_flatten_values`)

**Interfaces:**
- Consumes: same `Packet` shape as Task 1; `_is_vector_name` remains.
- Produces: `_is_vector_entry` identical semantics to Task 1; `_flatten_values(packet) -> list[float]` expands length-3 entries; banner labels write `stem1`/`stem2`/`stem3` for those entries.

- [x] **Step 1: Write the failing unit test**

Create `Python/tests/unit/test_comscrn_flatten_vector.py`:

```python
"""comscrn.asc flatten must expand length-3 packet vectors (incl. lowercase sbii)."""

from io import StringIO

import numpy as np

from cadac.io.comscrn import _flatten_values, write_comscrn_data
from cadac.kernel.combus import Packet


def test_comscrn_flatten_expands_lowercase_sbii():
    packet = Packet(
        name="c1",
        type="CRUISE3",
        status=1,
        vars={
            "time": 0.0,
            "sbii": np.array([1.0, 2.0, 3.0]),
            "alt": 1000.0,
        },
    )
    assert _flatten_values(packet) == [1.0, 2.0, 3.0, 1000.0]


def test_write_comscrn_data_emits_sbii_components_without_typeerror():
    packet = Packet(
        name="c1",
        type="CRUISE3",
        status=1,
        vars={
            "time": 0.0,
            "sbii": np.array([1.0, 2.0, 3.0]),
            "alt": 1000.0,
        },
    )
    stream = StringIO()
    write_comscrn_data(stream, [packet], 0.0)
    text = stream.getvalue()
    assert "sbii1" in text and "sbii2" in text and "sbii3" in text
    assert "1.0" in text and "2.0" in text and "3.0" in text
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_comscrn_flatten_vector.py -q
```

Expected: FAIL with TypeError on `float(value)` for `sbii` (same as traj).

- [x] **Step 3: Minimal implementation in comscrn.py**

Add the same `_is_vector_entry` as Task 1 (duplicate locally; do not invent a shared util unless both files already share one — YAGNI).

```python
def _is_vector_entry(name: str, value) -> bool:
    if _is_vector_name(name):
        return True
    if isinstance(value, (str, bytes, bytearray)):
        return False
    try:
        return len(value) == 3
    except TypeError:
        return False
```

In `write_comscrn_data` label loop, replace `_is_vector_name(name)` with `_is_vector_entry(name, first.vars[name])`.

In `_flatten_values`, replace `_is_vector_name(name)` with `_is_vector_entry(name, value)` and keep the three-component `extend`.

- [x] **Step 4: Run test to verify it passes**

Run:

```bash
cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_comscrn_flatten_vector.py tests/unit/test_y_comscrn.py -q
```

Expected: PASS.

- [x] **Step 5: Commit (only if the user asks)**

```bash
git add Python/tests/unit/test_comscrn_flatten_vector.py Python/src/cadac/io/comscrn.py
git commit -m "$(cat <<'EOF'
fix(comscrn): expand length-3 packet vectors like sbii

EOF
)"
```

Skip unless the user explicitly requested a commit.

---

### Task 3: Cruise one-step smoke + UPDATES

**Files:**
- Modify: `UPDATES.md` (new top entry)
- Verify only: `Python/tests/unit/test_cruise5_one_step.py` (no assertion edits required if smoke already checks alt/health)

**Interfaces:**
- Consumes: Task 1 traj fix (input_1.jsonc has `"traj": true`, so `run_scenario` hits `write_traj_banner` / `write_traj_data`).
- Produces: green cruise one-step suite; `UPDATES.md` `0.191.1` note.

- [x] **Step 1: Run the previously failing smoke + related cruise units**

Run:

```bash
cd Python && PYTHONPATH=src:tools python -m pytest tests/unit/test_cruise5_one_step.py tests/unit/test_traj_flatten_vector.py tests/unit/test_comscrn_flatten_vector.py -q
```

Expected: PASS, including `test_smoke_one_step_alt_finite_and_health` (previously TypeError in traj flatten).

If smoke still fails for a non-traj reason, stop and report — do not reopen Environment/weather packing.

- [x] **Step 2: Update UPDATES.md**

Add on top (adjust version if `0.191.1` already exists):

```markdown
## 0.191.1 - Traj/comscrn flatten length-3 packet vectors
- `traj` and `comscrn` treat length-3 packet values (e.g. lowercase `sbii`) as vectors and emit three float columns; cruise one-step traj smoke no longer TypeErrors.
- Tests: `test_traj_flatten_vector.py`, `test_comscrn_flatten_vector.py`; regression `test_cruise5_one_step.py`.
```

Do not edit `README.md` (architecture unchanged).

- [x] **Step 3: Commit (only if the user asks)**

```bash
git add UPDATES.md Python/tests/unit/test_traj_flatten_vector.py Python/tests/unit/test_comscrn_flatten_vector.py Python/src/cadac/io/traj.py Python/src/cadac/io/comscrn.py
git commit -m "$(cat <<'EOF'
fix(io): flatten lowercase length-3 combus vectors in traj/comscrn

EOF
)"
```

Prefer one commit for the whole fix if the user asks to commit after all tasks; otherwise skip.

---

## Self-review

1. **Spec coverage:** Review Minor #5 (traj `_flatten_packet_values` TypeError on `sbii`) → Task 1. Sibling same-heuristic writer → Task 2. Cruise smoke regression cited in fix-report → Task 3. Out-of-scope deferrals explicitly excluded in Global Constraints.
2. **Placeholder scan:** No TBD/TODO; tests and implementation code are concrete; pytest commands named.
3. **Type consistency:** `_is_vector_entry(name, value) -> bool` and three-component flatten are the same contract in Tasks 1–2; `nvariables` adds `+2` per vector entry to match banner expansion.
