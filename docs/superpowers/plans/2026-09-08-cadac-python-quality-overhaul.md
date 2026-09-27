# CADAC Python Quality Overhaul Implementation Plan

**Status:** executed 2026-09-08; merged to `main` (`aa90bd2`). Tasks 1–15 complete. Do not re-dispatch.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. After **each** task: Grok non-fast reviewer (`cursor-grok-4.6-high`). After Task 12: whole-branch reviewer. Then Tasks 13–15 (unit gate, e2e matrix, metrics re-survey).

**Goal:** Move `cadac` from the 8 Sep 2026 quality survey to the parity-safe target (faster store/executive, shared 3-vector math, cached lookup, typed kernel, readable EOM) without changing JSONC, store keys, or harvested C++ plot CSVs.

**Architecture:** Keep the CADAC module chain and named `StateStore`. Speed comes from skip-coerce, O(1) membership, once-bound `execute` lists, unrolled 3×3 ijk, and lookup index cache — not from vectorizing time or Monte Carlo. Shared `cadac.math` helpers replace copied `_skew` / `_cadac_sign`. EOM grows docstrings and fused kernels; vehicles keep CADAC names.

**Tech Stack:** Python >= 3.11, numpy, pytest. Dev: ruff. Work in `Python/`. pytest `pythonpath` is `["src", "tools"]`.

**Spec:** `docs/superpowers/specs/2026-09-08-cadac-python-quality-overhaul-design.md`

**Survey (baseline):** [CADAC Python quality survey](/home/valentin/.cursor/projects/home-valentin-Books-2025-Zeipfel-Modeling-and-simulation-of-aerospace-vehicles/canvases/cadac-python-quality-survey.canvas.tsx)

## Global Constraints

- Spec + parent `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
- Do not edit `CADAC_Simulations/` (no `.cpp` / `.hpp`)
- Do not rename store keys, plot columns, or combus names
- Do not replace Round6/ROCKET6/HYPER6 INS `cadac_matmul` / `cadac_inverse` with `numpy @` / `linalg.inv`
- Do not unify Flat6 `np.linalg.inv` with `cadac_inverse`
- Do not vectorize across `sim_time` or batch `nmonte`
- No new runtime dependency (no numba, jax, scipy)
- Unit `rtol=1e-12`, `atol=1e-14` unless an existing test uses a tighter `array_equal`
- CSV e2e: existing tols in each e2e file (typically `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`)
- Implementer + reviewer: `cursor-grok-4.6-high` (no Fast, no non-Grok)
- TDD: failing test → implement → pass. Reviewer must see the test command and output
- Parent does **not** `git commit` unless the user asked for commits on this run; still prepare the message in the Commit step
- Each task bumps `UPDATES.md` (newest on top). Use `0.169.N` starting at `0.169.0`. Update `README.md` only if architecture text is wrong (Task 15)
- `Field.value` mutation in `StateStore.define` is existing behavior; `ModuleBase` must pass a **new** `Field` instance per vehicle

## File map

- Create: `Python/tools/cadac_quality/__init__.py`, `metrics.py`, `baseline.json`
- Create: `Python/tests/unit/test_cadac_quality_metrics.py`
- Modify: `Python/src/cadac/kernel/state.py`, `executive.py`, `module.py`
- Modify: `Python/src/cadac/math/frames.py`, `Python/src/cadac/math/__init__.py`
- Modify: `Python/src/cadac/tables/lookup.py`
- Modify: `Python/src/cadac/eom/{flat6,round6,flat3,flat0,round3,rotor}.py`
- Modify: every `Python/src/cadac/**/*.py` that copies `_skew` / `_cadac_sign` or uses `in store.names()` for membership
- Modify: `Python/pyproject.toml` (ruff)
- Test: `Python/tests/unit/test_state.py`, `test_executive.py`, `test_frames.py`, `test_lookup_*.py`, `test_flat6_*.py`, plus new tests named below
- E2E: `Python/tests/e2e/test_*.py` (15 files; skip only if golden missing)
- Final: canvas `cadac-python-quality-survey.canvas.tsx`, `UPDATES.md`, `README.md` if needed

---

### Task 1: Quality metrics scanner and baseline

**Files:**
- Create: `Python/tools/cadac_quality/__init__.py` (empty or docstring)
- Create: `Python/tools/cadac_quality/metrics.py`
- Create: `Python/tools/cadac_quality/baseline.json` (whatever this scanner reports on HEAD at task time)
- Test: `Python/tests/unit/test_cadac_quality_metrics.py`

**Interfaces:**
- Consumes: `Python/src/cadac` tree
- Produces: `scan_cadac(root: Path) -> dict` with keys `files`, `loc`, `code`, `store_get`, `store_set`, `store_names`, `look_up`, `np_zeros`, `np_array`, `typed_defs`, `untyped_defs`, `empty_terminate`, `empty_initialize`, `skew_defs`, `cadac_sign_defs`. `write_metrics(path, data)`, `load_metrics(path)`. CLI: `PYTHONPATH=src:tools python -m cadac_quality.metrics --root src/cadac --out tools/cadac_quality/baseline.json` from `Python/`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from cadac_quality.metrics import load_metrics, scan_cadac

ROOT = Path(__file__).resolve().parents[1] / "src" / "cadac"
BASELINE = Path(__file__).resolve().parents[1] / "tools" / "cadac_quality" / "baseline.json"


def test_scan_has_required_keys():
    data = scan_cadac(ROOT)
    for key in (
        "files",
        "loc",
        "store_get",
        "store_set",
        "store_names",
        "look_up",
        "np_zeros",
        "np_array",
        "typed_defs",
        "untyped_defs",
        "empty_terminate",
        "empty_initialize",
        "skew_defs",
        "cadac_sign_defs",
    ):
        assert key in data
        assert isinstance(data[key], int)
        assert data[key] >= 0


def test_scan_matches_checked_in_baseline():
    data = scan_cadac(ROOT)
    base = load_metrics(BASELINE)
    for key in base:
        assert data[key] == base[key], key
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && pytest tests/unit/test_cadac_quality_metrics.py -v`

Expected: FAIL (`cadac_quality` not found or `baseline.json` missing)

- [ ] **Step 3: Write scanner and freeze baseline**

Count with the same regexes as the survey: `store.get(`, `store.set(`, `store.names(`, `look_up(`, `np.zeros(`, `np.array(`. `skew_defs` = files whose text contains `def _skew` or `def skew(` at line start after optional spaces. `cadac_sign_defs` = `def _cadac_sign` or `def cadac_sign`. Typed/untyped via `ast` on `FunctionDef`/`AsyncFunctionDef` (`returns` or any `args.annotation`). Empty `terminate`/`initialize` = body is only `Pass` (and optional docstring `Expr`). Write `baseline.json` from a real scan of current `src/cadac` (do not hand-type the 8 Sep numbers if the tree moved; the file is the freeze).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd Python && pytest tests/unit/test_cadac_quality_metrics.py -v`

Expected: PASS

- [ ] **Step 5: UPDATES + Commit (if user approved commits)**

`UPDATES.md` top: `## 0.169.0 - Quality metrics scanner`. Message: `feat: freeze cadac quality metrics baseline`

---

### Task 2: StateStore skip-coerce, contains, get_optional

**Files:**
- Modify: `Python/src/cadac/kernel/state.py`
- Test: `Python/tests/unit/test_state.py` (add tests; keep existing)

**Interfaces:**
- Consumes: existing `Field`, `StateStore.define` / `get` / `set` / `names` / `field`
- Produces: `StateStore.__contains__(self, name: str) -> bool`. `get_optional(self, name: str, default=None)`. `set` still type-checks; skips `int()` / `float()` / `np.asarray` when `type(value) is int` (int fields), `type(value) is float` (real), or `isinstance(value, np.ndarray) and value.shape == (3,) and value.dtype == float` (vec) / `shape == (3, 3)` (mat). Lists and wrong dtypes still coerce or raise. `names()` unchanged (`list` of keys in define order).

- [ ] **Step 1: Write the failing tests** (append to `test_state.py`)

```python
def test_contains_and_get_optional():
    s = StateStore()
    assert "time" not in s
    assert s.get_optional("time") is None
    assert s.get_optional("time", 0.0) == 0.0
    s.define(Field("time", 0.0, "real", "exec", "environment"))
    assert "time" in s
    assert s.get_optional("time") == 0.0
    assert s.names() == ["time"]


def test_set_skips_float_and_matching_vec():
    s = StateStore()
    s.define(Field("time", 0.0, "real", "exec", "environment"))
    s.define(Field("sbii", np.zeros(3), "vec", "state", "newton"))
    s.set("time", 1.5)
    assert type(s.get("time")) is float
    vec = np.array([1.0, 2.0, 3.0])
    s.set("sbii", vec)
    np.testing.assert_allclose(s.get("sbii"), [1.0, 2.0, 3.0])
    s.set("time", 2)
    assert type(s.get("time")) is float
    assert s.get("time") == 2.0
    s.set("sbii", [4.0, 5.0, 6.0])
    np.testing.assert_allclose(s.get("sbii"), [4.0, 5.0, 6.0])
```

Keep `test_vec_mat_shape`, `test_int_vs_real`, `test_unknown_get_set_raises_keyerror`.

- [ ] **Step 2: Run new tests to verify they fail**

Run: `cd Python && pytest tests/unit/test_state.py::test_contains_and_get_optional tests/unit/test_state.py::test_set_skips_float_and_matching_vec -v`

Expected: FAIL (`__contains__` / `get_optional` missing)

- [ ] **Step 3: Implement**

```python
def __contains__(self, name):
    return name in self._fields

def get_optional(self, name, default=None):
    field = self._fields.get(name)
    if field is None:
        return default
    return field.value

def set(self, name, value):
    field = self._fields[name]
    field.value = self._coerce(field.type, value)
```

In `_coerce`, if `ftype == "real"` and `type(value) is float`: return `value`. If `ftype == "int"` and `type(value) is int`: return `value`. If vec/mat and `isinstance(value, np.ndarray)` with the right `shape` and `dtype == float` (accept `float64`): return `value` without `asarray`. Else existing coerce/raise.

- [ ] **Step 4: Run tests**

Run: `cd Python && pytest tests/unit/test_state.py -v`

Expected: PASS (old + new)

- [ ] **Step 5: UPDATES `0.169.1` + Commit if approved**

---

### Task 3: Executive binds modules once and reuses SimContext

**Files:**
- Modify: `Python/src/cadac/kernel/executive.py`
- Test: `Python/tests/unit/test_executive.py`

**Interfaces:**
- Consumes: existing `run_loop(vehicles, modules_by_vehicle, module_order, end_time, int_step, on_step=None)`
- Produces: same signature and loop semantics (`sim_time <= end_time + int_step`, health/status skip, `ctx.int_step` resize, skip missing module names). Internally: one `list[tuple[object, ...]]` of execute-callables ordered by `module_order` **before** the time loop. One `SimContext` per vehicle (or one reused object with fields assigned each step). Do not rebuild `{module.name: module}` inside the time loop.

- [ ] **Step 1: Write the failing test**

```python
def test_run_loop_does_not_rebuild_name_map_each_step():
    vehicle = _Vehicle()
    calls = {"maps": 0}

    class _Named:
        name = "watch"

        def define(self, vehicle):
            pass

        def initialize(self, vehicle, ctx):
            pass

        def execute(self, vehicle, ctx):
            pass

        def terminate(self, vehicle, ctx):
            pass

    class _Dict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getitem__(self, key):
            if key is vehicle or key == vehicle:
                calls["maps"] += 1
            return super().__getitem__(key)

    dummy = _Named()
    modules = _Dict({vehicle: [dummy]})
    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle=modules,
        module_order=["watch"],
        end_time=0.2,
        int_step=0.1,
    )
    # 4 time stations (0, 0.1, 0.2, 0.3). Lookup once at bind, not once per step.
    assert calls["maps"] == 1
```

If wrapping `dict.__getitem__` is too brittle, instead add a module that records `id(ctx)` and assert the same id on every step:

```python
def test_run_loop_reuses_simcontext_per_vehicle():
    vehicle = _Vehicle()
    ids = []

    class _Watch:
        name = "watch"

        def define(self, vehicle):
            pass

        def initialize(self, vehicle, ctx):
            pass

        def execute(self, vehicle, ctx):
            ids.append(id(ctx))

        def terminate(self, vehicle, ctx):
            pass

    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [_Watch()]},
        module_order=["watch"],
        end_time=0.1,
        int_step=0.1,
    )
    assert len(ids) >= 2
    assert len(set(ids)) == 1
```

Ship **both** tests. Keep `test_modules_follow_module_order`, `test_skip_module_absent_on_vehicle`, `test_run_loop_adopts_ctx_int_step`.

- [ ] **Step 2: Run new tests to verify fail**

Run: `cd Python && pytest tests/unit/test_executive.py::test_run_loop_reuses_simcontext_per_vehicle tests/unit/test_executive.py::test_run_loop_does_not_rebuild_name_map_each_step -v`

Expected: FAIL (new context every step / many map hits)

- [ ] **Step 3: Implement bind-once**

Before `while sim_time <= ...`:

```python
chains = []
for vehicle in vehicles:
    named = {module.name: module for module in modules_by_vehicle[vehicle]}
    chains.append(tuple(named[name] for name in module_order if name in named))
contexts = [
    SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=slot,
    )
    for slot, vehicle in enumerate(vehicles)
]
```

In the loop, set `ctx.sim_time`, `ctx.int_step` (current loop `int_step`), `ctx.event_time`, then `for module in chains[slot]: module.execute(vehicle, ctx)`. Keep `_publish` and `on_step`. After execute, `int_step = ctx.int_step` as today (last vehicle wins — current behavior).

- [ ] **Step 4: Run executive tests + smoke e2e**

Run: `cd Python && pytest tests/unit/test_executive.py -v && pytest tests/e2e/test_hyper3_climb.py tests/e2e/test_falcon6_gamma.py tests/e2e/test_agm6_freeflight.py -q`

Expected: unit PASS. E2E PASS or skip-if-no-golden (same as today).

- [ ] **Step 5: UPDATES `0.169.2` + Commit if approved**

---

### Task 4: Membership via `in store` (drop `in store.names()`)

**Files:**
- Modify: every `Python/src/cadac/**/*.py` that uses `"…" in store.names()` or `n in names` after `names = store.names()` **for membership**
- Do **not** remove `names()` where the list is the value (plot/combus column lists)
- Test: `Python/tests/unit/test_cadac_quality_metrics.py` (add assertion) and `Python/tests/unit/test_state.py` (already has contains)

**Interfaces:**
- Consumes: Task 2 `__contains__`
- Produces: production membership is `"mfreeze" in store`. Scanner `store_names` count must drop vs `baseline.json` (membership sites gone; leftover `names()` only where the list is used)

- [ ] **Step 1: Write the failing test**

```python
def test_src_membership_does_not_use_names_list():
    root = Path(__file__).resolve().parents[1] / "src" / "cadac"
    hits = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if "in store.names()" in line:
                hits.append(f"{path}:{i}:{stripped}")
            if "in names" in line and "store.names()" in text:
                if "for n in names" in line or "n in names" in line:
                    hits.append(f"{path}:{i}:{stripped}")
    assert hits == [], "use `name in store`:\n" + "\n".join(hits)
```

Tune the second check so `packet_from_store` / plot code that iterates `for name in names` (a caller-supplied list, not `store.names()`) is not flagged. Only flag `store.names()` membership. A strict rule: **no** substring `in store.names()` in `src/cadac`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Python && pytest tests/unit/test_cadac_quality_metrics.py::test_src_membership_does_not_use_names_list -v`

Expected: FAIL with a non-empty `hits` list (eom/flat6, vehicles, …)

- [ ] **Step 3: Replace membership**

`if "mfreeze" in store.names():` → `if "mfreeze" in store:`.  
`names = store.names()` + `if all(n in names for n in (…))` → `if all(n in store for n in (…))`.  
Keep `names()` in `test_state.py` and in plot helpers that need the full key list.

- [ ] **Step 4: Run tests**

Run: `cd Python && pytest tests/unit/test_cadac_quality_metrics.py tests/unit/test_state.py tests/unit/test_flat6_kinematics.py tests/unit/test_flat6_euler.py tests/unit/test_flat3_newton_step.py -q`

Expected: PASS

- [ ] **Step 5: UPDATES `0.169.3` + Commit if approved**

---

### Task 5: Shared `cadac_sign` and `skew`

**Files:**
- Modify: `Python/src/cadac/math/frames.py`, `Python/src/cadac/math/__init__.py`
- Modify: every production file with `def _cadac_sign` or `def _skew` (21 skew, 9 sign at survey time)
- Test: `Python/tests/unit/test_frames.py`

**Interfaces:**
- Consumes: none new
- Produces: `cadac_sign(variable) -> int` with `variable < 0.0 → -1` else `+1` (not `numpy.sign`; `+0` is `+1`). `skew(vec) -> np.ndarray` shape `(3, 3)`, dtype float, `[[0,-z,y],[z,0,-x],[-y,x,0]]`. Re-exports from `cadac.math` if `__init__.py` is the package surface; otherwise import `from cadac.math.frames import cadac_sign, skew`.

- [ ] **Step 1: Write the failing tests**

```python
from cadac.math.frames import cadac_sign, skew

def test_cadac_sign_matches_cpp():
    assert cadac_sign(-1.0) == -1
    assert cadac_sign(0.0) == 1
    assert cadac_sign(2.5) == 1


def test_skew_cross_product_matrix():
    k = skew(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(
        k,
        [[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]],
        rtol=1e-12,
        atol=1e-14,
    )
```

```python
def test_only_one_skew_and_sign_definition():
    root = Path(__file__).resolve().parents[1] / "src" / "cadac"
    skews, signs = [], []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^def (_)?skew\(", text, re.M):
            skews.append(str(path.relative_to(root)))
        if re.search(r"^def (_)?cadac_sign\(", text, re.M):
            signs.append(str(path.relative_to(root)))
    assert skews == ["math/frames.py"]
    assert signs == ["math/frames.py"]
```

Put the uniqueness test in `test_cadac_quality_metrics.py` or `test_frames.py`.

- [ ] **Step 2: Run tests to verify fail**

Run: `cd Python && pytest tests/unit/test_frames.py::test_cadac_sign_matches_cpp tests/unit/test_frames.py::test_skew_cross_product_matrix tests/unit/test_cadac_quality_metrics.py::test_only_one_skew_and_sign_definition -v`

Expected: FAIL (import error or many defs)

- [ ] **Step 3: Implement and replace copies**

Add functions to `frames.py`. Replace each local `_skew` / `_cadac_sign` with the import. Do not change call-site argument order. Keep using `cadac_sign` (not `np.sign`) at every former `_cadac_sign` site.

- [ ] **Step 4: Run tests**

Run: `cd Python && pytest tests/unit/test_frames.py tests/unit/test_flat6_kinematics.py tests/unit/test_flat6_euler.py tests/unit/test_round6_kinematics.py tests/unit/test_round6_euler.py tests/unit/test_cadac_quality_metrics.py::test_only_one_skew_and_sign_definition -q`

Expected: PASS

- [ ] **Step 5: UPDATES `0.169.4` + Commit if approved**

---

### Task 6: `hypot3`, `quat_to_dcm`, `matvec3` and Flat6 kinematics

**Files:**
- Modify: `Python/src/cadac/math/frames.py`
- Modify: `Python/src/cadac/eom/flat6.py` (`Flat6Kinematics.execute`, `Flat6Environment` norm, `Flat6Newton` norms)
- Test: `Python/tests/unit/test_frames.py`, `Python/tests/unit/test_flat6_kinematics.py`

**Interfaces:**
- Consumes: `cadac_sign` (Task 5)
- Produces: `hypot3(vec) -> float` equal to `sqrt(x*x+y*y+z*z)` (not `np.linalg.norm`, same formula). `quat_to_dcm(q0, q1, q2, q3) -> np.ndarray (3,3)` using the **same nine expressions** as current `flat6.py` TBL. `matvec3(mat, vec) -> np.ndarray (3,)` as three dots (or `mat @ vec` only if a unit test `array_equal`s a reference ijk loop — prefer explicit loops for TBL @ vbal if that path must match C++; Flat6 currently uses `@`, keep `@` there unless a golden requires ijk). Flat6 still writes `q0..q3` keys.

- [ ] **Step 1: Write the failing tests**

```python
from cadac.math.frames import hypot3, quat_to_dcm

def test_hypot3_zero_and_3_4_12():
    assert hypot3(np.array([0.0, 0.0, 0.0])) == 0.0
    np.testing.assert_allclose(hypot3(np.array([3.0, 4.0, 12.0])), 13.0, atol=1e-14)


def test_quat_to_dcm_identity():
    dcm = quat_to_dcm(1.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(dcm, np.eye(3), atol=1e-14)
```

In `test_flat6_kinematics.py`, keep existing execute assertions. Add: after `execute`, `TBL` equals `quat_to_dcm(q0,q1,q2,q3)` from the store.

- [ ] **Step 2: Run tests to verify fail**

Run: `cd Python && pytest tests/unit/test_frames.py::test_hypot3_zero_and_3_4_12 tests/unit/test_frames.py::test_quat_to_dcm_identity -v`

Expected: FAIL (ImportError)

- [ ] **Step 3: Implement and switch Flat6**

Copy TBL assignments from `flat6.py` into `quat_to_dcm` (allocate `(3,3)` once, same order). `Flat6Kinematics.execute` calls `tbl = quat_to_dcm(q0, q1, q2, q3)` instead of nine lines. `dvba` / `dvbe` / ground-range use `hypot3` where the argument is a 3-vector. Quaternion integration stays four `integrate` calls (store keys unchanged). Do **not** pack q0..q3 into one store field.

- [ ] **Step 4: Run tests**

Run: `cd Python && pytest tests/unit/test_frames.py tests/unit/test_flat6_kinematics.py tests/unit/test_flat6_environment.py tests/unit/test_flat6_euler.py tests/unit/test_flat6_newton.py -q`

Expected: PASS (use the actual newton test filename if different: glob `test_flat6_*.py`)

- [ ] **Step 5: UPDATES `0.169.5` + Commit if approved**

---

### Task 7: Unroll 3×3 `cadac_matmul` (keep ijk)

**Files:**
- Modify: `Python/src/cadac/math/frames.py` (`cadac_matmul` only)
- Test: `Python/tests/unit/test_frames.py` (new) + existing `tests/unit/test_round6_kinematics.py::test_cadac_inverse_matches_cpp_adjoint_over_det_not_numpy`

**Interfaces:**
- Consumes: current `cadac_matmul` semantics
- Produces: same function name. If `a.shape == (3, 3)` and `b` is `(3, 3)` or `(3,)`/`(3,1)`, use unrolled ijk **in the same i, then k, then add order as the current `for i in range(nrow*ncol)` / `for k` loops**. Other shapes keep the generic loop. Must `np.array_equal` a reference nested-loop in the test, and must **not** `array_equal` a random 3×3 `numpy @` on the ROCKET6-sensitive matrices used in `test_cadac_inverse_matches_cpp_adjoint_over_det_not_numpy` (that test already proves inverse ≠ LAPACK).

- [ ] **Step 1: Write the failing test** (reference loop copied into the test file)

```python
def _ijk_matmul(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    squeeze = False
    if b.ndim == 1:
        b = b.reshape(-1, 1)
        squeeze = True
    nrow, nmid = a.shape
    ncol = b.shape[1]
    result = np.zeros((nrow, ncol), dtype=float)
    for i in range(nrow * ncol):
        r = i // ncol
        c = i % ncol
        acc = 0.0
        for k in range(nmid):
            acc += a[r, k] * b[k, c]
        result[r, c] = acc
    if squeeze:
        return result.reshape(nrow)
    return result


def test_cadac_matmul_3x3_matches_ijk_not_required_to_match_at():
    from cadac.math.frames import cadac_matmul

    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 3))
    b = rng.standard_normal((3, 3))
    got = cadac_matmul(a, b)
    np.testing.assert_array_equal(got, _ijk_matmul(a, b))
    v = rng.standard_normal(3)
    np.testing.assert_array_equal(cadac_matmul(a, v), _ijk_matmul(a, v))
```

This test will **pass** on current code. Add a second test that the 3×3 fast path exists by checking a monkeypatch is unnecessary — instead add:

```python
def test_cadac_matmul_generic_shape_still_ijk():
    from cadac.math.frames import cadac_matmul

    a = np.arange(8, dtype=float).reshape(2, 4)
    b = np.arange(12, dtype=float).reshape(4, 3)
    np.testing.assert_array_equal(cadac_matmul(a, b), _ijk_matmul(a, b))
```

The **new** requirement is documentation in `cadac_matmul` docstring: “3×3 unrolled, same ijk as C++ `Matrix::operator*`”. Implement unroll; tests stay `array_equal` to `_ijk_matmul`.

Because TDD needs an initial fail: add

```python
def test_cadac_matmul_3x3_unrolled_documented():
    import inspect
    from cadac.math.frames import cadac_matmul

    src = inspect.getsource(cadac_matmul)
    assert "a[0, 0] * b[0, 0]" in src or "nrow == 3" in src
```

- [ ] **Step 2: Run to verify fail**

Run: `cd Python && pytest tests/unit/test_frames.py::test_cadac_matmul_3x3_unrolled_documented -v`

Expected: FAIL (no unroll in source)

- [ ] **Step 3: Unroll 3×3 / 3×1**

```python
if a.shape == (3, 3) and b.shape == (3, 3):
    # nine accumulators, k innermost 0..2
    ...
```

Match `_ijk_matmul` bit for bit. Keep squeeze behavior for 1-D `b`.

- [ ] **Step 4: Run tests**

Run: `cd Python && pytest tests/unit/test_frames.py tests/unit/test_round6_kinematics.py tests/unit/test_rocket6_ins_ideal.py tests/unit/test_wgs84.py -q`

Expected: PASS

- [ ] **Step 5: UPDATES `0.169.6` + Commit if approved**

---

### Task 8: `look_up` last-index cache

**Files:**
- Modify: `Python/src/cadac/tables/lookup.py`
- Test: `Python/tests/unit/test_lookup_1d.py`, `test_lookup_2d.py`, `test_lookup_3d.py` plus new cache tests in `test_lookup_1d.py`

**Interfaces:**
- Consumes: existing `find_index`, `_interpolate_1d/2d/3d`
- Produces: `Datadeck` may keep `_loc_cache: dict[str, tuple]`. Cache is optional speed; **every** `look_up` return must equal today’s interpolation (including `EPS` dx, clamp at last index). Wrong cache → fail existing tests. Do not change `find_index` algorithm.

- [ ] **Step 1: Write tests that must keep passing plus a cache-hit equality test**

```python
def test_repeated_look_up_identical():
    d = deck()
    first = d.look_up("y", 5.0)
    for _ in range(32):
        assert d.look_up("y", 5.0) == first
    assert d.look_up("y", 20.0) == 10.0
    assert d.look_up("y", 5.0) == first


def test_look_up_two_tables_do_not_share_locs():
    t1 = Table(name="y", dim=1, x1=np.array([0.0, 10.0]), x2=None, x3=None, values=np.array([0.0, 10.0]))
    t2 = Table(name="z", dim=1, x1=np.array([0.0, 2.0]), x2=None, x3=None, values=np.array([1.0, 3.0]))
    d = Datadeck.from_tables([t1, t2])
    assert d.look_up("y", 5.0) == 5.0
    assert d.look_up("z", 1.0) == 2.0
    assert d.look_up("y", 5.0) == 5.0
```

Existing `test_midpoint`, `test_upper_constant`, `test_center` (2d), 3d tests must remain.

- [ ] **Step 2: Run new tests on current code**

Run: `cd Python && pytest tests/unit/test_lookup_1d.py tests/unit/test_lookup_2d.py tests/unit/test_lookup_3d.py -v`

Expected: PASS already (cache is speed). Add a test that fails until cache exists:

```python
def test_datadeck_has_loc_cache_attribute():
    d = deck()
    d.look_up("y", 5.0)
    assert hasattr(d, "_loc_cache")
    assert "y" in d._loc_cache
```

- [ ] **Step 3: Implement cache**

On `look_up`, after `find_index`, store `(loc1, loc2, loc3)` keyed by table name. Re-run `find_index` when the query `x1/x2/x3` is not the same as the cached query (cache key must include the x values, not only the name — otherwise a second x on the same table is wrong). Recommended: `_loc_cache[name] = (x1, x2, x3, loc1, loc2, loc3)` and reuse locs only when `x1, x2, x3` match. That still helps aero (`vmach` reused across 20 tables is a **different name** each time). Better cache: per-table last breakpoints search for that table’s x1 independently: `_idx_cache[(id(table.x1), float(x1))]` or last `(x1, loc)` on each `Table`. Attach `Table.last_x1`, `Table.last_loc1` updated in `find_index` via `Datadeck.find_index`. Simplest correct win: **memoize `find_index(max, value, breakpoints)`** with last `(id(breakpoints), max, value) -> loc` on the Datadeck.

```python
def find_index(self, max: int, value: float, breakpoints):
    key = (id(breakpoints), max, value)
    cached = self._idx_last
    if cached is not None and cached[0] == key:
        return cached[1]
    loc = ... existing binary search ...
    self._idx_last = (key, loc)
    return loc
```

Aero calls `find_index` on the **same** `table.x1` (mach) many times in one execute — last-key memo hits.

- [ ] **Step 4: Run tests**

Run: `cd Python && pytest tests/unit/test_lookup_1d.py tests/unit/test_lookup_2d.py tests/unit/test_lookup_3d.py tests/unit/test_agm6_aero.py tests/unit/test_plane6_aero.py tests/unit/test_sam6_aero.py -q`

Expected: PASS (skip aero tests that do not exist; glob `test_*aero*.py` if needed)

- [ ] **Step 5: UPDATES `0.169.7` + Commit if approved**

---

### Task 9: `ModuleBase` and EOM `define`

**Files:**
- Modify: `Python/src/cadac/kernel/module.py`
- Modify: `Python/src/cadac/eom/flat6.py` first; then `flat3.py`, `flat0.py`, `round3.py`, `round6.py`, `rotor.py` in the same task **only if** each class still defines the same Field names (reviewer checks one vehicle one-step test per family)
- Test: `Python/tests/unit/test_module_base.py` (create)

**Interfaces:**
- Consumes: `Field`, `Module` protocol
- Produces: `class ModuleBase:` with `name: str`, `fields: tuple[tuple, ...] | tuple[Field, ...]` — **must copy**: `store.define(Field(name, value, type, role, module, outputs))` so two vehicles do not share one `Field` instance. Default `initialize`/`terminate` = `pass`. `execute` raises `NotImplementedError` if not overridden. `DummyModule` may subclass it. EOM classes subclass `ModuleBase` and set `fields`. `Agm6Kinematics(Flat6Kinematics)` must keep working.

- [ ] **Step 1: Write the failing test**

```python
from cadac.kernel.module import ModuleBase
from cadac.kernel.state import Field, StateStore


class _Env(ModuleBase):
    name = "environment"
    fields = (
        Field("press", 0.0, "real", "out", "environment"),
    )

    def execute(self, vehicle, ctx):
        vehicle.store.set("press", 1.0)


def test_module_base_define_does_not_share_field_objects():
    a, b = type("V", (), {})(), type("V", (), {})()
    a.store, b.store = StateStore(), StateStore()
    mod = _Env()
    mod.define(a)
    mod.define(b)
    a.store.set("press", 3.0)
    assert b.store.get("press") == 0.0


def test_module_base_terminate_is_noop():
    v = type("V", (), {})()
    v.store = StateStore()
    _Env().define(v)
    _Env().terminate(v, None)
    assert v.store.get("press") == 0.0
```

- [ ] **Step 2: Run to verify fail**

Run: `cd Python && pytest tests/unit/test_module_base.py -v`

Expected: FAIL (ModuleBase missing)

- [ ] **Step 3: Implement ModuleBase; convert EOM**

Convert EOM `define` loops to `fields = (...)` + inherit. Delete empty `initialize`/`terminate` where they were only `pass`. Keep real `initialize` bodies. **New Field per define** (copy constructor).

- [ ] **Step 4: Run tests**

Run: `cd Python && pytest tests/unit/test_module_base.py tests/unit/test_flat6_kinematics.py tests/unit/test_flat6_euler.py tests/unit/test_round3_newton_step.py tests/unit/test_flat0_newton.py tests/unit/test_rotor_one_step.py tests/unit/test_agm6_kinematics.py tests/unit/test_executive.py -q`

Expected: PASS

- [ ] **Step 5: UPDATES `0.169.8` + Commit if approved**

---

### Task 10: Flat6 incidence helper, newton unpack, inverse comment

**Files:**
- Modify: `Python/src/cadac/eom/flat6.py`
- Modify: `Python/src/cadac/math/frames.py` if `incidence_angles` lives there (prefer `frames.py` so AGM6 can reuse)
- Test: `Python/tests/unit/test_flat6_kinematics.py`

**Interfaces:**
- Consumes: `cadac_sign`, `EPS`, `PI`, `DEG`
- Produces: `incidence_angles(vbab, dvba) -> tuple[float, float, float, float]` as `(alpha, beta, alpp, phip)` matching current C++ branches including `fabs(dum)>1`, `vbab2==0 and vbab3==0`, `fabs(vbab2)<EPS` then `vbab3>0 → 0`, `vbab3<0 → PI`, else `atan2(vbab2, vbab3)`. Do **not** keep the dead `phip = 0.0` before the `vbab3` tests. `Flat6Euler.execute` comment: Flat6 goldens use `np.linalg.inv`; Round6/ROCKET6 use `cadac_inverse` (UPDATES 0.168.12). `Flat6Newton` may call `_flight_path_angles` (already extracted).

- [ ] **Step 1: Write the failing tests**

```python
from cadac.constants import EPS, PI
from cadac.math.frames import incidence_angles


def test_incidence_phip_eps_negative_vbab3():
    alpha, beta, alpp, phip = incidence_angles(
        np.array([1.0, 0.5 * EPS, -1.0]), 2.0
    )
    # |vbab2| < EPS → PI when vbab3 < 0
    assert phip == PI


def test_incidence_phip_zero_when_both_zero():
    _, _, _, phip = incidence_angles(np.array([1.0, 0.0, 0.0]), 1.0)
    assert phip == 0.0
```

If AGM6 kinematics overrides incidence, do not change `Agm6Kinematics` formulas; only Flat6 default.

- [ ] **Step 2: Run to verify fail**

Run: `cd Python && pytest tests/unit/test_flat6_kinematics.py tests/unit/test_frames.py::test_incidence_phip_eps_negative_vbab3 -v`

Expected: FAIL (function missing)

- [ ] **Step 3: Extract helper; docstring on each Flat6 class** (Zipfel 6-DOF flat Earth + CADAC `flat6_*` module name). Wire `execute` to the helper. Comment Euler inverse.

- [ ] **Step 4: Run tests**

Run: `cd Python && pytest tests/unit/test_flat6_kinematics.py tests/unit/test_agm6_kinematics.py tests/unit/test_sraam6_kinematics.py tests/e2e/test_falcon6_gamma.py tests/e2e/test_agm6_freeflight.py -q`

Expected: PASS / skip-if-no-golden

- [ ] **Step 5: UPDATES `0.169.9` + Commit if approved**

---

### Task 11: Remaining EOM docstrings and inverse split

**Files:**
- Modify: `Python/src/cadac/eom/round6.py`, `flat3.py`, `round3.py`, `rotor.py`, `flat0.py` — module/class docstrings only plus the Round6 Euler comment pointing at `cadac_inverse`. No numeric edits unless a helper from Task 5–6 is an identical drop-in (optional `hypot3` / `skew` if not already done in Task 5).
- Test: `Python/tests/unit/test_eom_docs.py` (create)

**Interfaces:**
- Produces: every EOM class (`Round6Environment`, `Round6Kinematics`, `Round6Euler`, `Round6Newton`, `Flat3*`, `Round3*`, `Rotor*`, `Flat0*`, `Flat6*`) has a non-empty `__doc__` mentioning CADAC or Zipfel.

- [ ] **Step 1: Write the failing test**

```python
import cadac.eom.flat0 as flat0
import cadac.eom.flat3 as flat3
import cadac.eom.flat6 as flat6
import cadac.eom.round3 as round3
import cadac.eom.round6 as round6
import cadac.eom.rotor as rotor

_CLASSES = [
    flat6.Flat6Environment,
    flat6.Flat6Kinematics,
    flat6.Flat6Euler,
    flat6.Flat6Newton,
    round6.Round6Environment,
    round6.Round6Kinematics,
    round6.Round6Euler,
    round6.Round6Newton,
    round3.Round3Environment,
    round3.Round3Newton,
    flat3.Flat3Environment,
    flat3.Flat3Kinematics,
    flat3.Flat3Newton,
    flat0.Flat0Kinematics,
    flat0.Flat0Newton,
    rotor.RotorEnvironment,
    rotor.RotorTrajectory,
    rotor.RotorAttitude,
]


def test_eom_classes_have_docstrings():
    missing = [cls.__name__ for cls in _CLASSES if not cls.__doc__]
    assert missing == []
```

If a class name differs, use the actual names from those modules.

- [ ] **Step 2: Run to verify fail**

Run: `cd Python && pytest tests/unit/test_eom_docs.py -v`

Expected: FAIL (empty docs)

- [ ] **Step 3: Add docstrings + Round6 Euler inverse comment** (why not `np.linalg.inv`)

- [ ] **Step 4: Run tests**

Run: `cd Python && pytest tests/unit/test_eom_docs.py tests/unit/test_round6_euler.py tests/unit/test_round6_kinematics.py tests/unit/test_round6_newton_step.py -q`

Expected: PASS

- [ ] **Step 5: UPDATES `0.169.10` + Commit if approved**

---

### Task 12: ruff + type hints on kernel, math, tables, env, eom

**Files:**
- Modify: `Python/pyproject.toml` (`[project.optional-dependencies] dev` adds `ruff>=0.6`; `[tool.ruff]` `src = ["src"]`, `target-version = "py311"`)
- Modify: `Python/src/cadac/kernel/*.py`, `math/*.py`, `tables/*.py`, `env/*.py`, `eom/*.py` — annotations on public functions (`def integrate(...)`, `StateStore` methods, `cadac_matmul`, `look_up`, `run_loop`). Do not mass-annotate 133 vehicle files.
- Test: `Python/tests/unit/test_cadac_quality_metrics.py` (typed_defs must increase vs baseline)

**Interfaces:**
- Produces: `ruff check src/cadac/kernel src/cadac/math src/cadac/tables src/cadac/env src/cadac/eom` exit 0. Scanner `typed_defs` > `baseline["typed_defs"]`.

- [ ] **Step 1: Write the failing test**

```python
def test_typed_defs_exceed_baseline():
    root = Path(__file__).resolve().parents[1] / "src" / "cadac"
    data = scan_cadac(root)
    base = load_metrics(BASELINE)
    assert data["typed_defs"] > base["typed_defs"]
```

- [ ] **Step 2: Run to verify fail**

Run: `cd Python && pytest tests/unit/test_cadac_quality_metrics.py::test_typed_defs_exceed_baseline -v`

Expected: FAIL (equal to baseline)

- [ ] **Step 3: Annotate + ruff**

Add ruff to `pyproject.toml`. Annotate `state.py`, `integrate.py`, `executive.py`, `module.py`, `frames.py`, `lookup.py`. Fix ruff issues in those trees only (no drive-by vehicle reformat).

- [ ] **Step 4: Run tests + ruff**

Run: `cd Python && pytest tests/unit/test_cadac_quality_metrics.py tests/unit/test_state.py tests/unit/test_executive.py tests/unit/test_frames.py -q && ruff check src/cadac/kernel src/cadac/math src/cadac/tables src/cadac/env src/cadac/eom`

Expected: pytest PASS, ruff exit 0

- [ ] **Step 5: UPDATES `0.169.11` + Commit if approved**

---

### Task 13: Whole-branch unit gate

**Files:** none expected. Reviewer + implementer only run tests.

**Interfaces:**
- Consumes: Tasks 1–12
- Produces: green unit suite proof

- [ ] **Step 1: Whole-branch reviewer** (Grok non-fast) against spec: no `numpy @` on Round6 INS/kinematics multiply; no store key renames; ModuleBase copies Fields; `cadac_sign` not `np.sign`.

- [ ] **Step 2: Run full unit suite**

Run: `cd Python && pytest tests/unit -q --ignore-glob='*harvest*' -m 'not integration'`

Expected: PASS. Do not run `CADAC_HARVEST=1`. If a test is marked `integration`, it must skip.

- [ ] **Step 3: If anything fails, stop and fix in a focused follow-up; do not start e2e**

- [ ] **Step 4: UPDATES `0.169.12 - Unit suite green after quality overhaul` if a doc-only note is needed; skip code**

---

### Task 14: Full e2e matrix — all models and examples

**Files:** none unless a golden comparison fails (then fix the regression in the offending module; do not loosen CSV tols).

**Interfaces:**
- Consumes: `cadac.run_scenario` + cases under `Python/cases/`
- Produces: proof log of all 15 e2e files

- [ ] **Step 1: Run the full e2e folder**

Run: `cd Python && pytest tests/e2e -v --tb=short`

Expected: every test PASS, or `SKIPPED` only when the golden file is absent (`test_sam6_rf.py` and any existing skip). **No new skips.** Families that have goldens today must still pass:

- `test_hyper3_climb.py`
- `test_falcon5_turning.py`
- `test_falcon6_gamma.py`
- `test_hyper5_pronav.py`
- `test_hyper6_climb.py`
- `test_aim5_hori.py`
- `test_cruise5_input1.py`
- `test_magsix_attitude.py`
- `test_magsix_trajectory.py`
- `test_rocket6_insertion.py`
- `test_sam6_autopilot.py`
- `test_sraam6_1v1.py`
- `test_agm6_freeflight.py`
- `test_agm6_testcase.py`

- [ ] **Step 2: Record outcomes** in the Task 15 UPDATES entry (passed / skipped / failed). Failed → stop; do not re-survey as success.

- [ ] **Step 3: Commit if approved** only after green or documented skips: `test: e2e matrix green after quality overhaul`

---

### Task 15: Re-survey metrics and update the canvas

**Files:**
- Create: `Python/tools/cadac_quality/latest.json` (scan after Task 14)
- Modify: canvas `/home/valentin/.cursor/projects/home-valentin-Books-2025-Zeipfel-Modeling-and-simulation-of-aerospace-vehicles/canvases/cadac-python-quality-survey.canvas.tsx` — before/after stats, bar chart current vs baseline vs target, findings marked done/remaining
- Modify: `UPDATES.md`, `README.md` only if Architecture must mention `ModuleBase`, skip-coerce, or `cadac_quality`
- Test: `Python/tests/unit/test_cadac_quality_metrics.py`

**Interfaces:**
- Consumes: `scan_cadac`, `baseline.json`, e2e proof
- Produces: `latest.json`; canvas scores; UPDATES `0.169.13` (or next) with the table below filled from **actual** `latest.json` vs `baseline.json`

- [ ] **Step 1: Write the comparison test**

```python
def test_overhaul_improved_survey_counters():
    root = Path(__file__).resolve().parents[1] / "src" / "cadac"
    now = scan_cadac(root)
    base = load_metrics(BASELINE)
    assert now["skew_defs"] == 1
    assert now["cadac_sign_defs"] == 1
    assert now["typed_defs"] > base["typed_defs"]
    assert now["store_names"] < base["store_names"]
```

Do **not** assert `store_get` dropped (keys stay). Do not assert loc dropped.

- [ ] **Step 2: Run scanner + test**

Run: `cd Python && PYTHONPATH=src:tools python -m cadac_quality.metrics --root src/cadac --out tools/cadac_quality/latest.json && pytest tests/unit/test_cadac_quality_metrics.py -v`

Expected: PASS. If `--out` CLI was not in Task 1, add it here with a unit test.

- [ ] **Step 3: Re-score the four axes** using the same rubric as the original canvas (judgment 0–10) and the new counters. Targets were style 8, speed 8, vectorization 5, readability 8. New scores must be **justified** from `latest.json` (e.g. one skew def, fewer `names()`, more typed defs). Vectorization stays ≤ 5 unless a new batch API appeared (it must not).

- [ ] **Step 4: Update the canvas** with a two-series bar chart: baseline scores `[4, 3, 2, 5]` vs new scores; a table of baseline vs latest for every scanner key; caption `Source: cadac_quality.metrics · after Task 14 e2e`.

- [ ] **Step 5: UPDATES + README + Commit if approved**

UPDATES must include the numeric before/after table and the e2e result line (`15 collected, N passed, K skipped, 0 failed`).

---

## Self-review

**Spec coverage:** store coerce / contains / get_optional → T2. names membership → T4. executive bind → T3. cadac_matmul unroll → T7. lookup cache → T8. skew/sign → T5. fused quat/hypot → T6. ModuleBase EOM → T9. incidence + inverse comment → T10–11. ruff/types → T12. no time-batch / no MC batch → global constraints. e2e all models → T14. re-survey canvas → T15. metrics freeze → T1.

**Placeholders:** none. Class names in T11 must be checked against the files at execute time.

**Types:** `scan_cadac` → T1; `__contains__` / `get_optional` → T2; `cadac_sign` / `skew` → T5; `hypot3` / `quat_to_dcm` → T6; `incidence_angles` → T10; `ModuleBase` → T9.

**Out of scope (do not sneak in):** fast Monte Carlo, numba, vehicle-wide ModuleBase, HYPER6 RADAR0.
