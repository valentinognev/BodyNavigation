# CADAC Python quality overhaul — design

Date: 2026-09-08
Status: executed 2026-09-08; merged to `main` (`aa90bd2`).

## Goal

Take `Python/src/cadac` from the surveyed state (C++ transcription with a string store inner loop) to the best parity-safe quality: faster per-step Python, shared 3-vector math, cached table lookup, typed kernel, readable EOM — without changing JSONC, plot CSV keys, or harvested C++ goldens.

## Oracle

Harvested CADAC plot CSVs under `Python/tests/e2e/goldens/` remain the correctness oracle. Unit `rtol=1e-12`, `atol=1e-14`. CSV e2e `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)` unless a test already documents a different floor.

## Non-goals

- Vectorizing across `sim_time` (events, flags, `rand()` order are causal).
- Batching Monte Carlo as `(N, …)` arrays (glibc `rand()` order). `nmonte` stays serial.
- Numba/JAX or new runtime dependencies beyond `numpy` + `pytest` + `ruff` (dev).
- Renaming CADAC store keys (`hbe`, `alppx`, `q0`, …).
- Replacing `cadac_matmul` / `cadac_inverse` with `numpy @` / `linalg.inv` on Round6 / ROCKET6 / HYPER6 INS paths.
- Unifying Flat6 `np.linalg.inv` with `cadac_inverse` (document the split; do not silently change Flat6).
- Editing any file under `CADAC_Simulations/`.
- Porting HYPER6 `RADAR0` / `SAT3` or SAM6 RF golden if still absent.
- Rewriting all 133 vehicle modules onto `ModuleBase` in this overhaul (EOM only).

## Target architecture

```
kernel/state.py      dict store; skip-coerce; O(1) __contains__; get_optional
kernel/executive.py  bind module.execute once per vehicle; reuse SimContext
kernel/module.py     Module protocol + ModuleBase (no-op init/term; copy Field on define)
math/frames.py       cadac_sign, skew, hypot3, quat_to_dcm, matvec3; 3×3 ijk unroll
tables/lookup.py     last-index cache; CADAC find_index / EPS unchanged
eom/*.py            shared helpers; Zipfel/C++ docstrings; unpack/pack; keep keys
tools/cadac_quality  scan the same metrics as the survey canvas
```

Public API stays `cadac.run_scenario(path)`. Store keys, combus names, and plot columns do not change.

## Parity-safe speed

1. **StateStore** — `set` skips `int`/`float`/`np.asarray` when the value already has the stored type and shape. Still coerce lists and wrong dtypes. `names()` still returns `list` for plot/combus. Membership is `"x" in store`, not `"x" in store.names()`.
2. **Executive** — `{module.name: module}` is built once. `SimContext` is reused and mutated (`sim_time`, `int_step`, `event_time`, `vehicle_slot`). `ctx.int_step` assignment still resizes the loop.
3. **3-vector kernels** — explicit `skew`, `hypot3`, `quat_to_dcm`, `matvec3`. Numpy on length-3 is not the win.
4. **`cadac_matmul`** — unrolled ijk for `(3,3)×(3,3)` and `(3,3)×(3,)`. Same addition order as today’s Python loops. Generic loop remains for other shapes (GPS 8×8).
5. **`look_up`** — cache last `find_index` loc per table name. Interpolation formulas unchanged.

## Style and readability

- One `cadac.math.frames.cadac_sign` (`< 0 → -1` else `+1`) and `skew`. Delete local copies.
- `ModuleBase` for `eom/` only. `define` must construct a **new** `Field(...)` per call (class-level Field tuples would mutate defaults across vehicles).
- EOM class docstring: Zipfel topic + CADAC C++ function name. Inline comments only where Python diverges (EPS pole, ijk, `nmonte==0` gauss).
- `Flat6Kinematics` incidence (`alpha`, `beta`, `alpp`, `phip`) in `incidence_angles`. Keep C++ EPS / `vbab2==0` control flow; drop the dead `phip=0` before the `vbab3` tests.
- Flat6 Euler: comment why `np.linalg.inv` vs Round6 `cadac_inverse`.
- `ruff` + annotations on `kernel/`, `math/`, `tables/`, `env/`, `eom/`. Vehicles later, out of this plan except membership/`skew` replacements.

## Metrics (survey canvas)

Scanner `Python/tools/cadac_quality/metrics.py` (pytest `pythonpath` already includes `tools`) emits JSON:

| key | baseline 8 Sep 2026 |
|---|---|
| files | 168 |
| loc | 24518 |
| store_get | 2687 |
| store_set | 2120 |
| store_names | 104 |
| look_up | 228 (plus 1 def) |
| np_zeros | 112 |
| np_array | 252 |
| typed_defs | 45 |
| untyped_defs | 957 |
| empty_terminate | 143 |
| empty_initialize | 106 |
| skew_defs | 21 |
| cadac_sign_defs | 9 |

Judgment scores (canvas): style 4, speed 3, vectorization 2, readability 5. Parity-safe targets: 8, 8, 5, 8.

After the e2e gate, re-scan, write `latest.json`, update the canvas with before/after, bump `UPDATES.md`.

## E2E matrix (final gate)

`cd Python && pytest tests/e2e -q` — all 15 files. Skip only when a golden file is absent (`test_sam6_rf.py` and any other documented skip). Families with goldens must pass:

| File | Case |
|---|---|
| `test_hyper3_climb.py` | HYPER3 climb |
| `test_falcon5_turning.py` | FALCON5 turning to IP |
| `test_falcon6_gamma.py` | FALCON6 gamma |
| `test_hyper5_pronav.py` | HYPER5 Demo 4.7 |
| `test_hyper6_climb.py` | HYPER6 climb |
| `test_aim5_hori.py` | AIM5 hori |
| `test_cruise5_input1.py` | CRUISE5 input_1 |
| `test_magsix_attitude.py` | MAGSIX attitude |
| `test_magsix_trajectory.py` | MAGSIX trajectory |
| `test_rocket6_insertion.py` | ROCKET6 insertion |
| `test_sam6_autopilot.py` | SAM6 autopilot |
| `test_sam6_rf.py` | SAM6 RF (skip if no golden) |
| `test_sraam6_1v1.py` | SRAAM6 1v1 |
| `test_agm6_freeflight.py` | AGM6 free flight |
| `test_agm6_testcase.py` | AGM6 test case |

After kernel tasks, a smoke subset is enough: `test_hyper3_climb.py`, `test_falcon6_gamma.py`, `test_agm6_freeflight.py`.

## Execution

Grok non-fast (`cursor-grok-4.6-high`) implementer per task, Grok reviewer per task, whole-branch reviewer before the e2e gate. TDD. No CADAC `.cpp` edits. No git commit unless the user asked for commits on that run.
