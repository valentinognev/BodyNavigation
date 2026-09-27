# CADAC Python vs C++ e2e comparison failures

Date: 2026-09-07  
Source: live `pytest` on harvested Linux g++ goldens (parity audit).  
Command (from `Python/`):

```bash
pytest \
  tests/e2e/test_agm6_freeflight.py::test_hbe_matches_golden_at_t0 \
  tests/e2e/test_agm6_freeflight.py::test_hbe_and_vmach_match_golden_at_shared_times \
  tests/e2e/test_agm6_testcase.py::test_hbe_matches_golden_at_t0 \
  tests/e2e/test_agm6_testcase.py::test_hbe_and_vmach_match_golden_at_shared_times \
  tests/e2e/test_falcon5_turning.py::test_alt_matches_golden_at_t0 \
  tests/e2e/test_falcon5_turning.py::test_shared_plot_columns_match_golden \
  tests/e2e/test_hyper5_pronav.py::test_alt_matches_golden_at_t0 \
  tests/e2e/test_hyper5_pronav.py::test_shared_plot_columns_match_golden \
  tests/e2e/test_magsix_attitude.py::test_hbe_matches_golden_at_t0 \
  tests/e2e/test_magsix_attitude.py::test_all_shared_plot_columns_match_golden_at_shared_times \
  tests/e2e/test_rocket6_insertion.py::test_alt_matches_golden_at_t0 \
  tests/e2e/test_rocket6_insertion.py::test_alt_and_vmach_match_golden_at_shared_times \
  tests/e2e/test_sam6_autopilot.py::test_time_matches_golden_at_t0 \
  tests/e2e/test_sam6_autopilot.py::test_shared_missile_plot_columns_match_golden \
  -v --tb=short
```

Result: **9 failed, 5 passed** in 146 s. CSV tolerances are unchanged: `rtol=1e-5`, `atol=max(1e-6, 5e-6*|golden|)`. Sentinel `time=-1` skipped. Python plot is vehicle slot 0 only.

Inventory: `Python/tools/cadac_cpp/inventory.json` records these seven JSONC families as `kind=e2e` `status=diverged`. Spec: leave the tests red; do not loosen tolerances; do not port extra modes to make a different ASC pass.

Not in this list: SAM6 RF (`test_sam6_rf.py`) still **skips** (no JSONC / no golden). That is not a comparison failure.

---

## Summary

| Family | Test file | Live test | Outcome | First failure |
|---|---|---|---|---|
| AGM6 free flight | `test_agm6_freeflight.py` | `test_hbe_matches_golden_at_t0` | FAIL | `ValueError: math domain error` in `atmosphere76` |
| AGM6 free flight | `test_agm6_freeflight.py` | `test_hbe_and_vmach_match_golden_at_shared_times` | FAIL | same crash (never reaches CSV compare) |
| AGM6 test case | `test_agm6_testcase.py` | `test_hbe_matches_golden_at_t0` | FAIL | `ZeroDivisionError` in `aerodynamics_der` |
| AGM6 test case | `test_agm6_testcase.py` | `test_hbe_and_vmach_match_golden_at_shared_times` | FAIL | same crash |
| FALCON5 | `test_falcon5_turning.py` | `test_alt_matches_golden_at_t0` | PASS | — |
| FALCON5 | `test_falcon5_turning.py` | `test_shared_plot_columns_match_golden` | FAIL | `FSPV3` at `t=0.0` |
| HYPER5 | `test_hyper5_pronav.py` | `test_alt_matches_golden_at_t0` | PASS | — |
| HYPER5 | `test_hyper5_pronav.py` | `test_shared_plot_columns_match_golden` | FAIL | `psivgx` at `t=0.0` |
| MAGSIX attitude | `test_magsix_attitude.py` | `test_hbe_matches_golden_at_t0` | PASS | — |
| MAGSIX attitude | `test_magsix_attitude.py` | `test_all_shared_plot_columns_match_golden_at_shared_times` | FAIL | `KeyError: ('sim_time', 0.3501)` |
| ROCKET6 | `test_rocket6_insertion.py` | `test_alt_matches_golden_at_t0` | PASS | — |
| ROCKET6 | `test_rocket6_insertion.py` | `test_alt_and_vmach_match_golden_at_shared_times` | FAIL | `vmach` at `t=0.1` |
| SAM6 autopilot | `test_sam6_autopilot.py` | `test_time_matches_golden_at_t0` | PASS | — |
| SAM6 autopilot | `test_sam6_autopilot.py` | `test_shared_missile_plot_columns_match_golden` | FAIL | `thtvlcx` at `t=0.0` |

JSONC families whose live golden comparisons **all passed** (not in the failure set): HYPER3 climb, FALCON6 gamma, HYPER6 climb, AIM5 hori, CRUISE5 input_1, MAGSIX trajectory, SRAAM6 1v1.

---

## 1. AGM6 free flight — crash (`atmosphere76`)

- JSONC: `Python/cases/agm6/input_freeflight.jsonc`
- Golden: `Python/tests/e2e/goldens/agm6/plot.csv` (harvested; C++ run completed)
- Tests: `test_hbe_matches_golden_at_t0`, `test_hbe_and_vmach_match_golden_at_shared_times`

```
src/cadac/vehicles/flat6/agm6/environment.py:88  atmosphere76(hbe)
src/cadac/env/us76.py:46
ValueError: math domain error
  delta = ptab[i] * math.pow((tbase / tlocal), (gmr / tgrad))
```

Python never produces a plot row. Typical cause: `hbe` left the US76 table (negative `tbase/tlocal` or a bad layer index). Free-flight JSONC is `mair=0` (US76). C++ harvested a full CSV, so this is a Python environment/altitude bug on the representative case, not a missing golden.

---

## 2. AGM6 test case — crash (`aerodynamics_der`)

- JSONC: `Python/cases/agm6/input_testcase.jsonc`
- Golden: `Python/tests/e2e/goldens/agm6/test_case_plot.csv`
- Tests: `test_hbe_matches_golden_at_t0`, `test_hbe_and_vmach_match_golden_at_shared_times`

```
src/cadac/vehicles/flat6/agm6/aero.py:200  aerodynamics_der
src/cadac/vehicles/flat6/agm6/aero.py:261
ZeroDivisionError: division by zero
  a12 = dma / dna
```

Again no plot. C++ harvested CSV. `dna==0` in the aero derivative path (test case uses weather/`mair=212`). Unit smoke only ran `end_time=0.05`; live e2e runs the full case.

---

## 3. FALCON5 turning-to-IP — `FSPV3` at t=0

- JSONC: `Python/cases/falcon5/input_turning_to_IP.jsonc`
- Golden: `Python/tests/e2e/goldens/falcon5/plot.csv`
- Pass: `test_alt_matches_golden_at_t0`
- Fail: `test_shared_plot_columns_match_golden` (first shared column that exceeds tol)

| | Python | C++ golden | abs Δ | rel Δ | rtol | atol |
|---|---|---|---|---|---|---|
| `FSPV3` at t=0 | −0.724651 | −0.724621 | 3.028e-5 | 4.178e-5 | 1e-5 | 3.623e-6 |

Slightly over `rtol`. First mismatch is already at t=0 on specific force, so later columns were not evaluated. Golden was converted from FALCON5 `plot1.asc` after C++ `exit(1)` (intercept); layout matches CADAC CSV.

---

## 4. HYPER5 Demo 4.7 — `psivgx` at t=0

- JSONC: `Python/cases/hyper5/input.jsonc`
- Golden: `Python/tests/e2e/goldens/hyper5/plot.csv`
- Pass: `test_alt_matches_golden_at_t0`
- Fail: `test_shared_plot_columns_match_golden`

| | Python | C++ golden | abs Δ | rel Δ | rtol | atol |
|---|---|---|---|---|---|---|
| `psivgx` at t=0 | 0.000292 | 0.000294 | 1.874e-6 | 6.365e-3 | 1e-5 | 1e-6 |

Near-zero heading: `atol` floors at 1e-6, so a 1.87e-6 heading difference fails. `alt` at t=0 is inside tolerance. Later times not evaluated.

---

## 5. MAGSIX attitude — plot time grid (`sim_time`)

- JSONC: `Python/cases/magsix/input.jsonc` (`plot_step`: 0.005)
- Golden: `Python/tests/e2e/goldens/magsix/plot.csv`
- Pass: `test_hbe_matches_golden_at_t0` (aligned at `sim_time=0`)
- Fail: `test_all_shared_plot_columns_match_golden_at_shared_times`

```
KeyError: ('sim_time', 0.3501)
```

C++ golden `sim_time` starts `0, 0.005, 0.01, …`. Alignment uses `abs(python.sim_time - golden.sim_time) < 1e-9`. Golden later contains `0.3501` (likely CADAC float, 70×0.005 ≠ 0.35 exactly in the CSV). Python has no row at that stamp, so comparison aborts before column checks. MAGSIX **trajectory** e2e against `goldens/magsix/trajectory/plot.csv` passed.

---

## 6. ROCKET6 insertion — `vmach` at t=0.1

- JSONC: `Python/cases/rocket6/input.jsonc`
- Golden: `Python/tests/e2e/goldens/rocket6/plot.csv`
- Pass: `test_alt_matches_golden_at_t0`
- Fail: `test_alt_and_vmach_match_golden_at_shared_times` (compares only `alt` and `vmach`)

| | Python | C++ golden | abs Δ | rel Δ | rtol | atol |
|---|---|---|---|---|---|---|
| `vmach` at t=0.1 | 0.008492 | 0.008491 | 1.327e-6 | 1.563e-4 | 1e-5 | 1e-6 |

Barely over both `atol` (1e-6 floor) and `rtol`. Harvest forced `MONTE 0` on `MONTE 1 <seed>` so this is not live `rand()`.

---

## 7. SAM6 autopilot — `thtvlcx` at t=0

- JSONC: `Python/cases/sam6/input_SAM_autopilot.jsonc`
- Golden: `Python/tests/e2e/goldens/sam6/plot.csv`
- Pass: `test_time_matches_golden_at_t0`
- Fail: `test_shared_missile_plot_columns_match_golden`

| | Python | C++ golden | abs Δ | rel Δ | rtol | atol |
|---|---|---|---|---|---|---|
| `thtvlcx` at t=0 | 80.0 | 79.9238 | 0.0762 | 9.53e-4 | 1e-5 | 3.996e-4 |

Largest numeric miss. Python looks like a round 80° command; C++ golden is 79.9238°. Same-row C++ `thtvlx` is 79.9969 (not compared first). Later columns not evaluated. Harvest MONTE-off; test file still documents that the golden must be zero-MC (this harvest used OPTIONS `y_csv` + `nmonte==0`).

---

## How to reproduce one family

```bash
cd Python
pytest tests/e2e/test_falcon5_turning.py::test_shared_plot_columns_match_golden -v --tb=short
```

Goldens live under `Python/tests/e2e/goldens/`. Rebuild C++ plots with `CADAC_HARVEST=1 pytest tests/unit/test_cadac_cpp_harvest.py::test_harvest_all_writes_or_skips`.
