# Updates

## 0.12.0 - ISO 62 atmosphere (Round3)
- Added `cadac.env.iso62`: tropopause split at 11000 m; `iso62(alt_m, dvbe)` returns k, press, rho, vsound, mach, pdynmc. Uses R from constants; no US76, no gravity.
- Tests: `Python/tests/unit/test_iso62.py` (alt=3000 tropopause; alt=20000 stratosphere).

## 0.11.0 - US 1976 standard atmosphere
- Added `cadac.env.us76.atmosphere76`: US76 tables + geopotential with internal `rearth=6369.0` km (not `REARTH`). Returns `(rho, press, tempk)`.
- Tests: `Python/tests/unit/test_us76.py` (sea level; 11 km geometric).

## 0.10.0 - CADAC trapezoidal integrate
- Added `cadac.kernel.integrate`: stored-slope trapezoid `y + (dydx_new + dydx) * dt / 2` for scalars and ndarrays. No scipy, no RK4.
- Tests: `Python/tests/unit/test_integrate.py` (scalar 11.0; vec [1, 1]).

## 0.9.0 - Translate ASC decks to JSONC
- Added `cadac.io.translate.deck_asc_to_jsonc`: parse ASC, json-dump title+tables with Python lists; `load_deck` reload matches ASC `x1`/`x2`/`values`.
- Tests: `Python/tests/translate/test_deck_roundtrip.py` (HYPER3 `ghame3_aero_deck.asc` 1D, `ghame3_prop_deck.asc` 2D).

## 0.8.0 - CADAC 2D table look_up
- Extended `Datadeck.look_up(name, x1, x2)` with bilinear interpolate as HYPER3 2D `interpolate`; constant upper per axis, slope lower; `dx>EPS` else dumx=0. 1-arg path unchanged.
- Tests: `Python/tests/unit/test_lookup_2d.py` (center, upper x1 constant).

## 0.7.0 - CADAC 1D table look_up
- Added `Datadeck.find_index` (C++ binary search) and `Datadeck.look_up(name, x1)` with linear interpolate, constant upper extrapolation, slope below min; `dx>EPS` else dumx=0.
- Tests: `Python/tests/unit/test_lookup_1d.py` (midpoint, upper constant, lower slope, on-node).

## 0.6.0 - Parse CADAC 2DIM ASC decks
- Extended `cadac.io.asc_deck.parse_asc_deck` with 2DIM packing matching HYPER3 `Cruise::read_tables` (x1 rows, x2 columns, dangling x2 after the matrix).
- Test uses real HYPER3 `ghame3_prop_deck.asc` (`ca_vs_alpha_mach` 9x13).

## 0.5.0 - Parse CADAC 1DIM ASC decks
- Added `cadac.io.asc_deck.parse_asc_deck`: TITLE plus 1DIM `NX1 n` / `x y` rows to `Table`; 2DIM/3DIM raise `NotImplementedError`.
- Test uses real HYPER3 `ghame3_aero_deck.asc`.

## 0.4.0 - Table dataclass and JSONC deck load
- Added `cadac.tables.lookup.Table` / `Datadeck.from_tables` (load only; no `look_up`).
- Added `cadac.io.deck.load_deck`: JSONC `{title, tables}` via `jsonc.loads`; 1D/2D/3D shape check raises `ValueError`.

## 0.3.0 - JSONC loader
- Added `cadac.io.jsonc` (`loads`/`load`): strip `//` and non-nested `/* */`, then stdlib `json.loads`. No trailing commas. String-aware scan.

## 0.2.0 - CADAC Python design and atomic TDD plans
- Approved library design: JSONC I/O, named state, CADAC numerics, first slice HYPER3/FALCON5/FALCON6.
- Spec: `docs/superpowers/specs/2026-09-04-cadac-python-design.md`.
- Plans: kernel+HYPER3 (32 TDD tasks), FALCON5 (17), FALCON6 (17). Grok implementer/reviewer per task.
- EOM taxonomy: Round3, Flat3, Flat6, Round6, Ground0, Flat0, Rotor (MAGSIX).

## 0.1.0 - Project docs and CADAC inventory
- Added root `README.md` / `UPDATES.md`.
- Inventoried twelve CADAC++ simulations under `CADAC_Simulations/` (3/5/6-DOF, round/flat Earth, copied executive + Variable/Matrix/Datadeck).
- `Python/` exists and is empty; unified library not started pending design approval.
