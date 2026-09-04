# Updates

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
