# Updates

## 0.26.0 - Round3 newton step
- `Round3Newton.execute` ports HYPER3 `Round3::newton`: `abii_new = TIG @ ((TGV @ FSPV) + grav_vec)` with `grav_vec=[0,0,grav]`; stored-slope trapezoid `integrate` of vbii then sbii (previous `abii` is the slope; then `abii=abii_new`); TGE/TGI/VBEG/SBEG; `polar_from_cart` speed/heading/FPA; TIG=TGI.T, TGV=TVG.T. `FSPV` from store (forces; not defined here), `grav` from store (environment). `cadtei(ctx.sim_time)`. No cruise aero/prop/forces.
- Tests: `Python/tests/unit/test_round3_newton_step.py` (Task 25 ICs, zero FSPV, dt=0.01; alt finite; sbii changes; vbii/sbii replica of integrate order).

## 0.25.0 - Round3 newton initialize
- Added `cadac.eom.round3.Round3Newton` (`name="newton"`) in the same `round3.py` as `Round3Environment`. `define` registers C++ `def_newton` fields (lonx, latx, alt, dvbe, psivgx, thtvgx, sbii, vbii, abii, sbeg, vbeg, tgv, tig, tge, weii, …). `initialize` ports `Round3::init_newton` (lon/lat/alt → SBII via `cadtge`/`cadtei`, heading/FPA/speed → VBEG/VBII, WEII skew-sym with `WEII3`). Angles in store are degrees. No `execute` newton step (Task 26). Does not define environment fields; `alt`/`dvbe` belong to newton.
- Tests: `Python/tests/unit/test_round3_newton_init.py` (HYPER3 ICs lonx=-80.55, latx=28.43, alt=3000, psivgx=90, thtvgx=0, dvbe=250; `cadsph(sbii)` alt; WEII; east VBEG via `polar_from_cart`).

## 0.24.0 - Round3 environment module
- Added `cadac.eom.round3.Round3Environment` (`name="environment"`): `define` registers time/event_time/int_step_new/out_step_fact/grav/rho/pdynmc/mach/vsound/press; `initialize` sets time=sim_time, int_step_new=int_step; `execute` reads newton `alt`/`dvbe`, writes ISO62 outputs and `grav=gravity(alt)`, copies `ctx.sim_time` to `time`, sets `ctx.int_step` from `int_step_new` and `ctx.out_fact` from `out_step_fact`. Uses `iso62`+`gravity`, not US76.
- `run_loop` adopts `ctx.int_step` after modules for `event_time +=` and `sim_time +=` (shared C++ local; while condition uses the same variable).
- Tests: `Python/tests/unit/test_round3_environment.py` (alt=3000, dvbe=250 vs iso62/gravity); `test_run_loop_adopts_ctx_int_step` in `test_executive.py`.

## 0.23.0 - CADAC-style plot CSV writer
- Added `cadac.io.plot.write_plot_csv(path, title, columns, rows)`: line 1 title, line 2 `0  0 N`, line 3 `col,col,`, data rows comma-separated with trailing comma (CADAC `plot1.csv` shape). Unix LF. Not wired into the executive (Task 31).
- Tests: `Python/tests/unit/test_plot_csv.py` (round-trip two rows; parse back `time,alt`).

## 0.22.0 - Translate CADAC IF/ENDIF events
- `translate_scenario_asc` maps sequential `IF var op value` … `ENDIF` to ordered vehicle `events`: `when: {var: {op: value}}` plus `set` of assignments. Ops `<` `=` `>`. HYPER3 `input_climb.asc` → two events (`time>10`: mprop=2, qhold=50000, tq, alphax; `time>50`: alphax). Output still loads via `load_scenario` into `EventSpec`. No `asc_scenario.py`. Unix LF.
- Tests: `Python/tests/translate/test_asc_events.py`.

## 0.21.0 - Translate input.asc (no events)
- Added `cadac.io.translate.translate_scenario_asc(src, dst_dir)`: CADAC `input.asc` → `{stem}.jsonc` in `dst_dir`. `y_*`/`n_*` options drop the prefix; MODULES name+phases; TIMING floats; VEHICLES type/name/params; `AERO_DECK`/`PROP_DECK` → `aero_deck`/`prop_deck` `.jsonc` path strings (decks not rewritten). Skips `IF`/`ENDIF` bodies (Task 22). Output loads with `load_scenario`. Unix LF.
- Tests: `Python/tests/translate/test_asc_scenario.py` (HYPER3 `input_climb.asc`: `CRUISE3`, `lonx==-80.55`, `end_time==90`, `environment`+`init`; IF-body not in params; `n_*` → false).

## 0.20.0 - Scenario JSONC schema
- Added `cadac.io.scenario`: `load_scenario(path) -> RunConfig` via `cadac.io.jsonc.loads`. Dataclasses `RunConfig`, `VehicleSpec`, `ModuleSpec`; events reuse `EventSpec`.
- Known `options` keys `scrn`/`events`/`plot`/`doc`/`csv`/`tabout`/`merge`/`comscrn`/`traj`; omitted flags False; unknown keys `ValueError`. Extra `params` keys kept. Relative `aero_deck`/`prop_deck` resolved against scenario parent. `CRUISE3` is a string.
- Tests: `Python/tests/unit/test_scenario.py` plus fixture `Python/tests/fixtures/minimal_cruise3.jsonc` (one CRUISE3, one time event).

## 0.19.0 - Combus packets
- Added `cadac.kernel.combus`: `@dataclass Packet(name, type, status: int, vars: dict)` and `packet_from_store(store, names)` copying named fields. Status 1 alive, 0 dead, -1 hit; builder defaults status=1.
- `run_loop` initializes `combus` as a list of Packets (health from `vehicle.health` then `vehicle.status`, default 1). Skip execute when `combus[slot].status != 1`. After modules, if vehicle has `com_names`, save health, publish `packet_from_store` at `vehicle_slot`, restore health. `ctx.combus` is that list.
- Tests: `Python/tests/unit/test_combus.py` (two named vars; publish after modules; save/restore status; slot index).

## 0.18.0 - Executive loop with dummy module
- Added `cadac.kernel.module`: `Module` protocol (`name`, `define`, `initialize`, `execute`, `terminate`) and `DummyModule` that sets `store.time` to `ctx.sim_time`.
- Added `cadac.kernel.executive`: `SimContext(sim_time, int_step, event_time, out_fact, combus, vehicle_slot)` and `run_loop`. CADAC `while sim_time <= end_time+int_step`; per vehicle evaluate events then execute `module_order` if health==1 (default 1; `health`/`status` if present); `event_time += int_step`; `sim_time += int_step`. Returns sim_time at start of each iteration. `combus` is None.
- Tests: `Python/tests/unit/test_executive.py` (`end_time=0.2`, `int_step=0.1` → `[0.0, 0.1, 0.2, 0.3]`; dummy time; health/status skip; events before modules; module order).

## 0.17.0 - Sequential event engine
- Added `cadac.kernel.events`: `EventSpec(when, set)` and `EventEngine.evaluate(store)` (CADAC `event_epoch`). One event armed; ops `<` `=` `>`; int watch as int, float raw compare (no epsilon). Nested `when={"time": {">": 10}}` and `when={"var": ..., "op": "=", "value": ...}`. On fire: `store.set` then advance; after last event, False forever.
- Tests: `Python/tests/unit/test_events.py` (time then set; var/op/=; less-than; sequential arming).

## 0.16.1 - Atomic define and unknown Field type
- `StateStore.define` coerces then inserts so a failed vec/mat shape does not register the name.
- `_coerce` raises `ValueError` on unknown `type` (not real/int/vec/mat).
- Tests: `test_define_shape_error_does_not_register_name`, `test_unknown_type_raises_valueerror`.

## 0.16.0 - Named state store
- Added `cadac.kernel.state`: `Field(name, value, type, role, module, outputs=())` and `StateStore` with `.define`/`.get`/`.set`/`.names()`. Lookup by name; no `Variable[i]`. Duplicate `define` raises `ValueError`; unknown `get`/`set` raises `KeyError`. Int stored as int; vec `(3,)`, mat `(3,3)`. No `units`.
- Tests: `Python/tests/unit/test_state.py` (define/get/set, duplicate, unknown, int vs real, vec/mat shape).

## 0.15.0 - cadtei, cadtge, cadsph
- Added `cadac.math.earth`: `cadtei(sim_time)` earth-wrt-inertial T.M. (`xi=WEII3*t`); `cadtge(lon_rad, lat_rad)` geographic-wrt-earth; `cadsph(sbie)` lon/lat/alt with CADAC longitude quadrant ifs (PI not atan2). Returns numpy (3,3) / (lon, lat, alt). Did not change frames.py.
- Tests: `Python/tests/unit/test_earth.py` (cadtei(0) identity; cadsph([REARTH,0,0]) zeros; cadtge(0,0) C++ assignments).

## 0.14.0 - polar_from_cart and mat2tr
- Added `cadac.math.frames`: `polar_from_cart(v)` CADAC `pol_from_cart` (d, atan2(v2,v1), elev); `mat2tr(psivg, thtvg)` element-by-element. Returns numpy (3,)/(3,3). No mat3tr/cadtei/cadtge/cadsph.
- Tests: `Python/tests/unit/test_frames.py` (east [0,250,0]; mat2tr(0,0) identity).

## 0.13.0 - Newtonian gravity
- Added `cadac.env.gravity`: `gravity(alt_m) = G*EARTH_MASS/(REARTH+alt_m)**2` with constants from `cadac.constants`.
- Tests: `Python/tests/unit/test_gravity.py` (alt=3000).

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
