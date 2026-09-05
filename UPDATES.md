# Updates

## 0.42.0 - FALCON5 load-factor control
- Added `cadtbv(phi, alpha)` to `cadac.math.frames`: C++ zeros then `assign_loc`; `(1,0)` stays 0. Used by load-factor control.
- `Plane5Control.define` adds load-factor fields from C++ `def_control` used by `control_load` (anposlimx, anneglimx, gacp, ta, alphax, alpposlimx, alpneglimx, xi, xid, alp, alpd, anx, qq, tip, ancomx). Does not register mcontrol/TBV/alcomx/altitude/heading.
- `control_load(vehicle, ancomx, int_step)` ports FALCON5 `Plane::control_load`: TBV=`cadtbv(phimvx*RAD, alphax*RAD)`, FSPB=TBV@FSPV, clip ancomx to [anneglimx, anposlimx], anx=-FSPB[2]/grav, P-I (`gr` starts 0; if ta<=0 then xi=0 and qq=0), incidence lag, clip returned alpx. Writes xi,xid,alp,alpd,anx,qq,tip; returns alpx; does not write alphax. `execute` remains the bank wrap.
- Tests: `Python/tests/unit/test_plane5_control_load.py` (cadtbv (1,0)==0; turning_to_IP gacp=10, ta=0.8, anposlimx=3, anneglimx=-1, alpposlimx=15, alpneglimx=-10; one-step vs C++ replica; ta<=0; ancomx/alpx clips; execute still bank-only). Plant FSPV/grav/mass/dvbe/pdynmc/thrust/area/cla registered by tests.

## 0.41.0 - FALCON5 bank-angle control
- Added `cadac.vehicles.plane5.control.Plane5Control` (`name="control"`). `define` registers bank fields from C++ `def_control` used by `control_bank`: phimvx (out, scrn/plot), phicx (data, scrn/plot), phix (state, plot), phixd (state), philimx, tphi. Does not register the rest of `def_control`. `control_bank(vehicle, phicx, int_step)` ports FALCON5 `Plane::control_bank`: clip phicx to ±philimx (local), `phixd_new=(phicx-phix)/tphi`, stored-slope `integrate`, writes phix/phixd, returns phix. `execute` wraps `phimvx = control_bank(vehicle, store.phicx, ctx.int_step)`. Protocol `vehicle.store`. No load/altitude/heading/mcontrol.
- Tests: `Python/tests/unit/test_plane5_control_bank.py` (turning_to_IP philimx=70, tphi=1, int_step=0.05; one- and two-step lag; limiter phicx=90 and -90).

## 0.40.0 - FALCON5 Plane5 forces
- Added `cadac.vehicles.plane5.forces.Plane5Forces` (`name="forces"`). `define` registers C++ `def_forces` `FSPV` only (vec, out, plot). Does not define `phimvx` (control) or `alphax`. No skip-if-FSPV-exists (Flat3 newton does not define FSPV). `execute` ports FALCON5 `Plane::forces`: `fspv1=(-pdynmc*area*cd+thrust*cos(alpha))/mass`, `fspv2=sin(phimv)*(pdynmc*area*cl+thrust*sin(alpha))/mass`, `fspv3=-cos(phimv)*(pdynmc*area*cl+thrust*sin(alpha))/mass` with `phimv=phimvx*RAD`, `alpha=alphax*RAD`. Protocol `vehicle.store`.
- Tests: `Python/tests/unit/test_plane5_forces.py` (pdynmc=17000, area=27.87, mass=12701, alphax=5; phimvx=0, 90, 30 vs C++ formulas).

## 0.39.0 - FALCON5 Plane5 propulsion
- Added `cadac.vehicles.plane5.propulsion.Plane5Propulsion` (`name="propulsion"`). Constructor takes Datadeck. `define` registers C++ `def_propulsion` (mprop, fidle, thrust_com, thrust, treqd/treq, fmassed/fmasse, fuelmass, mach_com, gfthm, tfth, mass, tav, mass_init, fuel_init, ff). Does not define pdynmc/mach/alt/cd/area/alphax. No unused C++ local `cg`. `initialize` sets mass=mass_init. `execute`: mprop==0 local thrust/ff=0 and return without writing; 1 commanded, 2 idle (`iff_vs_alt`), 3 max; mprop>3 Mach hold (forces 4, then 5/6 idle/max clips). Mach hold uses `integrate` and `RAD`. Fuel integrate; mass=mass_init-fmasse; fuelmass<=0 zeros thrust (mprop unchanged). Protocol `vehicle.store`.
- Tests: `Python/tests/unit/test_plane5_propulsion.py` (parsed `Falcon5_prop_deck.asc`; turning_to_IP IC mass_init=12701, fuel_init=4461, gfthm=893620, tfth=1; mprop 0/1/2/3/4/5/6 and fuel cutoff).

## 0.38.0 - FALCON5 Plane5 aerodynamics
- Added `cadac.vehicles.plane5.aero.Plane5Aero` (`name="aerodynamics"`). `define` registers C++ `def_aerodynamics` (`cl`, `cd`, `cl_ov_cd`, `area` default 27.87, `mac` int, `cla`). Does not define `alphax` (control) or `mach` (environment). `execute` branches on mac 30/35/40; 2D `look_up` of `cl_*MAC_vs_mach_alphax` / `cd_*MAC_vs_mach_alphax`; `cla=(clp-cln)/4` at alphax±2; `cl_ov_cd=cl/cd`. No drag-polar. No `time>0.5` debug. Protocol `vehicle.store`.
- Tests: `Python/tests/unit/test_plane5_aero.py` (parsed `Falcon5_aero_deck.asc`; mach=0.6, alphax=5 from store; mac 30/35/40 vs look_up replica, rtol 1e-12).

## 0.37.0 - Flat3 newton step
- `Flat3Newton.execute` ports FALCON5 `Flat3::newton`: `NEXT_ACC = TBL.T @ FSPV + [0,0,grav]`; stored-slope `integrate` of VBEL then SBEL; `ABEL=NEXT_ACC`; `polar_from_cart`; `TVL=mat2tr`; TBV from `phiavout` (0 if absent); `TBL=TBV@TVL`; `alt=-SBEL[2]`. `FSPV`/`grav` from store (forces/environment; not defined here). `dt` is `ctx.int_step`.
- Tests: `Python/tests/unit/test_flat3_newton_step.py` (after init sbel=[0,0,-3500], FSPV=0, dt=0.05: SBEL[2] increases toward 0; replica NEXT_ACC matches ABEL).

## 0.36.0 - Flat3 newton initialization
- Added `cadac.eom.flat3.Flat3Newton` (`name="newton"`). `define` registers C++ `def_newton` (`TBL`/`TBV`/`TVL`, `dvbe`, `SBEL`/`VBEL`/`ABEL`, `psivlx`/`thtvlx`, `sbel1/2/3`, `psivl`/`thtvl`, `alt`). `initialize` ports FALCON5 `Flat3::init_newton`: SBEL from sbel1/2/3, VBEL from `cart_from_pol(dvbe,psivl,thtvl)`, TVL=`mat2tr`, TBV from `phiavout` (0 if absent; FALCON5 never defines it), TBL=TBV@TVL, `alt=-SBEL[2]`. Angles in store are degrees. No `execute` newton step (Task 4). Does not define environment fields; `alt` belongs to newton.
- Tests: `Python/tests/unit/test_flat3_newton_init.py` (sbel=[0,0,-3500], dvbe=200, psivlx=0, thtvlx=0 → SBEL[2]==-3500, dvbe==200, alt==3500).

## 0.35.0 - Flat3 kinematics timing
- Added `cadac.eom.flat3.Flat3Kinematics` (`name="kinematics"`). `define` registers C++ `def_kinematics` (`time` exec scrn/plot, `event_time` exec). `initialize` sets `time=sim_time`. `execute` copies `ctx.sim_time`/`ctx.event_time` as FALCON5 `Flat3::kinematics` (timing only; no int_step). No newton. Protocol `vehicle.store` (HYPER3).
- Tests: `Python/tests/unit/test_flat3_kinematics.py` (time=1.5, event_time=0.2).

## 0.34.0 - Flat3 US76 environment
- Added `cadac.eom.flat3.Flat3Environment` (`name="environment"`). `define` registers C++ `def_environment` fields (`grav`, `rho`, `pdynmc`, `mach`, `vsound`, `press`). Does not define `alt`/`SBEL`/`dvbe` (newton). `execute` uses `alt=-SBEL[2]`, `atmosphere76`+`gravity`, writes vsound/mach/pdynmc as FALCON5 `Flat3::environment`. No ISO62. No kinematics/newton. Protocol `vehicle.store` (HYPER3).
- Tests: `Python/tests/unit/test_flat3_environment.py` (SBEL[2]=-3500, dvbe=200 vs US76/gravity; alt not registered).

## 0.33.0 - CADAC 3D table parse and look_up
- `parse_asc_deck` 3DIM packing matches FALCON5 `read_tables` (x1 rows, x2 blocks, x3 columns). Source: `Falcon5_prop_deck.asc` `ff_vs_thrust_alt_mach` 6×2×4.
- `Datadeck.look_up(name, x1, x2, x3)` trilinear as HYPER3 3D interpolate (constant upper per axis, slope lower). 1D/2D signatures unchanged.
- Tests: `Python/tests/unit/test_asc_deck_3d.py`, `Python/tests/unit/test_lookup_3d.py`.

## 0.32.0 - HYPER3 climb e2e vs plot1.csv
- `plot_row` emits CADAC plot names/order (`SBEG1` not `sbeg1`): time, FSPV1-3, pdynmc, mach, lonx, latx, alt, dvbe, psivgx, thtvgx, SBEG1-3, VBEG1-3, throttle, mass, thrust, fmassr, cl_ov_cd. `run_scenario` writes `plot.csv` beside the case when `options.plot` and `options.csv`. t=0 row is after init+execute at sim_time=0 (HYPER3). Golden: copy of CADAC `plot1.csv`. Skip sentinel `time=-1`. CSV rtol=1e-5, atol=max(1e-6, 5e-6*|g|).
- Tests: `Python/tests/e2e/test_hyper3_climb.py` (alt at t=0 and 0.2; full column grid vs golden). `test_plot_csv_written_when_plot_and_csv`.

## 0.31.1 - Reject int_step <= 0
- `run_scenario` raises `ValueError` with path and key `int_step` if the timing key is missing or `int_step <= 0` (zero would hang `run_loop`). Tests: `test_int_step_zero_raises`, `test_int_step_negative_raises`, `test_missing_int_step_raises`.

## 0.31.0 - CRUISE3 vehicle + run_scenario
- Added `cadac.vehicles.cruise3.vehicle.Cruise3`: registers environment, aerodynamics, propulsion, forces, newton. `cadac.cli.run_scenario(path)` loads JSONC, builds CRUISE3 only (ValueError on other types), applies params by name, loads aero/prop decks, define → initialize → `run_loop`. Returns `RunResult.plot_rows` (plot-flagged names; vecs as `name1..3`) on the CADAC `plot_step` grid. `cadac run` CLI wrapper. Export `run_scenario`.
- Translated HYPER3 `input_climb.asc` + aero/prop decks to `Python/cases/hyper3/` (`end_time` 90). Tests: `Python/tests/unit/test_cruise3_one_step.py` (1.0 s climb: time ≥ 1, mass < mass0, alt > 2999).

## 0.30.0 - Cruise3 specific-force module
- Added `cadac.vehicles.cruise3.forces.Cruise3Forces` (`name="forces"`). `define` registers `FSPV` (vec, out, plot) and `phimvx` (aero Task 27 omitted it). Skips if the name is already on the store (newton does not define `FSPV`). `execute` ports C++ `Cruise::forces`: `fspv1=(-pdynmc*area*cd+thrust*cos(alpha))/mass`, `fspv2=sin(phimv)*(pdynmc*area*cl+thrust*sin(alpha))/mass`, `fspv3=-cos(phimv)*(pdynmc*area*cl+thrust*sin(alpha))/mass`. No `run_scenario`.
- Tests: `Python/tests/unit/test_cruise3_forces.py` (pdynmc=28410, area=557.42, cd=0.05, cl=0.2, thrust=239241, mass=136077, alphax=7, phimvx=0; plus phimvx=90 banked).

## 0.29.0 - Cruise3 autothrottle propulsion
- `Cruise3Propulsion.execute` ports C++ `mprop==2` autothrottle: `if mprop>0` then nested 1/2. `mprop==2`: denom from first spi/ca look_up; `thrst_req=area*cd*qhold/cos(alphax*RAD)`; `throtl_req`; `gainq`; `ethrotl`; throttle; idle/max limiters; spi look_up again; thrust formula. Fuel integrate inside the `mprop>0` wrapper (`mass_flow` only if spi!=0). No forces.
- Tests: `Python/tests/unit/test_cruise3_prop_auto.py` (qhold=50000, tq=1, alphax=2.5, iso62 rho/pdynmc, cd=0.05, area=557.42; max clip and idle clip; thrust and fuel).

## 0.28.0 - Cruise3 fixed-throttle propulsion
- Added `cadac.vehicles.cruise3.propulsion.Cruise3Propulsion(deck)` (`name="propulsion"`). `define` registers C++ `def_propulsion` fields. `initialize` sets `mass=mass0`. `execute`: `mprop==0` → thrust=0, fmassd=0. `mprop==1` → `spi=look_up("spi_vs_throttle_mach", throttle, mach)`, `ca=look_up("ca_vs_alpha_mach", alphax, mach)`, `thrust=spi*0.029*throttle*AGRAV*rho*dvbe*ca*acowl`; `fmassd_next=thrust/(spi*AGRAV)` if spi!=0; `fmasse=integrate(...)`; `mass=mass0-fmasse`; `fmassr=fmass0-fmasse`; if fmassr<=0 then mprop=0. No autothrottle mprop=2 (Task 29). Prop deck `ghame3_prop_deck.asc`.
- Tests: `Python/tests/unit/test_cruise3_prop_fixed.py` (mprop=0; mprop=1 climb IC throttle=0.2, mass0=136077, fmasse=0, numeric rho/dvbe; fuel cutoff).

## 0.27.0 - Cruise3 drag-polar aerodynamics
- Added `cadac.vehicles.cruise3.aero.Cruise3Aero(deck)` (`name="aerodynamics"`). `define` registers alphax, area, cl, cd, cla, cl_ov_cd. `execute` reads store `mach` (environment) and `alphax`; `cl=cla0+cla*alphax`, `cd=cd0+ckk*(cl-cl0)**2`, `cl_ov_cd=cl/cd` with tables `cd0_vs_mach`, `cl0_vs_mach`, `cla_vs_mach`, `ckk_vs_mach`, `cla0_vs_mach`. No propulsion/forces. Does not define `phimvx`.
- Tests: `Python/tests/unit/test_cruise3_aero.py` (parsed `ghame3_aero_deck.asc`; mach=0.760854, alphax=7; expected via look_up then formulas).

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
