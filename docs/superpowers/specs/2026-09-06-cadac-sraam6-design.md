# CADAC SRAAM6 (MISSILE6 / TARGET3 Flat) — design

Date: 2026-09-06
Status: approved (chat). Second slice plan 3. Depends on HYPER5 kernel skip + deck-optional factory (already shipped). Family registry is new in this plan if missing.

Parent: `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
Siblings: `docs/superpowers/specs/2026-09-06-cadac-hyper5-design.md`, `docs/superpowers/specs/2026-09-06-cadac-hyper6-design.md`

## Goal

Run CADAC SRAAM6 from JSONC: family `sraam6` types `MISSILE6` (Flat6 missile) and `TARGET3` (Flat3 target aircraft). First e2e is `input_1v1.asc` (1 missile + 1 target, `ENDTIME` 12). Numerics regression-close to `CADAC_Simulations/SRAAM6_250130/SRAAM6/`.

## Non-goals

SAM6, AGM6, Plane6 vehicle modules. HYPER5 Round3 `TARGET3` behavior (stays the global `TARGET3`). Multi-vehicle decks (`input_3v1_*`, `input_2v2`, `input_5v1_*`, `input_1v1_sweep`). Monte Carlo / `markov_noise` / `GAUSS`/`MARKOV` sampling. `cadac translate-asc` CLI. CADAC packet-slot indexing (`Packet data[i]`). `sys.exit` / `cout` intercept banners.

## Layout

```
Python/src/cadac/vehicles/flat6/sraam6/
  vehicle.py       # Sraam6Missile type="MISSILE6"
  target.py        # Sraam6Target type="TARGET3"
  environment.py   # SRAAM6 Flat6 environment (not FALCON6 Flat6Environment)
  kinematics.py    # SRAAM6 Flat6 kinematics (not FALCON6 Flat6Kinematics)
  euler.py         # SRAAM6 Flat6 euler (ai11/ai33, not IBBB)
  aero.py
  propulsion.py
  seeker.py
  guidance.py
  control.py
  actuator.py
  forces.py
  tvc.py
  intercept.py
Python/src/cadac/eom/flat3.py   # add aircraft-named Flat3 classes (SAEL/dvae)
Python/cases/sraam6/            # translated input_1v1.asc + both decks
Python/tests/unit/test_sraam6_*.py
Python/tests/e2e/test_sraam6_1v1.py
```

Reuse `cadac.eom.flat6.Flat6Newton` (body-velocity stored-slope matches SRAAM6 `newton.cpp`). Do **not** import `cadac.vehicles.flat6.falcon6` or SAM6/AGM6 (those packages may not exist). Do **not** reuse `Flat6Euler` (FALCON6 `IBBB`/`eng_ang_mom` vs SRAAM6 `ai11`/`ai33`). Do **not** reuse `Flat6Kinematics` (FALCON6 incidence from `VBAL`; SRAAM6 from `VBEB`; SRAAM6 writes `time`/`trcond`/`int_step_new`). Do **not** reuse `Flat6Environment` (SRAAM6 Mach from `dvbe` plus `mguid==6` `trcond` and `mfreeze` latch).

SRAAM6 Target is **not** HYPER5 Round3 `Target3`. Reuse Flat3 numerics (US76, stored-slope, `TAL=TAV@TVL`) with SRAAM6 names (`SAEL`, `VAEL`, `dvae`, `FSPA`, `sael1`/`sael2`/`sael3`, `psialx`/`thtalx`). Add `Flat3AircraftEnvironment` and `Flat3AircraftNewton` in `cadac.eom.flat3` so FALCON5 `PLANE` (`SBEL`/`dvbe`/`FSPV`) is unchanged. Reuse `Flat3Kinematics` (time only).

## Family registry

Match the AIM5 canonical API (`docs/superpowers/specs/2026-09-06-cadac-aim5-design.md`). **`VehicleSpec.family` is the source of truth.** Do **not** invent `RunConfig.family` as the primary store (`RunConfig` has no family field).

JSONC vehicle may include `"family": "sraam6"`. Optional scenario-level `"family"` is copied onto vehicles that omit it at load time. Vehicle-level `"family"` wins.

`translate_scenario_asc(..., family="sraam6")` writes `"family"` on **each vehicle** (not a scenario-root key). `family is None` → no `family` key on vehicles.

Lookup in `_build_vehicle(path, spec)` (two arguments; family comes from `spec.family`):

| `spec.family` | Table | Example |
|---|---|---|
| set | **only** `_VEHICLE_FAMILIES[(family, type)]` | `("sraam6","MISSILE6")` → `Sraam6Missile` |
| `None` | **only** `_VEHICLE_TYPES[type]` | `"TARGET3"` → HYPER5 `Target3` |

No fallthrough from families to `_VEHICLE_TYPES`. Missing pair → `ValueError` including `type` and `family`. Missing global type → `ValueError` including `type`.

Register:

- `("sraam6", "MISSILE6")` = `Sraam6Missile`
- `("sraam6", "TARGET3")` = `Sraam6Target`

Never put SRAAM6 `TARGET3` in `_VEHICLE_TYPES`. HYPER5 `TARGET3` stays global. `AIM5` is (or will be) a **global** `_VEHICLE_TYPES` token — do not treat it as the unknown-type sentinel and do not retarget existing AIM5 unknown-type tests.

If `VehicleSpec.family` / `_VEHICLE_FAMILIES` already exist (AIM5), reuse them idempotently. If missing, Task 1 adds that AIM5 API (not `RunConfig.family`).

## JSONC types (family `sraam6`)

| `type` | Class | Constructor | Decks |
|---|---|---|---|
| `MISSILE6` | `Sraam6Missile` | `(name, aero_deck, prop_deck, events=None)` | **both required** |
| `TARGET3` | `Sraam6Target` | `(name, events=None)` | none (C++ `Target::read_tables` is a dummy; `input_1v1.asc` Target has no `AERO_DECK`/`PROP_DECK`) |

Do not add SRAAM6 `TARGET3` to `_NO_DECK_TYPES`. Deck skip is `(family, type)` or the Target constructor, not the global HYPER5 set.

## Kernel / I/O

- `run_loop` already skips scenario modules absent on a vehicle (HYPER5). Target has no aero/prop/seeker/actuator/euler/tvc/intercept.
- `plot_rows` / `plot.csv`: vehicle slot 0 only (the missile). Plot-flagged store names, not a hard-coded column list.
- Seeker/guidance/target-escape read combus **by `Packet.type` and field names**, not C++ `data[i]` slots and not `id.find("t")`.
- **`tgt_num` as C++:** 1-based index among packets with `type=="TARGET3"` in combus order (C++ packet id `"t"+tgt_num`). `input_1v1.asc`: `tgt_num=1` is the only target. `sht_num` (default 0) uses the same TARGET3 index; `sht_num==0` matches nothing, `SSEL` stays zero.
- **`msl_num`:** 1-based index among packets with `type=="MISSILE6"` (1v1: `msl_num=1` is the missile; coincides with C++ `combus[msl_num-1]`).
- Target kinematics published as `SAEL`/`VAEL` (not `STEL`/`VTEL`). Missile seeker copies those into local `STEL`/`VTEL`.
- `com_names` = store fields with `"com"` in `outputs` (same as other vehicles). Missile C++ `com` set includes `time`, `vmach`, `SBEL`, `VBEL`, `mseek`. Target C++ `com` set includes `dvae`, `SAEL`, `VAEL`, `psial`, `thtal`, `phiavx`, `anx`.
- No `sys.exit`. Intercept sets `vehicle.health=0` and `ctx.combus[slot].status=0` (and the tracked target’s status on hit). No console banners.

## Monte Carlo

Out of scope (no sampling, no `markov_noise`). Shared translator (ROCKET6/AGM6) stores deterministic prefixes: `GAUSS name mean sigma` → `params[name]=mean`; `MARKOV` → `0` (or the stored mean if that parser already does). **Do not skip** those tokens if that parser exists — 1v1 GAUSS means are 0 (`biast`,`biasp`,`biaseh`). If the parser is still absent, add store-means (do not rip a sibling translator). `MONTE` stays ignored at scenario level. Runtime uses those stored values (zeros for 1v1).

## First case (`input_1v1.asc`)

Translate `CADAC_Simulations/SRAAM6_250130/SRAAM6/input_1v1.asc` + `sraam6_aero_deck.asc` + `sraam6_prop_deck.asc` to `Python/cases/sraam6/`. Each vehicle `"family": "sraam6"`. `end_time` **12** (from `ENDTIME`). `int_step` 0.001, `plot_step` 0.05.

Vehicles:

1. `MISSILE6` Missile — ICs: `sbel1=0`, `sbel2=0`, `sbel3=-5000`, `psiblx=thtblx=phiblx=0`, `alpha0x=beta0x=0`, `dvbe=250`. `tgt_num=1`, `mterm=1`. `alplimx=46`. `mprop=1`, `aexit=0.0125`. `mact=2`, `dlimx=28`, `ddlimx=600`, `wnact=100`, `zetact=0.7`. `maut=2`, `alimit=50`, `dqlimx=drlimx=dplimx=28`, `phicomx=0`, `wrcl=20`, `zrcl=0.9`, `zetlagr=0.6`. `mseek=2`, `ms1dyn=1`, `dblind=3`, `racq=7000`, `dtimac=0.250`, `gk=10`, `zetak=0.9`, `wnk=60`, `fovyaw=fovpitch=0.03140`, `trphid=14`, `trtht=1`, `trthtd=10`, `trate=1`. `mnav=3`, `gnav=3.75`. Event `time>0.25` sets `maut=3`, `mguid=3`.
2. `TARGET3` Target aircraft — `msl_num=1`, `tgt_option=1`, `gturn=1`, `guid_gain=1`, `sael1=10000`, `sael2=500`, `sael3=-2000`, `psialx=180`, `thtalx=0`, `dvae=250`. No decks.

CADAC MODULES: `environment`, `kinematics`, `aerodynamics`, `propulsion`, `seeker`, `guidance`, `control`, `actuator`, `forces`, `euler`, `newton`, `intercept`. `tvc` is defined on the missile so JSONC may list it; 1v1 omits it from MODULES (define still runs, execute skipped). `mtvc` default 0.

## Missile modules (1v1 modes)

Port SRAAM6 `aerodynamics.cpp`, `propulsion.cpp`, `seeker.cpp`, `guidance.cpp`, `control.cpp`, `actuator.cpp`, `forces.cpp`, `tvc.cpp`, `intercept.cpp`, `missile_functions.cpp` (combus/define names only; not Monte Carlo). Unused mode integers → `ValueError`.

**Environment:** US76 + `grav=G*EARTH_MASS/(REARTH+hbe)**2`. Mach/dynamic pressure from `dvbe`. If `mguid==6`: `vmach<=trmach` → `trcond=2`; `pdynmc<=trdynm` → `trcond=3`. `mfreeze` latch as C++. Skip `mfreeze` if the name is absent.

**Kinematics:** `time=ctx.sim_time`; `int_step_new`/`out_step_fact` like other Flat6 (`ctx.int_step`/`out_fact`). Quaternion init/step as SRAAM6 `kinematics.cpp` using **`pp`,`qq`,`rr`** (not FALCON6 `WBEB` for the derivative). Incidence from **`VBEB`**. `phip` ports **both** C++ branches: `fabs(vbeb2)<EPS && fabs(vbeb3)<EPS` → `0`; else if `fabs(vbeb2)<SMALL` → `atan2(SMALL, vbeb3)`; else `atan2(vbeb2, vbeb3)`. `EPS` from `cadac.constants`; `SMALL=1e-7` module-level. `ortho_error` vs `trcvel` (see aero init) → `trcond=1`; `alpp>tralp` → `trcond=5`. Defines `halt`,`stop`,`lconv`.

**Euler:** Port `euler.cpp`. `ppd_new=FMB[0]/ai11`; `qqd_new=((ai33-ai11)*pp*rr+FMB[1])/ai33`; `rrd_new=(-(ai33-ai11)*pp*qq+FMB[2])/ai33`; stored-slope `integrate`; write `WBEB=(pp,qq,rr)` and `ppx`/`qqx`/`rrx` in deg/s so `Flat6Newton` can run.

**Newton:** `cadac.eom.flat6.Flat6Newton`. Skip-if-exists on `define` name collisions (`time`, `SBEL`, …).

**Aerodynamics:** Tables from `sraam6_aero_deck.asc` (`ca0_vs_mach`, `caa_vs_mach`, `cad_vs_mach`, `caoff_vs_mach`, `cyp_vs_mach_alpha`, `cn0_vs_mach_alpha`, `cnp_vs_mach_alpha`, `cndq_vs_mach_alpha`, `cllap_vs_mach_alpha`, `cllp_vs_mach_alpha`, `clldp_vs_mach_alpha`, `clm0_vs_mach_alpha`, `clmp_vs_mach_alpha`, `clmq_vs_mach`, `clmdq_vs_mach_alpha`, `clnp_vs_mach_alpha`). Body/aeroballistic transform as C++. `init_aerodynamics` sets `refl=0.1524`, `refa=0.01824`, `trcvel=10e-4`, `trmach=0.5`, `trdynm=10e+3`, `trload=3`, `tralp=1`, `trcond=0` **after** JSONC params (C++ `vehicle_data` then `init_aerodynamics`). Store field name is `trcvel` (C++ slot 182); kinematics orthogonality and terminal closing-speed both read it. Then `aerodynamics_der` as C++ (bypass if `alppx>=alplimx-3`).

**Propulsion:** `mprop==0` thrust 0 (mass/xcg/ai stay). `mprop==1` look up `thrust_vs_time`, `mass_vs_time`, `cg_vs_time`, `moipitch_vs_time`, `moiroll_vs_time`; `thrust=tsl+(101325-press)*aexit`; if `time>2.69` set `mprop=0`. Else `ValueError`. `mfreeze` latch as C++. Defaults from `def_propulsion`: `vmass=92`, `xcgref=xcg=1.536`, `ai11=0.308`, `ai33=59.80`.

**Actuator:** `mact==0` (or C++ `mact<2`) position limit only; `mact==2` `actuator_scnd`. Else `ValueError`. Four-fin mix as C++. CADAC `sign` (`<0 → -1` else `+1`); local helper, not `np.sign`, not `flat6._cadac_sign`.

**Control:** `maut==0` return. `maut>=1` calls `control_roll`. `maut==2` `control_rate`. `maut==3` `control_accel`. 1v1 uses 2 then 3. Allowed `{0,1,2,3}`; else `ValueError`. `SMALL=1e-7` module-level; do not add to `cadac.constants`. `control_accel` overwrites local `wacl`/`zacl`/`pacl` from `pdynmc` as C++ (`pacl=14`).

**Guidance:** Always extrapolate target from last `mnav==3` snapshot. `mguid==0` no accel law. `mguid==3` `guidance_mid`. `mguid==6` `guidance_term`. Else `ValueError`. `mnav==3` latch then set `mnav=0`; `mnav==0` keep stored; other `mnav` → `ValueError`. Seeker lock (`mseek==4`) sets `mguid=6` as C++.

**Seeker:** Download TARGET3 by `tgt_num`. `mseek==0` download/geometry only, return. `==2` enable; if `dbt<racq` set 3. `==3` acquire (`ms1dyn==1` dynamic + FOV + `dtimac`; `==0` kinematic + `dtimac`). `==4` lock, `mguid=6`, output LOS rates. `==5` blind (`dbt<dblind`), hold. Else `ValueError`. `ms1dyn` in `{0,1}` else `ValueError`. `TTL` identity shortcut as C++. `seeker_uthpb` copies C++ operator precedence: `fabs(sinpsi) && fabs(tantht)<SMALL` (not a two-abs conjunction). Break-lock sets `mseek=2`, `mguid=3`, `trcond` 6–9.

**TVC:** `mtvc==0` return (1v1). `==1` no dynamics. `==2` `tvc_scnd`. `==3` inline `gtvc` then scnd. Else `ValueError`. Forces: if `mtvc==0` or field absent, add `thrust` to `FAPB[0]`; else add `FPB`/`FMPB`. Port C++ diagnostic assignment (`etacx`/`zetcx` written from `etax`/`zetx`).

**Forces:** `FAPB`/`FMB` as `Missile::forces`. Do not write `FSPB` (newton-owned).

**Intercept:** Port `intercept.cpp` (no `cout`/`exit`). `mterm` in `{0,1,2}` else `ValueError`. 1v1 is `mterm=1` (I-plane). Sphere `dbt<100` and `mseek>=3`. C++ `closing_speed = UTBL·VTBEL` is LOS range-rate of target wrt missile (positive = **opening** after CPA). Fire when `closing_speed>0 && write` (first opening sample after closest approach); then `write=0`. Do not invert to “range-rate went non-positive.” Halt / ground (`alt=-SBEL[2]<=0`) / `trcond && stop` set missile dead. Hit sets missile and target `status=0`. `write` latch as C++ (`int` default 1). No `sys.exit`.

## Target3 (SRAAM6)

Modules: `environment`, `kinematics`, `newton`, `guidance`, `control`, `forces`. Dummy the rest (kernel skip).

**Guidance:** `tgt_option==0` gravity bias `ACOML=(0,0,-grav)`. `==1` horizontal g-turn `ACOMV=(0,gturn*grav,-grav)`, `ACOML=TVL.T@ACOMV`. `==2` escape when attacking missile `mseek%10==4` as C++ (1v1 uses 1; still port). Else `ValueError`.

**Control:** Bank and load-factor lags (`tphi`/`tanx` default 0 → no lag). `philimx=120`, `alplimx=40`, `clalpha=0.0523`, `wingloading=3247`. Writes `phiavout` for newton. CADAC `sign`.

**Forces:** `FSPA=(acc_longx*grav, 0, -anx*grav)`. `acc_longx` default 0.

## Testing

Unit: `rtol=1e-12`, `atol=1e-14` vs CADAC formulas. TDD each task. Tests `test_sraam6_*.py`. Parse ASC decks in unit tests (JSONC case is the vehicle task).

E2E: `tests/e2e/test_sraam6_1v1.py`. Skip if `tests/e2e/goldens/sraam6/plot.csv` absent (do not create the file). Else `run_scenario` on the translated 1v1 case; compare missile plot columns present in both; skip sentinel `time=-1`; CSV `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`. Pattern `test_falcon6_gamma.py`.

Regression: HYPER3 e2e; HYPER5/HYPER6/FALCON5/FALCON6 units on commits that touch kernel/cli/translate/plot/`flat3.py`.

## Process

Grok implementer `cursor-grok-4.6-high` per task; Grok task reviewer; Grok whole-plan review. No Composer Fast / Kimi / Fast. Controller does not patch physics. Isolated worktree; no commit to main without user ask.

## Python style

Named state, numpy `@`, CADAC numeric order copied not “improved.” Same module **name** binds Sraam6Missile vs Sraam6Target vs Plane6/Hyper5 by vehicle class. Local CADAC `sign`; `SMALL=1e-7` module-level where C++ uses it.
