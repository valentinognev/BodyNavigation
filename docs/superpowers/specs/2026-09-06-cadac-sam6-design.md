# CADAC SAM6 (Missile + Aircraft + Rocket + Radar) — design

Date: 2026-09-06
Status: approved (parent dispatch). Depends on first-slice kernel + HYPER5 kernel skip/deck-optional factory.

Parent: `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
Siblings: `docs/superpowers/specs/2026-09-06-cadac-hyper5-design.md`, `docs/superpowers/specs/2026-09-06-cadac-hyper6-design.md`
Canonical family API: `docs/superpowers/specs/2026-09-06-cadac-aim5-design.md` (SAM6 matches AIM5; does not invent a second store).

## Goal

Run CADAC SAM6 from JSONC. Register family pairs only: `("sam6","MISSILE6")`, `("sam6","AIRCRAFT3")`, `("sam6","ROCKET5")`, `("sam6","RADAR0")`. First e2e is `input_SAM_autopilot.asc` (1× MISSILE6, no radar, ENDTIME 30). Numerics regression-close to `CADAC_Simulations/SAM6_250217/SAM6/` with **zero Monte Carlo**. The program is complete: all four types are implemented, registered, and unit/smoked — not missile-only.

## Non-goals

HYPER6 Ground0 radar. Monte Carlo / `gauss` / `uniform` / `MARKOV` / `GAUSS` draws (treat as 0). `cadac translate-asc` CLI. CADAC packet-slot indexing and C++ ids `m1`/`a1`/`r1`/`f1`. Unused missile modes (`maut==4`, `mtvc!=0`, `mrcs_moment!=0`, `mrcs_force!=0`, `mins` not in `{0,1}`, `mguide` mid==3). Do not import Plane6 / Plane5 / AIM5 vehicle modules. Do not import `cadac.eom.flat6` classes (`Flat6Environment`, `Flat6Kinematics`, `Flat6Euler`, `Flat6Newton`). Do not edit `flat6.py` (FALCON6). Do not overwrite global `_VEHICLE_TYPES`. Do not retarget unknown-type tests that still use `"AIM5"`. Do not wipe `_VEHICLE_FAMILIES` if AIM5 (or another family) already populated it.

## Layout

```
Python/src/cadac/eom/flat0.py          # Flat0 kinematics + newton (parent reserved)
Python/src/cadac/vehicles/sam6/
  vehicle.py      # Sam6Missile type="MISSILE6"
  environment.py  # Sam6Environment — port SAM6 environment.cpp (not Flat6Environment)
  kinematics.py   # Sam6Kinematics — port SAM6 kinematics.cpp (copy quaternion math; not Flat6Kinematics)
  euler.py        # SAM6 euler.cpp (ai11/ai33) — not Flat6Euler
  newton.py       # SAM6 newton.cpp (mass, alt+hbe)
  aero.py
  propulsion.py
  forces.py
  actuator.py
  control.py
  guidance.py
  ins.py
  sensor.py
  intercept.py
  tvc.py
  rcs.py
  aircraft.py     # AIRCRAFT3 modules + (later) Sam6Aircraft
  rocket.py       # ROCKET5 modules + (later) Sam6Rocket
  radar.py        # Sam6Radar type="RADAR0" + radar sensor
  flat3.py        # SAM6 Flat3 names (SAEL/VAEL/dvae/FSPA/TAL)
Python/cases/sam6/                    # translated input_SAM_autopilot.asc + SAM decks
Python/tests/unit/test_sam6_*.py test_flat0_*.py
Python/tests/e2e/test_sam6_autopilot.py
```

C++ source of truth: `CADAC_Simulations/SAM6_250217/SAM6/`.

## JSONC types (family `sam6` only)

| `type` | Class | Decks |
|---|---|---|
| `MISSILE6` | `Sam6Missile` | `aero_deck` required; `prop_deck` required |
| `AIRCRAFT3` | `Sam6Aircraft` | none |
| `ROCKET5` | `Sam6Rocket` | `aero_deck` required (`SRBM_aero_deck`); no `prop_deck` (analytic motor) |
| `RADAR0` | `Sam6Radar` | no aero/prop. `sam_deck` + `srmb_deck` required when `mtrack==1`; unused when `mtrack==0` or `2` |

`AIRCRAFT3` / `ROCKET5` / `RADAR0` collide with AIM5/AGM6 later — **family-only**. Never insert these keys into `_VEHICLE_TYPES`. `TARGET3` stays the global HYPER5 binding.

Unknown types still `ValueError`. `CRUISE3` / `PLANE` / `PLANE6` / `HYPER5` / `HYPER6` / `TARGET3` / `SATELLITE3` stay on `_VEHICLE_TYPES`.

## Kernel: family registry (AIM5 canonical)

Family API may already exist (AIM5). This plan adds or keeps it **idempotently**. Do **not** replace `_VEHICLE_FAMILIES` with `{}`. Do **not** make `RunConfig.family` the only store. `RunConfig` has **no** family field (AIM5).

**`VehicleSpec.family` is the source of truth.**

- `VehicleSpec.family: str | None = None`. JSONC vehicle may include `"family": "sam6"`.
- Optional scenario-level `"family"` is a **load-time default**: copied onto vehicles that omit a vehicle-level key. Vehicle-level `"family"` **wins**.
- `VehicleSpec` also carries optional `sam_deck` / `srmb_deck` (`Path | None`) for radar traj tables. Translate maps CADAC `SAM_DECK` / `SRBM_DECK`.
- Lookup in `_build_vehicle(path, spec)` — reads **`spec.family`**, not a scenario-level argument:
  - `spec.family` set → resolve **only** `_VEHICLE_FAMILIES[(spec.family, spec.type)]`. Do not fall through to `_VEHICLE_TYPES`. Missing → `ValueError` mentioning type **and** family (e.g. `{path}: unknown vehicle type {type!r} for family {family!r}`).
  - `spec.family` is `None` → `_VEHICLE_TYPES[type]` only. Missing → `ValueError` mentioning type.
- `register_family_type(family, type_name, cls)` writes `_VEHICLE_FAMILIES[(family, type_name)] = cls`. Same class twice is a no-op. Different class for an occupied pair is `ValueError`. Never mutates `_VEHICLE_TYPES`. Never clears other families' pairs.
- Register exactly (do not unregister AIM5 or others):
  - `("sam6","MISSILE6")` → `Sam6Missile`
  - `("sam6","AIRCRAFT3")` → `Sam6Aircraft`
  - `("sam6","ROCKET5")` → `Sam6Rocket`
  - `("sam6","RADAR0")` → `Sam6Radar`
- `translate_scenario_asc(src, dst_dir, family=None)`. If `family` is given, stamp `"family"` on **each vehicle** (not a scenario-root-only key). Autopilot ASC has a duplicate bare `ENDIF`; translator skips unmatched `ENDIF`.
- Do not retarget tests that use `"AIM5"` as the unknown type.

Factory (`_build_vehicle(path, spec)`):

- `MISSILE6`: require aero + prop decks.
- `AIRCRAFT3`: `Sam6Aircraft(name, events)` — no decks.
- `ROCKET5`: require aero; `prop_deck` must be absent (error if present).
- `RADAR0`: `Sam6Radar(name, events, sam_deck=..., srmb_deck=...)`. If `params.mtrack==1` (default 0), both traj decks required; else traj decks optional and unused.

## Packet identity (no C++ ids)

Identify combus packets by `Packet.type` and **field names**, never `id.find` / `"m1"` / `"a1"` / `"r1"` / `"f1"` / CADAC `data[i]` slots.

| Need | How |
|---|---|
| Radar | `packet.type == "RADAR0"`; read `lnch_delay_m1`/`m2`/`m3`, `SIEL1`/`SIEL2`/`SIEL3` |
| Aircraft target | `packet.type == "AIRCRAFT3"`; kinematics `SAEL`, `VAEL` |
| Rocket target | `packet.type == "ROCKET5"`; kinematics `SAEL`, `VAEL` |
| Missile | `packet.type == "MISSILE6"`; `SBEL`, `VBEL` |
| Pairing | k-th `MISSILE6` in combus order uses radar `lnch_delay_m{k+1}` and `SIEL{k+1}`; k-th missile vs k-th `AIRCRAFT3` or `ROCKET5` of that type (same order among that type). Do not assume missiles occupy slots 0..n-1 if JSONC lists another type first — index **among that type**. |

`com_names` = store fields with `"com"` in outputs (C++ `com` flags). Radar com at least `time`, `lnch_delay_m1..3`, `SIEL1..3`. Aircraft/rocket com include `time`, `SAEL`, `VAEL`, `psivlx`, `thtvlx`, `alt`, plus module `com` fields.

Radar delay names are **not** interchangeable:

| Name | C++ role | Default | After track commit |
|---|---|---|---|
| `lnch_delay_m1..3` | sensor **com** (uploaded to combus) | **0** | `launch_delay{k} + lnch_dly_bias{k}` (`mtrack==2`) or `launch_delay + lnch_dly_bias{k}` (`mtrack==1`) |
| `launch_delay` | sensor **save** (rocket path, one shared) | **9999** | set at rocket apogee from traj look_up |
| `launch_delay1..3` | sensor **save** (aircraft path) | **9999** | `sim_time` when that aircraft enters lethal range (`mtrack==2`) |

Missile kinematics reads **com** `lnch_delay_m*`, never the save `launch_delay*`.

## Flat0 (new EOM)

Parent reserved `Python/src/cadac/eom/flat0.py`. Ground0 (HYPER6 round-earth radar site) is **out of this plan**.

Port `flat0_modules.cpp`:

**Kinematics:** `time`, `launch_delay`, `launch_epoch`, `launch_time`. Init: `time=sim_time`, `launch_epoch=launch_delay`. Exec: `launch_time=sim_time-launch_epoch`, `time=sim_time`. No quaternion, no atmosphere.

**Newton:** data `srel1`/`srel2`/`srel3`; out `SREL`. Init packs `SREL=(srel1,srel2,srel3)`. Exec is a no-op (fixed site).

No environment, no euler, no forces on Flat0.

## SAM6 Flat6 (missile) — vehicle-local, not FALCON6

FALCON6 `cadac.eom.flat6.Flat6Environment` / `Flat6Kinematics` are **not** SAM6 Flat6. They use `hbe`/`vmach`/`mwind` and FALCON6 incidence. **Do not import those classes. Do not edit `flat6.py`.** Reuse is OK only for helpers you **copy** (`integrate`, `atmosphere76`, quaternion derivative algebra).

Port SAM6 `environment.cpp` and `kinematics.cpp` in the sam6 package.

**`Sam6Environment`** (`environment.py`):

- Read altitude as C++ `alt`. Newton writes both `alt` and `hbe` with `hbe=alt`. First environment execute requires Newton init first: `VBEL` (and `dvbe`) must already exist — unit-test asserts `VBEL` from newton init before env exec.
- US76 via `atmosphere76(alt)`; gravity `G*EARTH_MASS/(REARTH+alt)**2` as C++ (not FALCON6 `gravity(hbe)` unless that helper is bit-identical — prefer the C++ formula).
- C++ `mach` → store **`vmach`**. Missile aero/control/INS that C++ labels `mach` read `vmach`.
- No wind (SAM6 environment has none). Skip `mfreeze` latch if the name is absent.
- `guid_term==6` and `pdynmc<=trdynm` → `trcond=3` as C++.

**`Sam6Kinematics`** (`kinematics.py`):

- Port SAM6 clocks: `time`, `launch_delay` (default 99999 as C++), `launch_epoch`, `launch_time`, `msl_time`, `stop`, `lconv`, `int_step_new`, `out_step_fact`.
- Quaternion integrate: **copy** SAM6/Flat6 quaternion math locally (init from Euler, stored-slope `integrate`, TBL from q, Euler extract). Do not subclass or call `Flat6Kinematics`.
- Incidence from **`VBEB`**: `alpha=atan2(vbeb3,vbeb1)`, `beta=asin(vbeb2/||VBEB||)`, `alpp`/`phip` and `tralp` → `trcond=2`; `trortho` on quaternion error → `trcond=1`. As C++ `kinematics.cpp`.
- `msl_time`: if a `RADAR0` packet exists, `lnch_delay` from that packet’s **com** `lnch_delay_m{k+1}` for this missile’s index among `MISSILE6`; else C++ default `lnch_delay=0` (do not substitute kinematics `launch_delay`). `msl_time=max(0, sim_time-lnch_delay)`. Autopilot has no radar → `msl_time=sim_time`.

**`Sam6Euler` / `Sam6Newton`:** SAM6 C++ Euler and Newton are **not** FALCON6:

- SAM6 `euler.cpp` integrates `pp`/`qq`/`rr` with `ai11`/`ai33` (no `IBBB`, no engine angular momentum). Port as `Sam6Euler`.
- SAM6 `newton.cpp` uses `mass` (not `vmass`) and writes `alt`. Port as `Sam6Newton`. Same stored-slope integrate of `VBEB`/`SBEL` as C++. Also write `hbe=alt`. Skip `mfreeze` if the name is absent. Init writes `VBEB` and `VBEL` (environment’s first exec depends on this).

## Flat3 reuse (aircraft + rocket)

Do **not** modify `cadac.eom.flat3` (PLANE uses `SBEL`/`dvbe`/`FSPV`). SAM6 Flat3 names are `SAEL`/`VAEL`/`dvae`/`FSPA`/`TAL`/`TAV`/`phiavout`. Port SAM6 `flat3_modules.cpp` as `Sam6Flat3Environment` / `Sam6Flat3Kinematics` / `Sam6Flat3Newton` in `vehicles/sam6/flat3.py`. Reuse `atmosphere76`, `gravity`, `integrate`, `mat2tr`, `polar_from_cart`. C++ comments NASA Marshall US76 — **use existing `atmosphere76`** (parent Flat3 policy). Skip `us76_nasa2002`.

Kinematics: `time`, `launch_delay`, `launch_epoch`, `launch_time` as SAM6 Flat3.

Newton: ICs `sael1/2/3`, `dvae`, `psivlx`, `thtvlx`; `NEXT_ACC = TAL.T @ FSPA + (0,0,grav)`; write `SAEL`/`VAEL`/`alt`/`TAL`/`TVL`. `phiavout` 0 if absent (rocket).

Environment: alt = `-SAEL[2]`; `mach` from `dvae` (SAM6 name `mach`, not `vmach`).

## First case (e2e)

Translate `CADAC_Simulations/SAM6_250217/SAM6/input_SAM_autopilot.asc` + `SAM_aero_deck.asc` + `SAM_prop_deck.asc` to `Python/cases/sam6/` with `family="sam6"` stamped on the vehicle. `end_time` 30 from file. One vehicle `MISSILE6` name `SAM`.

CADAC MODULES: `environment`, `kinematics`, `propulsion`, `aerodynamics`, `ins`, `guidance`, `control`, `actuator`, `forces`, `euler`, `newton`, `intercept`. No sensor/tvc/rcs in MODULES (kernel skip). Still **define** sensor/tvc/rcs on the missile so JSONC may list them.

In-scope autopilot flags (else `ValueError` inside that module):

| Flag | Autopilot | Implement | Else |
|---|---|---|---|
| `mact` | 2 | `mact<2` position-limit (includes **1**); `mact==2` second-order `actuator_scnd` | other → `ValueError` |
| `maut` | 2 then event 3 | 0 return; ≥1 roll; 2 rate; 3 accel | `4` and other → `ValueError` |
| `mins` | 1 | 0 ideal copy; 1 error algebra with **zero** `gauss`/`uniform` | 2, 3, other → `ValueError` |
| `mguide` | default 0 | 0 (still writes `ancomx`/`alcomx` from zero ACBX + circular limiter); mid=2 line; term=6 IR comp-PN; term=7 RF PN | mid=3 → `ValueError` |
| `mtvc` | absent 0 | 0 no TVC: add `thrust` to `FAPB[0]` | `!=0` → `ValueError` this slice (do not add `FPB`/`FMPB`) |
| `mrcs_moment` / `mrcs_force` | absent 0 | 0 zeros `FMRCS`/`FARCS` | else `ValueError` |
| `mseek` | module omitted | 0 return; RF `skr_type==1` modes 2/3/4; IR `skr_type==2` as C++ | other `skr_type` → `ValueError` |
| `mfreeze` | absent | skip latch if name missing | — |

Events: `time>5` `maut=3` + accel biases; pulses `ancomx_test` at 10/13/16/19/22/25. `launch_delay` 0. `dvbe` 16, `thtblx` 80.

CADAC sign: `<0 → -1` else `+1`. Local helper; not `np.sign`; not `flat6._cadac_sign`. `SMALL=1e-7` module-level in sam6 files (C++ `global_constants.hpp`); do not add to locked `cadac.constants`.

Autopilot `mins=1` with every `gauss`/`uniform` **zeroed**. The e2e golden is a **zero-MC** plot, not a raw CADAC `plot.csv` from a Monte-Carlo-on run.

## Missile modules

Include on `Sam6Missile` in this order (define always; exec follows JSONC MODULES): environment, kinematics, propulsion, aerodynamics, ins, sensor, guidance, control, actuator, tvc, rcs, forces, euler, newton, intercept.

**Aerodynamics** (`aerodynamics.cpp` + `aerodynamics_der`): 3D/1D look_up as C++ table names (`ca0_vs_mach,betax,alphax`, `cad_vs_mach`, `cab_vs_mach` if `mprop==0`, `cy0`/`cydr`, `cn0`/`cndq`, `cll0`/`cllp`/`clldp`, `clm0`/`clmq`/`clmdq`, `cln0`/`clnr`/`clndr`). `deffx` mean abs of four fin deg. CG moment shift `-cn/refl*(xcgref-xcg)` and yaw analogue. `gmax`/`gavail` vs `alimitx`. Then `aerodynamics_der` dimensional derivatives `dna`/`dma`/`dmq`/`dmd`/`dlp`/`dld`/`dnd`/`dnr`/`dyb`/`dnb`/`dlnd` and pitch/yaw roots `realq1`/`realq2`/`wnq` as C++. Read `vmach` not `mach`. `refl=0.25`, `refa=0.0491` from C++ def. Init termination `trortho`/`tralp`/`trdynm`/`trload` as C++ `init_aerodynamics`.

**Propulsion:** tables `thrust_vs_time`, `mass_vs_time`, `cg_vs_time`, `moipitch_vs_time`, `moiroll_vs_time` vs **`msl_time`**. `thrust=tsl+(101325-press)*aexit`. `mprop=1` if `msl_time<=60` else 0. Skip `mfreeze` if absent.

**Forces:** `FAPB`/`FMB` from `pdynmc*refa` and `ca/cy/cn/cll/clm/cln` as C++. **`mtvc==0`:** add `thrust` to `FAPB[0]`. **`mtvc!=0`:** `ValueError` this slice (do not take the C++ `FPB`/`FMPB` branch). Always add `FARCS`/`FMRCS` (zeros if RCS off / names absent). Do not write `FSPB` (newton-owned).

**Actuator:** cross-fin mix `delcx1=-dpcx-drcx` etc as C++. **`mact<2`** position limit only (**includes `mact==1`**); **`mact==2`** second-order with rate limit, CADAC sign; else `ValueError`. Convert back `dpx`/`dqx`/`drx`.

**Control:** `maut==0` return. Always `control_roll` if `maut>=1`. `2` → `control_rate`; `3` → `control_accel`. `ancomx+=ancomx_test`, `alcomx+=alcomx_test`. Skip `factwacl`/`twcl` if those names are absent (C++ reads undeclared → 0). `control_accel` pole placement as C++ (`zacl=0.7*(1+zacl_bias)`, `wacl=|realq1|*(1+wacl_bias)`, `pacl=(|realq2|+35)*(1+pacl_bias)`).

**INS:** `mins==0` copy truth `TBL→TBLC`, `FSPB→FSPCB`, `WBEB→WBECB`, `SBEL→SBELC`, `VBEL→VBELC`, `dvbe→dvbec`, then Euler/flight-path from computed as C++ after the if/else. `mins==1` port `init_ins` / `ins` / `ins_gyro` / `ins_accl` / `ins_alt` with every `gauss`/`uniform` draw **0** (Monte Carlo out of scope). Cholesky `XX_INIT` is then 0 → perfect alignment plus zero instrument errors; execute still runs the error ODE (stays 0). Control reads `WBECB`/`FSPCB`/`thtblcx`/`phiblcx`. Autopilot uses `mins=1` under this zero-draw policy.

**Guidance:** decode `guid_mid=mguide//10`, `guid_term=mguide%10`. Mid 2: IP from radar `SIEL{k}` by name, `guidance_line`. Term 6: `guidance_term_comp`. Term 7: `guidance_term_pronav`. Then circular limiter vs `gmax` as C++ (runs even when ACBX is 0). Radar/target from `Packet.type` + names, not `"f1"`.

**Sensor:** `mseek==0` return. `mtarget==1` rocket packets `ROCKET5`; `==2` aircraft `AIRCRAFT3`; else `ValueError`. Pair k-th missile to k-th target of that type. `skr_dyn==1` dynamics with bias/random **0**. Glint/GAUSS/MARKOV terms 0. Port RF (`skr_type==1`) and IR (`skr_type==2`) mode machine as C++ (`mseek=10*type+mode`). Write `STEL`/`VTEL`/`tgt_slot`/`dta`/`SBTL`.

**Intercept:** no `sys.exit` / `cout`. `stop&&trcond` → `vehicle.health=0`, `ctx.combus[slot].status=0`. Ground `alt<=0` (read `alt` or `hbe`) same with `write` latch. IP `ip_sltrange<500` closing-speed sign change as C++. Sensor lock `skr_mode==4` closest-approach interpolation `mterm` 0/1 as C++; declare both missile and target dead. `mterm==2` (MC angles) → `ValueError`.

**TVC / RCS:** define C++ fields. `mtvc==0` / both RCS flags 0 leave `FPB`/`FMPB`/`FMRCS`/`FARCS` zero. Else `ValueError`.

## Aircraft (AIRCRAFT3)

Modules: environment, kinematics, guidance, control, forces, newton (SAM6 Flat3). No aero/prop/actuator/sensor/intercept (C++ dummies). Vehicle class wiring belongs with the factory / JSONC task — module classes are separate deliverables.

**Guidance:** `acft_option==0` `ACOML=(0,0,-grav)` (RF #1 / smoke). `1` horizontal g-turn inside `[man_start,man_stop)`. `2` escape vs first `MISSILE6` using `SBEL`/`VBEL` by name (C++ used `data[3]`/`data[4]`). Else `ValueError`. Outside the maneuver window, gravity bias as C++ `else`.

**Control:** bank from `TVL@ACOML`, lag `tphi` (0 = no lag), limit `philimx`, write `phiavout`. Load factor lag `tanx`; if `acft_option>0` alpha limiter `pdynmc*clalpha*alplimx/wingloading`.

**Forces:** `FSPA=(acc_longx*grav, 0, -anx*grav)`.

Smoke: 0.1 s straight-and-level from RF #1 ICs (`sael2=-30e3`, `sael3=-10e3`, `dvae=250`, `psivlx=90`, `acft_option=0`); `alt` near 10000; health 1.

## Rocket (ROCKET5)

Modules: environment, kinematics, aerodynamics, propulsion, sensor, guidance, control, forces, newton, intercept. Vehicle class wiring belongs with the factory / JSONC task.

**Aero:** `cltgt_vs_alpha_mach`, `cdtgt_vs_alpha_mach`; body `catgt`/`cntgt`/`cytgt`; `mprop==0` → `catgt*=1.1`; `cnalp=7.468` at init. `area` default 0.636.

**Propulsion:** analytic SRBM1: `mprop==1` mass flow `thrust_sl/(isp*9.81)`, `mass=mass_launch-mass_flow*launch_time`, `thrust=thrust_sl+(pres_sl-press)*aexit`, burnout `mass<=mass_launch-mass_fuel` → `mprop=0`. `mprop==0` leave mass. Else `ValueError`. Defaults from C++ def (`mass_launch=6000`, `mass_fuel=4000`, `isp=230`, `thrust_sl=128600`, `aexit=0.282`).

**Sensor:** `mseek==0` no track (ballistic). `mseek!=0` and `flag_exo` and `alt<alt_endo` kinematic LOS to `stel1/2/3` as C++. Else if `mseek` not in `{0}` and conditions off, C++ does nothing — match that.

**Guidance:** `mguide==0` writes limiter of zeros. `guid_mode==1` pronav; `guid_manvr==1` spiral as C++. Else if decoded fields not 0/1 → `ValueError`.

**Control:** `maut==0` ballistic (no accel loop). `maut==1` and `alt<alt_endo` P-I as C++ (`tr`, `gacp`, `ta=2.2`). Exo sets `flag_exo` and zeros states. Ascent applies `ancomx_bias` while `not flag_exo`. Else `ValueError`.

**Forces:** `FSPA` from thrust − axial aero / mass as C++. Ignore undeclared `acc_longx` if absent (0).

**Intercept:** no `sys.exit`. `dta<1000` and `mguide>0` and `dvta>0` → health 0. `alt<0` → health 0. `write` latch.

Smoke: 0.1 s from `input_SRBM_Ballistic.asc` ICs (`sael1=1000`, `sael2=-250e3`, `thtvlx=85`, `mprop=1`, `maut=1`); `alt` near 0; health 1.

## Radar (RADAR0)

Modules: kinematics, newton, sensor (Flat0). No aero/prop/forces/control.

**Sensor:** `mtrack==0` return (zeros already defined: com `lnch_delay_m*=0`, save `launch_delay*=9999`). `mtrack==2` aircraft: at `track_step` epochs, polar from `SAEL-SREL` for each `AIRCRAFT3`; **sigma errors 0** (no gauss); lethal range sets save `launch_delay{k}=sim_time` then com `lnch_delay_m{k}=launch_delay{k}+lnch_dly_bias{k}` and `SIEL*=STCEL` as C++ (pair by type-index). `mtrack==1` rocket: apogee when measured `VTCEL[2]` becomes `>0`; look_up `rocket_traj` / `missile_traj` table names as C++ (`apotime_vs_descent_altitude`, `time_vs_ascent_altitude`, `x_vs_launch_time`, `y_vs_launch_time`, `z_vs_launch_time`, `alt_vs_launch_time`); then com `lnch_delay_m{k}=launch_delay+lnch_dly_bias{k}`; sigma 0. `mtrack` not in `{0,1,2}` → `ValueError`. Up to three targets of the tracked type; ignore the rest.

Smoke: radar + one aircraft, `mtrack=2`, `lethal_rng=20e3`, 0.1 s; `SREL` finite; health 1. No traj decks.

## Testing

Unit: `rtol=1e-12`, `atol=1e-14` vs CADAC formulas. TDD each task.

E2E autopilot: `Python/tests/e2e/test_sam6_autopilot.py`. Skip if `Python/tests/e2e/goldens/sam6/plot.csv` absent (do not create the file). Else `run_scenario` on translated case; compare missile plot columns present in both; skip sentinel `time=-1`; CSV `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`.

That golden **must** be a **zero-MC** trajectory matching Python `mins=1` with `gauss`/`uniform` draws 0. Do **not** check in a raw CADAC `plot.csv` from a Monte-Carlo-on INS run — instrument noise would not match this slice.

Full 7-vehicle RF (`input.asc` / `input_SAM_RF_AC_Radar_#1_#2_#3.asc`) is e2e-optional: a skip-without-golden test may exist; **do not require that golden**; do not fail CI for its absence.

Regression: HYPER3 e2e and FALCON5/FALCON6/HYPER5/HYPER6 unit tests on commits that touch kernel/cli/translate.

## Process

Grok implementer `cursor-grok-4.6-high` per task; Grok task reviewer; Grok whole-plan review. No Composer Fast / Kimi / Fast. Controller does not patch physics. Isolated worktree; no commit to main without user ask.

## Python style

Named state, numpy `@`, CADAC numeric order copied not “improved.” Same module **name** binds Sam6Missile vs Sam6Aircraft vs Sam6Rocket vs Sam6Radar vs Plane6 by vehicle type. `spec.family` set → only `_VEHICLE_FAMILIES`.
