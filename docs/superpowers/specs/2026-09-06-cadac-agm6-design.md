# CADAC AGM6 (MISSILE6 + TARGET3 + AIRCRAFT3) — design

Date: 2026-09-06
Status: approved (parent dispatch). Depends on kernel + Flat6 (FALCON6) + family lookup (this plan Task 1 if missing).

Parent: `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
Siblings: HYPER5 / HYPER6 design specs (do not edit).

## Goal

Run CADAC AGM6 from JSONC. Types `MISSILE6`, `TARGET3`, `AIRCRAFT3` bind only as family `agm6`. First e2e: `input_3_1 AGM6 Free Flight.asc` (1× `MISSILE6`, `alpha0x=3`, no guidance/sensor). Numerics regression-close to `CADAC_Simulations/AGM6_250217/AGM6/`. Same plan still ports `TARGET3` + `AIRCRAFT3` and missile datalink/sensor/guidance/control/actuator/ins/intercept so `input_2_1 AGM6 Test Case.asc` / `input.asc` can be translated as a second case.

## Non-goals

Monte Carlo (`MONTE`, sampled `GAUSS`/`MARKOV`/`RAYL`). CADAC packet-slot indexing and ids `m1`/`a1`/`t1`. `sys.exit` / `cout` intercept. SAM6 / SRAAM6 / Plane6 class reuse. Editing FALCON6 `Flat6Environment` (`mwind==0` must keep working). Engagement-fan / lock-on-after-launch / IIR-only demos beyond the test-case modes listed below.

## Layout

```
Python/src/cadac/vehicles/flat6/agm6/
  vehicle.py      # Agm6Missile type="MISSILE6"
  aero.py
  propulsion.py
  forces.py
  actuator.py
  control.py
  ins.py
  guidance.py
  datalink.py
  sensor.py
  intercept.py
  environment.py  # Agm6Environment (mair + weather); not cadac.eom.flat6
  flat3io.py      # SAEL/FSPA ↔ Flat3 SBEL/FSPV bridge
  target.py       # Agm6Target type="TARGET3"
  aircraft.py     # Agm6Aircraft type="AIRCRAFT3"
Python/cases/agm6/            # free-flight JSONC + AGM6_aero_deck; test-case later
Python/tests/unit/test_agm6_*.py
Python/tests/e2e/test_agm6_freeflight.py
```

Reuse `cadac.eom.flat6` (`Flat6Kinematics`, `Flat6Euler`, `Flat6Newton`) and `cadac.eom.flat3` (via `flat3io` wrappers). Do not import Plane6 / SAM6 / SRAAM6 / Hyper5 `Target3`.

## Family registry

`family` set → resolve **only** `_VEHICLE_FAMILIES[(family, type)]`. `family` absent/`None` → **only** `_VEHICLE_TYPES[type]` (today’s HYPER5 `TARGET3`, `PLANE6`, …).

Register (this plan; not in `_VEHICLE_TYPES`):

| family | type | Class | Decks |
|---|---|---|---|
| `agm6` | `MISSILE6` | `Agm6Missile` | aero required; prop none (analytic); weather optional |
| `agm6` | `TARGET3` | `Agm6Target` | none |
| `agm6` | `AIRCRAFT3` | `Agm6Aircraft` | none |

Unknown `(family, type)` → `ValueError`. Do not retarget existing unknown-type tests (`AIM5`). HYPER5 `TARGET3` stays global. Translate writes `"family": "agm6"`. If the family API is missing, Task 1 adds it idempotently: keep any keys already in `_VEHICLE_FAMILIES`; do not replace the dict.

**Canonical family:** `VehicleSpec.family`. Scenario JSONC `"family"` is the default for every vehicle. A per-vehicle `"family"` wins when present. `_resolve_vehicle` / `_build_vehicle` read `spec.family` only (not a separate scenario argument after load).

## JSONC / translate

- `RunConfig.family: str | None` (scenario default). `VehicleSpec.family: str | None` (canonical; scenario default, vehicle wins). `VehicleSpec.weather_deck: Path | None`.
- `translate_scenario_asc(src, dst_dir, family=None)` writes scenario `"family"` when given. Vehicles omit `"family"` unless a later translator writes a per-vehicle override.
- `WEATHER_DECK name.asc` → `weather_deck: "name.jsonc"`.
- Stochastic prefixes (Monte Carlo out of scope): `GAUSS name mean sigma` / `MARKOV name mean beta` / `RAYL name value` store the **mean/value only** (`MARKOV randal 2 5` → `randal=2`). Do not sample. Shared `translate.py` **must keep means** — do not adopt a SRAAM6-style skip of these prefixes.
- `MONTE` lines ignored (already skipped).
- Vehicle name = first token after type (`Blue`, `Ground`).

## MISSILE6 (Flat6)

CADAC MODULES (free flight): `environment`, `kinematics`, `aerodynamics`, `propulsion`, `forces`, `euler`, `newton`.

Test-case MODULES add: `ins`, `datalink`, `sensor`, `guidance`, `control`, `actuator`, `intercept`. Put **all** missile modules on the vehicle so `define` always runs; `run_loop` skips names absent from the scenario MODULES list.

Constructor: `Agm6Missile(name, aero_deck, events=None, weather_deck=None)`. Skip-if-exists on name collisions (Plane6 pattern). `com_names` = store fields with `"com"` in outputs.

**Environment:** Do not change `cadac.eom.flat6.Flat6Environment` (FALCON6 `mwind!=0` → `ValueError`). AGM6 C++ uses `mair=|matmo|mturb|mwind|` and `WEATHER_DECK`. Vehicle-local `Agm6Environment` (`name="environment"`):

- `mair==0` (free flight default): US76 + `gravity(hbe)` + `VAEL=0` + `VBAL=VBEL` (equivalently `VBAL=VBEL-VAEL` with `VAEL=0`) + `dvba=||VBAL||` as C++ all-zero path. `initialize` sets `dvba=dvbe`. **Must write `VBAL`:** reused `Flat6Kinematics.execute` reads `VBAL`. Do not patch `cadac.eom.flat6`.
- Other `mair` → `ValueError` until the weather task. Then port C++: `matmo` 0 US76 / 2 weather tables (`density`,`pressure`,`temperature` vs `hbe`); `mwind` 0 / 1 constant (`dvae`,`psiwdx`) / 2 tables (`speed`,`direction`); `mturb` 0 / 1 Dryden (`environment_dryden`); always `VBAL=VBEL-VAEL`. Unknown triples → `ValueError`. Skip `mfreeze` latch if `mfreeze` absent. Dryden `rand()` is not CADAC-reproducible: unit-test the lag algebra with a frozen `gauss_value`; do not require C++ `rand` match.

**Kinematics / Newton:** reuse Flat6. `time` lives on Flat6 newton (FALCON6). If `launch_time` / `halt` / `stop` absent, guidance/intercept use `ctx.sim_time` / treat flags as 0. Do not add those fields to `cadac.eom.flat6`.

**Euler:** reuse `Flat6Euler`. AGM6 C++ uses `ai11`/`ai33` (same as `IBBB=diag(ai11,ai33,ai33)`, `eng_ang_mom=0`). Propulsion writes those two names plus `IBBB` and `eng_ang_mom=0`. Do not change `Flat6Euler`.

**Aerodynamics:** port `aerodynamics.cpp` + `aerodynamics_der`. Tables from `AGM6_aero_deck.asc`. **1D vs Mach:** `ca0_vs_mach`, `caa_vs_mach`, `cad_vs_mach`, `cndq_vs_mach`, `clmdq_vs_mach`, `clmq_vs_mach`, `cllap_vs_mach`, `clldp_vs_mach`, `cllp_vs_mach` (C++/deck are 1D `*_vs_mach`, **not** 2D `*_vs_mach_alpha`). **2D vs Mach and alpha:** `cn0_vs_mach_alpha`, `cnp_vs_mach_alpha`, `clm0_vs_mach_alpha`, `clmp_vs_mach_alpha`, `cyp_vs_mach_alpha`, `clnp_vs_mach_alpha`. Init: `refl=0.5`, `refa=0.196`, termination `trmach=0.4`, `trdynm=10e3`, `trload=0.5`, `tralp=1`, `trcond=0`. If `dpx`/`dqx`/`drx`/`alimit` absent, 0 (free flight). CADAC sign local; not `np.sign`; not `flat6._cadac_sign`.

**Propulsion:** analytic (no prop deck). `mprop` 0 off (`thrust=0`); 1 on: `fmassed=thrsl*throtl/(spi*9.81)` (C++ `9.81`, not `AGRAV`), stored-slope `integrate` of `fmasse`, `vmass=vmass0-fmasse`, `thrust=thrsl*throtl+(101325-press)*aexit`. `fmasse>=fmass0` → `mprop=0`. Else `ValueError`. `initialize`: `vmass=vmass0`. Execute also writes `IBBB` and `eng_ang_mom=0`. Skip `mfreeze` latch if `mfreeze` absent.

**Forces:** `FAPB=[-pdynmc*refa*ca+thrust, pdynmc*refa*cy, -pdynmc*refa*cn]`; `FMB=pdynmc*refa*refl*[cll,clm,cln]`. Do not write `FSPB`.

**Actuator:** `mact<2` position-limit only (C++: 0 and 1); `mact==2` `actuator_scnd` four-fin. Else `ValueError`. Mix `delcx*` from `dpcx`/`dqcx`/`drcx` as C++. `dt=ctx.int_step`. Local CADAC sign.

**Control:** `maut==0` return without writing. `maut in {1,2,3}`: always `control_roll`; `2` also `control_rate`; `3` also `control_accel`. Else `ValueError`. Reads INS `WBECB`/`phiblcx`/`FSPCB`. `SMALL=1e-7` module-level. Limit `|dpcx|`/`|dqcx|`/`|drcx|` as C++.

**INS:** `mins==0` ideal: copy `TBL→TBLC`, `FSPB→FSPCB`, `WBEB→WBECB`, `SBEL→SBELC`, `VBEL→VBELC`, `dvbe→dvbec`, `phiblx→phiblcx`, then common flight-path / Euler-from-`TBLC` as C++. `mins==1` Widnall error ODEs; C++ `gauss()` at define → **zeros** (no MC). JSONC `GAUSS`/`MARKOV` means stored as means. Else `ValueError`. Skip gyro/accl helpers when `mins==0`. No Plane6 import.

**Datalink:** packets by `Packet.type=="AIRCRAFT3"` (first if several), not id `a1`. `tgt_num` is 1-based index among `type=="TARGET3"` packets in combus order. Copy `STCEL{tgt_num}`/`VTCEL{tgt_num}`/`SAEL`/`VAEL` from that aircraft packet by **name**. If `|STCEL|-tgt_pos|>EPS`, `mnav=3`. No C++ slot arithmetic.

**Guidance:** `mguid=|mid|term|`. C++ order: if `mnav==3`, latch `epchta` / `STELM=STCEL` / `VTELC=VTCEL` **before** any `mguid==0` return. Then `mguid==0` return **without** writing `ancomx`/`alcomx`. In-scope after latch: `30` mid pro-nav vs datalink (`guidance_mid_pronav`); `6` terminal compensated (`guidance_term_comp`); `40` mid-4 true-target (`STEL-SBELC`). Else `ValueError` (no line `2x`, no term-5). Circular limiter vs `gmax`. Do **not** port C++ `missile[405].gets(mnav)` (clobbers `grav_bias`); `mnav` stays datalink-owned. `launch_time` absent → `ctx.sim_time`.

**Sensor:** `mseek` 0 off (still download target into `STEL`/`VTEL`); 2 enable / 3 acquire / 4 lock / 5 blind hold. C++ has **no `mseek==5` branch** (outputs held from the last lock-on cycle). `mseek==5` must **not** `ValueError`. Else including 1 → `ValueError`. Target packet: `type=="TARGET3"` selected by `tgt_num` (1-based TARGET3 order). Kinematics from `SAEL`/`VAEL` (bridge aliases). `skr_dyn==0` `sensor_ir_kin`; `==1` `sensor_ir_dyn` + helpers. Else `ValueError`. Define undeclared C++ slots used in execute (`timeac`, `dbtk`). Skip `fovlimx` / IRS names if absent. Gauss/Markov sensor errors: stored means; unit tests use 0.

**Intercept:** no `sys.exit` / `cout`. `vehicle.health=0` and `ctx.combus[slot].status=0` on halt, `trcond and stop`, ground `alt=-SBEL[2]<=0` (`write` latch), and target-plane closest approach when mid=4 or term in {5,6} as C++ (`dbt<100`, `sbtp3>0`). Packets by type + `tgt_num`. Skip halt/stop if absent.

**missile_functions.cpp:** executive I/O / C-index packets already in the kernel. No `missile_functions.py`. `com_names` from `"com"` flags.

## TARGET3 (Flat3, family-only)

Modules: Flat3 environment / kinematics / newton (via `flat3io`) + `Agm6TargetForces`. Constructor `(name, events=None)`.

C++ names `sael1`/`sael2`/`sael3`/`dvae`/`SAEL`/`VAEL`/`FSPA`/`TAL` vs library Flat3 `sbel*`/`dvbe`/`SBEL`/`VBEL`/`FSPV`/`TBL`. Algebra is the same. `flat3io.copy_in` / `copy_out` around delegated `Flat3Environment` / `Flat3Newton`. Do not modify `cadac.eom.flat3`.

**Forces:** `FSPA=[acc_longx*grav, acc_latx*grav, -grav]`; also write `FSPV=FSPA` for the delegate. Demo/test-case `acc_latx=0.01`, `acc_longx` default 0.

**com_names:** at least `time`, `SAEL`, `VAEL`, `psivlx`, `thtvlx`, `alt` (C++ `com` flags).

## AIRCRAFT3 (Flat3, family-only)

Modules: Flat3 env/kin/newton (same bridge) + guidance + control + forces + sensor. Constructor `(name, events=None)`. Collides AIM5/SAM6 — family-only.

**Guidance:** `acft_option` 0 gravity bias `ACOML=[0,0,-grav]`; 1 horizontal g-turn; 2 escape vs first `TARGET3` packet (names `SAEL`/`VAEL` on that packet). Else `ValueError`. Test case is 0.

**Control:** bank and load-factor lags as C++; writes `phiavout` for newton. Local CADAC sign.

**Forces:** `FSPA=[acc_longx*grav, 0, -anx*grav]`; `FSPV=FSPA`.

**Sensor:** every `track_step`, for each `TARGET3` packet (max 5, combus order): polar geometry, optional noise (means; tests use sigma 0), write `STCEL1`/`VTCEL1` … `STCEL5`/`VTCEL5` (`com` on 1–3 as C++). Identify targets by type, not `t1`.

## First case (required e2e)

Translate `CADAC_Simulations/AGM6_250217/AGM6/input_3_1 AGM6 Free Flight.asc` + `AGM6_aero_deck.asc` → `Python/cases/agm6/input_freeflight.jsonc`. `family="agm6"`. `end_time` 30 from file. 1× `MISSILE6`, `alpha0x=3`, `mprop=1`, `sbel3=-7000`, `dvbe=293`. No `mair` / weather. Smoke: `0.1 s`, `hbe` near 7000.

## Second case (translate + smoke; golden optional)

Translate `input_2_1 AGM6 Test Case.asc` (same as `input.asc`) + aero + `weather_deck.asc`. Vehicles: `MISSILE6`, `TARGET3`, `AIRCRAFT3`. Modes: `mair=212`, `mins=1` (zero gauss), `mseek` 0 then events `mnav=3`/`mprop=1`/`mseek=2`/`mguid=30`, then `mseek=4` → `mguid=6`, `maut=3`, `mact=2`. JSONC smoke (`end_time` 0.05, three vehicles `health==1`) **or** e2e skip without `tests/e2e/goldens/agm6/test_case_plot.csv`. Do not require goldens. Unit-level datalink/sensor tests are the correctness proof if the second e2e is skipped.

## Testing

Unit: `rtol=1e-12`, `atol=1e-14` vs CADAC formulas. TDD each task. Tests `test_agm6_*.py`.

E2E: `tests/e2e/test_agm6_freeflight.py`. Skip if `tests/e2e/goldens/agm6/plot.csv` absent (do not create it). Else plot-flagged columns present in both (`hbe`/`vmach` if both present); sentinel `time=-1`; CSV `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`. Pattern HYPER6 e2e.

Regression: FALCON6 units (`mwind==0`), HYPER5 `TARGET3` without family, HYPER3 e2e if kernel/cli touched.

## Process

Grok implementer `cursor-grok-4.6-high` per task; Grok reviewer; Grok whole-plan review. No Fast/Kimi. Controller does not patch physics. Isolated worktree; no commit to main without user ask.

## Python style

Named state; numpy `@`; CADAC numeric order copied, not “improved.” Same module **name** binds Agm6 vs Plane6 vs Hyper5 by vehicle type + family. Packets by `Packet.type` and named `vars`.
