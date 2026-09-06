# Updates

## 0.115.0 - SAM6 guidance line and pronav
- Added `cadac.vehicles.sam6.guidance.Sam6Guidance` (`name="guidance"`). Port of SAM6 `guidance.cpp` `guidance_line` / `guidance_term_comp` / `guidance_term_pronav`. Does not define INS/newton/aero/sensor names (`gmax`, `TBLC`/`VBELC`/`SBELC`/`thtvlcx`/`psivlcx`/`FSPCB`, `grav`, `SBEL`/`sbel1/2/3`, `STEL`/`VTEL`/`thtpb`/`psipb`/`sigdy`/`sigdz`, `ddab`/`psisb`/`thtsb`/`lamdqb`/`lamdrb`). No `guidance_mid_pronav`.
- `execute`: `guid_mid=mguide//10`, `guid_term=mguide%10`. `guid_mid==2` IP from `RADAR0` `SIEL{k+1}` minus `SBELC` (k-th `MISSILE6` among that type; `Packet.type`/names, not `"f1"`). `guid_term==6`/`7` as C++. `guid_mid==3` `ValueError`. Always circular limiter vs `gmax` (mguide=0 still writes `ancomx`/`alcomx` from zero ACBX). Protocol `vehicle.store`. No vehicle. No Flat6/Plane imports.
- Tests: `Python/tests/unit/test_sam6_guidance.py` (mguide=0 commands 0; mguide=20 radar `SIEL1` 1 km north finite vs C++ rtol=1e-12; mguide=30 raises; mguide=7 seeker kinematics finite; type-index not slot; limiter caps at `gmax`).

## 0.114.0 - SAM6 INS mins 0 and 1
- Added `cadac.vehicles.sam6.ins.Sam6Ins` (`name="ins"`). Port of SAM6 `ins.cpp` `init_ins` / `ins` / `ins_gyro` / `ins_accl` / `ins_alt`. Does not define kinematics/newton/euler truth names (`TBL`, `TLB`, `WBEB`, `SBEL`, `FSPB`, `VBEL`, `dvbe`, `alt`).
- `mins==0`: copy `TBL`→`TBLC`, `FSPB`→`FSPCB`, `WBEB`→`WBECB`, `SBEL`→`SBELC`, `VBEL`→`VBELC`, `dvbe`→`dvbec`, then Euler/FPA as C++. `mins==1`: error ODEs with every `gauss`/`uniform` draw 0 (Cholesky `XX_INIT` 0, `EWALKA=0`); autopilot case. Else including 2/3 `ValueError`. `ins_alt` `biasal=randal=0` → `hbem=alt`. Protocol `vehicle.store`. No vehicle. No Flat6/Plane/Hyper INS import.
- Tests: `Python/tests/unit/test_sam6_ins.py` (mins=0 copies SBEL/FSPB; mins=1 after init `SBELC==SBEL`, one execute `WBECB==WBEB` rtol=1e-12; mins=2 raises).

## 0.113.0 - SAM6 acceleration autopilot
- `Sam6Control.execute`: `maut==3` calls `control_roll` then `control_accel`. `maut==4` still `ValueError`; unknown modes still roll-only (Task 12).
- `control_accel` ports SAM6 `control.cpp`: `ancomx+=ancomx_test`, `alcomx+=alcomx_test`; circular limiter vs `alimitx`; poles `zacl=0.7*(1+zacl_bias)`, `wacl=|realq1|*(1+wacl_bias)`, `pacl=(|realq2|+35)*(1+pacl_bias)`; pitch/yaw gains as C++; stored-slope `integrate` of `zz`/`yy`; writes `dqcx`/`drcx`/`GAINFB` (yaw). Skip undeclared `factwacl`/`twcl`. Does not write `ancomx`/`alcomx`. Protocol `vehicle.store`. No Flat6/Plane imports.
- Tests: `Python/tests/unit/test_sam6_control_accel.py` (maut=3, `ancomx_test=1`, `realq1=-10` → `wacl==10` rtol=1e-12; `dqcx` finite vs C++; `zz`/`yy` stored-slope; maut=4 still raises).

## 0.112.0 - SAM6 control roll and rate
- Added `cadac.vehicles.sam6.control.Sam6Control` (`name="control"`). Port of SAM6 `control.cpp` `def_control` / `control_roll` / `control_rate`. Does not define INS/aero/newton names (`WBECB`, `thtblcx`, `phiblcx`, `dlp`/`dld`/`dna`/`dmd`, `dvbe`) or undeclared `factwacl`/`twcl`. No `control_accel` (Task 13).
- `execute`: `maut==0` return without writing; `maut==4` `ValueError`; else `control_roll`; `maut==2` also `control_rate`. Roll: `wrcl=-0.8*dlp*(1+factwrcl)`, pole-placement `gkp`/`gkphi`, `|thtblcx|>88` rate `kp`; writes `dpcx`. Rate: open-loop `zrate`/`aa`/`bb`, `|dmd|<SMALL` then `SMALL*sign` (CADAC; no `dld` guard), `dqcx=DEG*grate*qq` (not `qqcomx`); writes `dqcx`/`drcx`/`dqcx_rcs`/`drcx_rcs`. Module-level `SMALL=1e-7`. Local CADAC sign. Protocol `vehicle.store`. No vehicle. No Flat6/Plane imports.
- Tests: `Python/tests/unit/test_sam6_control_rate.py` (maut=0 `dpcx` stays 0; maut=2 dvbe=16 zetlagr=1.2 dummy aero `dqcx`/`drcx` finite vs C++ rtol=1e-12; maut=4 raises; maut=1 roll-only; INS `WBECB` not `WBEB`; CADAC sign 0 → +1).

## 0.111.0 - SAM6 actuator mact position-limit and second-order
- Added `cadac.vehicles.sam6.actuator.Sam6Actuator` (`name="actuator"`). Port of SAM6 `actuator.cpp` / `actuator_scnd`. Local CADAC sign (`<0 → -1` else `+1`). `define` registers C++ `def_actuator` (four-fin scalars, not 3-vec `DX`). Does not define control commands `dpcx`/`dqcx`/`drcx` or unused C++ local `time`.
- `execute`: cross-fin mix `delcx1=-dpcx-drcx`, `delcx2=-dpcx+dqcx`, `delcx3=-dpcx+drcx`, `delcx4=-dpcx-dqcx`. **`mact<2`** (includes **1**) position-limit only; **`mact==2`** second-order with rate/position limits and stored-slope `integrate`; else `ValueError`. Mix back `dpx=0.25*(-delx1-delx2-delx3-delx4)`, `dqx=0.5*(delx2-delx4)`, `drx=0.5*(-delx1+delx3)`. Protocol `vehicle.store`. No vehicle. No Flat6/Plane/Hyper6 import.
- Tests: `Python/tests/unit/test_sam6_actuator.py` (mact=0 dlimx=28 dpcx=10 dqcx=drcx=0 → delx all -10, `dpx==10`; mact=1 same path; mact=2 wnact=600 zetact=0.7 states 0 → `delx1` finite, first-step `ddx1==-1800`; mact=3 raises; CADAC sign 0 → +1).

## 0.110.0 - SAM6 missile forces FAPB/FMB
- Added `cadac.vehicles.sam6.forces.Sam6Forces` (`name="forces"`). Port of SAM6 `forces.cpp`. `define` registers C++ `def_forces` (`FAPB`/`FMB` vec out). Does not define aero/prop/tvc/rcs/newton names (`pdynmc`, `thrust`, `refa`/`refl`, `ca`/`cy`/`cn`/`cll`/`clm`/`cln`, `mtvc`, `FARCS`/`FMRCS`, `FSPB`).
- `execute`: `FAPB=[-pdynmc*refa*ca, pdynmc*refa*cy, -pdynmc*refa*cn]`; `FMB=pdynmc*refa*refl*[cll,clm,cln]`. `mtvc==0` (or absent) adds `thrust` to `FAPB[0]`. `mtvc!=0` → `ValueError` (no `FPB`/`FMPB`). Missing `FARCS`/`FMRCS` treated as 0; if present, add them. Does not write newton-owned `FSPB`. Protocol `vehicle.store`. No vehicle. No Flat6/Plane imports.
- Tests: `Python/tests/unit/test_sam6_forces.py` (pdynmc=156, refa=0.0491, ca=0.4, thrust=1000, mtvc=0 → `FAPB[0]==-156*0.0491*0.4+1000` rtol=1e-12; mtvc=1 raises; FSPB sentinel unchanged; absent RCS zeros).

## 0.109.0 - SAM6 kinematics msl_time and VBEB incidence
- Added `cadac.vehicles.sam6.kinematics.Sam6Kinematics` (`name="kinematics"`). Port of SAM6 `kinematics.cpp`; copies quaternion algebra locally (not `Flat6Kinematics`, does not edit `flat6.py`). Extra/exec fields: `time`, `launch_delay` default 99999, `launch_epoch`, `launch_time`, `msl_time`, `stop`, `lconv`, `int_step_new`, `out_step_fact`.
- Init: `launch_epoch=launch_delay`, quaternions from Euler, `TBL=mat3tr`. Exec: `time=ctx.sim_time`; `ctx.int_step=int_step_new`; quaternion stored-slope `integrate`; TBL; Euler; incidence from **VBEB** (`alphax`/`betax`/`alpp`/`phip`); `trortho`→`trcond=1`, `alpp>tralp`→`trcond=2` if those names exist. `msl_time` from RADAR0 com `lnch_delay_m{k+1}` for k-th `MISSILE6` among that type (`Packet.type`); no radar → `lnch_delay=0`; `msl_time=max(0, sim_time-lnch_delay)`. Local CADAC sign; `SMALL=1e-7`. Protocol `vehicle.store`.
- Tests: `Python/tests/unit/test_sam6_kinematics.py` (no radar t=5 → `msl_time==5`; radar `lnch_delay_m1=2` first MISSILE6 → 3; identity VBEB alphax/betax 0; VBEB pitch → alphax 10 rtol=1e-12; trortho → trcond 1; not a Flat6 subclass).

## 0.108.0 - SAM6 environment from environment.cpp
- Added `cadac.vehicles.sam6.environment.Sam6Environment` (`name="environment"`). Port of SAM6 `environment.cpp`, not `Flat6Environment`. `define` registers C++ `def_environment` (`press`/`rho`/`grav`/`tempk` out, `vsound` diag, `vmach`/`pdynmc` out+scrn/plot/com, `mfreeze_environ`/`pdynmcf`/`machf` save). Does not define `alt`/`dvbe`/`hbe`/`VBEL`, wind (`mwind`/`VAEL`/`VBAL`/`dvba`), or missile `mguide`/`trcond`/`trdynm`/`mfreeze`.
- `execute` reads newton `alt` (C++; newton keeps `hbe=alt`) and `dvbe`. US76 `atmosphere76(alt)`; gravity `G*EARTH_MASS/(REARTH+alt)**2`; `vmach=abs(dvbe/vsound)` (C++ `mach`); `pdynmc=0.5*rho*dvbe*dvbe`. No wind. Skip `mfreeze` latch if `mfreeze` absent. `guid_term==6` and `pdynmc<=trdynm` → `trcond=3` (`mguide%10`). Protocol `vehicle.store`. No vehicle. No Flat6/Plane5/Plane6 import. Does not edit `flat6.py`.
- Tests: `Python/tests/unit/test_sam6_environment.py` (Newton-init first dvbe=16, SBEL z=-1000 → alt=hbe=1000, VBEL finite; env vs `atmosphere76(1000)` rtol=1e-12; import does not load `Flat6Environment`; not a subclass; guid_term 6 tiny pdynmc sets trcond 3).

## 0.107.0 - SAM6 newton mass and alt
- Added `cadac.vehicles.sam6.newton.Sam6Newton` (`name="newton"`). Port of SAM6 `newton.cpp`, not `Flat6Newton`. `define` registers C++ `def_newton` plus `hbe` (`VBEBD`/`VBEB`/`SBELD`/`SBEL` state, `sbel1/2/3` data, `FSPB`/`VBEL`/`alt`/`SLEL` out, `dvbe` in/out, `alpha0x`/`beta0x` data, `hbe` out, FPA/`anx`/`ayx`/`ATB` diag, `mfreeze` saves). Does not define `mass`/`FAPB`/`TBL`/`WBEB`/`grav`/`mfreeze`/`vmass`.
- Init: `VBEB` from `alpha0x`/`beta0x`/`dvbe`; `VBEL=TBL.T@VBEB`; `SBEL`/`SLEL` from `sbel*`; `alt=hbe=-SBEL[2]`.
- `execute`: `FSPB=FAPB/mass`; `ATB=skew(WBEB)@VBEB`; stored-slope `integrate` of `VBEB` then `SBEL`; `VBEL=TBL.T@VBEB`; writes `alt` and `hbe` both `-SBEL[2]`. Skip `mfreeze` latch if `mfreeze` absent. Protocol `vehicle.store`. No vehicle. No Flat6/Plane5/Plane6 import.
- Tests: `Python/tests/unit/test_sam6_newton.py` (sbel=0, dvbe=16, identity TBL, mass=300, FAPB=0, grav=9.8, WBEB=0; init `alt==hbe==0`, `||VBEB||==16`, `VBEL==VBEB`; one execute `alt` finite; `FSPB=FAPB/mass` not `vmass`; rtol=1e-12).

## 0.106.0 - SAM6 missile euler
- Added `cadac.vehicles.sam6.euler.Sam6Euler` (`name="euler"`). Port of SAM6 `euler.cpp`, not `Flat6Euler`. `define` registers C++ `def_euler` (`ppd/pp`,`qqd/qq`,`rrd/rr` state; `ppx/qqx/rrx` out+plot; `WBEB` vec diag). Does not define `FMB`/`ai11`/`ai33`.
- No `init_euler` in C++; `initialize` is pass.
- `execute`: stored-slope `integrate` of `pp` then `qq` then `rr` with C++ sequential rates (`ppd_new=FMB[0]/ai11`; `qqd_new=((ai33-ai11)*pp*rr+FMB[1])/ai33` uses updated `pp`; `rrd_new=(-(ai33-ai11)*pp*qq+FMB[2])/ai33` uses updated `pp`/`qq`). Writes `WBEB=[pp,qq,rr]` and `ppx/qqx/rrx` in deg/s. Protocol `vehicle.store`. No vehicle. No Flat6/Plane5/Plane6 import.
- Tests: `Python/tests/unit/test_sam6_euler.py` (ai11=2.9, ai33=440, FMB=(1,0,0), pp=qq=rr=0, dt=0.001 → `pp==integrate(1/2.9,0,0,0.001)` rtol=1e-12; qq=rr=0).

## 0.105.0 - SAM6 missile propulsion
- Added `cadac.vehicles.sam6.propulsion.Sam6Propulsion` (`name="propulsion"`). Constructor takes `Datadeck`. `define` registers C++ `def_propulsion` (`mprop` out, `aexit` default 0.0314, `mass` 300, `thrust`, `xcgref`, `xcg` 2.9, `ai11` 2.9, `ai33` 440, `mfreeze` saves). Does not define `msl_time`/`press`/`mfreeze`.
- No `init_propulsion` in C++; `initialize` is pass.
- `execute` ports `Missile::propulsion`: `thrust=look_up("thrust_vs_time",msl_time)+(101325-press)*aexit`; `mass`/`xcg`/`ai33`/`ai11` from `mass_vs_time`/`cg_vs_time`/`moipitch_vs_time`/`moiroll_vs_time`; `mprop=1` if `msl_time<=60` else 0. Skip `mfreeze` latch if `mfreeze` absent. Tables from `SAM_prop_deck.asc`. Protocol `vehicle.store`. No vehicle. No Flat6/Plane5/Plane6 import.
- Tests: `Python/tests/unit/test_sam6_propulsion.py` (msl_time=0 press=101325 mass 300 mprop 1 sea-level table thrust; msl_time=61 mprop 0; back-pressure and tables vs look_up rtol=1e-12).

## 0.104.0 - SAM6 missile aerodynamics
- Added `cadac.vehicles.sam6.aero.Sam6Aero` (`name="aerodynamics"`). Constructor takes `Datadeck`. `define` registers C++ `def_aerodynamics` (`refl=0.25`, `refa=0.0491`, force/moment coeffs, dimensional der, `alplimx=40`, termination). Does not define `vmach`/`alphax`/`betax`/`dpx`.
- `initialize` ports `Missile::init_aerodynamics`: `trortho=1e-4`, `tralp=1.047`, `trdynm=1e4`, `trload=0.001`, `trcond=0`.
- `execute` ports `Missile::aerodynamics` then `aerodynamics_der`. Tables from `SAM_aero_deck.asc` (comma names). Read `vmach` as C++ `mach`. `mprop==0` adds `cab`. Skip TVC `gtvc`/`parm` if absent (C++ localizes them unused; treat 0). `SMALL=1e-7` module-level. No vehicle. No Flat6/Plane5/Plane6 import.
- Tests: `Python/tests/unit/test_sam6_aero.py` (mach=2, alphax=10, mass=300, xcg=xcgref, zero fins, mprop=1; `ca`/`cn` vs look_up+C++ sums rtol=1e-12; `dna` finite).

## 0.103.0 - SAM6 Flat3 EOM (SAEL names)
- Added `cadac.vehicles.sam6.flat3`: `Sam6Flat3Kinematics` / `Sam6Flat3Environment` / `Sam6Flat3Newton`. Port of SAM6 `flat3_modules.cpp`. Does not modify `cadac.eom.flat3`. Does not import `cadac.eom.flat6`.
- Kinematics: `time` (exec, com), `launch_delay` (data), `launch_epoch` (out, com), `launch_time` (diag). Init `time=sim_time`, `launch_epoch=launch_delay`. Exec `launch_time=sim_time-launch_epoch`, `time=sim_time`.
- Environment: US76 `atmosphere76(-SAEL[2])` + `gravity` (not NASA helper). `mach`/`pdynmc` from `dvae`. Store name `mach`, not `vmach`. Does not define `alt`/`SAEL`/`dvae`.
- Newton: ICs `sael1/2/3` → `SAEL`, `cart_from_pol` → `VAEL`. `NEXT_ACC = TAL.T @ FSPA + (0,0,grav)`. `phiavout` 0 if absent. Writes `alt=-SAEL[2]`.
- Tests: `Python/tests/unit/test_sam6_flat3.py`.

## 0.102.0 - CADAC Flat0 kinematics and fixed-site newton
- Added `cadac.eom.flat0.Flat0Kinematics` (`name="kinematics"`). `define` registers C++ `def_kinematics` (`time` out+com, `launch_delay` data, `launch_epoch` init, `launch_time` out). `initialize`: `time=ctx.sim_time`, `launch_epoch=launch_delay`. `execute`: `launch_time=sim_time-launch_epoch`, `time=sim_time`.
- Added `cadac.eom.flat0.Flat0Newton` (`name="newton"`). `define` `srel1/2/3` data, `SREL` vec out. `initialize` packs `SREL`. `execute` no-op (fixed site). No radar/vehicle. No `flat6` import.
- Tests: `Python/tests/unit/test_flat0_kinematics.py`, `Python/tests/unit/test_flat0_newton.py`.

## 0.101.0 - CADAC family vehicle registry
- `VehicleSpec.family` is the source of truth (`None` when omitted). Scenario-level `"family"` copies onto vehicles that omit it; vehicle key wins. No `RunConfig.family`. Optional `sam_deck` / `srmb_deck`.
- `_VEHICLE_FAMILIES` + `register_family_type`: same class twice is a no-op; different class for an occupied pair is `ValueError`. Never writes `_VEHICLE_TYPES`. Map created empty (no SAM6 pairs).
- `_build_vehicle(path, spec)`: family set → `_VEHICLE_FAMILIES[(family, type)]` only (missing mentions family and type); else `_VEHICLE_TYPES`.
- `translate_scenario_asc(..., family=None)` stamps `"family"` on each vehicle when given; `SAM_DECK`/`SRBM_DECK` → jsonc decks; bare `ENDIF` skipped.
- Tests: `Python/tests/unit/test_family_registry.py`. Unknown-type sentinel still `"AIM5"`.

## 0.100.1 - HYPER6 e2e skip without golden
- Added `Python/tests/e2e/test_hyper6_climb.py`. Skip if `tests/e2e/goldens/hyper6/plot.csv` is absent (file not created). Else compare plot-flagged `alt`/`vmach` when both present; sentinel `time=-1`; CSV `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`. Live climb `run_scenario` calls `require_golden` first.

## 0.100.0 - run HYPER6 from JSONC climb
- Added `cadac.vehicles.hyper6.vehicle.Hyper6` (`type="HYPER6"`, health=1). Constructor `(name, aero_deck, prop_deck, events=None)`. Modules in climb ASC order: kinematics, environment, aerodynamics, propulsion, ins, guidance, control, actuator, forces, newton, euler (Round6 EOM + Hyper6 modules). Skip-if-exists on name collisions (Plane6 pattern).
- `run_scenario` maps `HYPER6 -> Hyper6`; both decks required (PLANE6 path, not HYPER5). Plot columns: flagged like PLANE6; CRUISE3 path unchanged. Radar/Satellite/Ground0 not registered. Unknown-type sentinel is `AIM5`.
- Translated climb case `Python/cases/hyper6/` from `input_climb.asc` + GHAME decks. Event `time>10` → `thtvdcomx=10`. `end_time` 60.
- `Round6Environment.initialize` copies `dvba=dvbe` (C++ `init_environment`) so first kinematics exec has nonzero airspeed.
- Tests: `Python/tests/unit/test_hyper6_one_step.py` (0.1 s smoke, `alt` near 10000; JSONC type HYPER6; AIM5 still raises). Retargeted unknown-type tests in plane5, cruise3, hyper5 from `"HYPER6"` to `"AIM5"`.

## 0.99.0 - HYPER6 ideal INS
- Added `cadac.vehicles.hyper6.ins.Hyper6Ins` (`name="ins"`). `define` registers C++ `def_ins` (mins/frax, computed `SBIIC`/`VBIIC`/`TBIC`/`WBICB`/`WBICI`/`FSPCB`, lon/lat/alt/Euler/flight-path `*c` names, gyro/accel error data zeros not CADAC gauss, states `RICI`/`ESBI` plot). Does not define kinematics/newton/euler truth names (`TBI`, `FSPB`, `SBII`, `VBII`, `WBIB`, `WBII`, `time`). No GPS/star fields.
- `initialize` no-op for `mins==0`. `execute` copies `TBI`→`TBIC`, `FSPB`→`FSPCB`, `SBII`→`SBIIC`, `VBII`→`VBIIC`, `WBIB`→`WBICB`, `WBII`→`WBICI`, then C++ common geographic/incidence/Euler path (`cad_geo84_in`/`cad_tdi84`). `mins!=0` → `ValueError`. Skip GPS/star if absent. No gyro/accl/grav helpers. Local CADAC sign. No vehicle. No Plane6 import.
- Tests: `Python/tests/unit/test_hyper6_ins_ideal.py` (mins=0 copies SBII→SBIIC rtol=1e-12; mins=1 raises; control `*c` names; GPS/star absent).

## 0.98.0 - HYPER6 guidance stub
- Added `cadac.vehicles.hyper6.guidance.Hyper6Guidance` (`name="guidance"`). `define` registers C++ `def_guidance` (`mguide` int data, line/pronav/LTG/glideslope slots including `wp_sltrange`/`wp_grdrange` default 999999). Does not define newton/control/INS names those functions would read (`time`, `grav`, `maut`, `alcomx`/`ancomx`/`phicomx`, `TBIC`, `mprop`). `initialize`/`terminate` pass (no C++ `init_guidance`).
- `execute`: `mguide==0` return without writing (climb default; C++ zeros locals then returns before `gets`). Else including 5 ValueError (no LTG/line/pronav). Protocol `vehicle.store`. No vehicle. No Plane6/Hyper5 guidance import.
- Tests: `Python/tests/unit/test_hyper6_guidance_noop.py` (mguide=0 no raise/no write; mguide=5 raises; define-only C++ fields).

## 0.97.0 - HYPER6 forces FAPB/FMB
- Added `cadac.vehicles.hyper6.forces.Hyper6Forces` (`name="forces"`). `define` registers C++ `def_forces` (`FAPB`/`FMB` vec out). Does not define aero/prop/newton names (`pdynmc`, `thrust`, `refa`/`refb`/`refc`, `cx`/`cy`/`cz`/`cll`/`clm`/`cln`, `FSPB`). `initialize`/`terminate` pass.
- `execute` ports `Hyper::forces`: `FAPB=[pdynmc*refa*cx+thrust, pdynmc*refa*cy, pdynmc*refa*cz]`; `FMB=[pdynmc*refa*refb*cll, pdynmc*refa*refc*clm, pdynmc*refa*refb*cln]`. If `FARCS`/`FMRCS` absent, treat as zero (RCS out of this plan); if present, add them. Does not write newton-owned `FSPB`. No Plane6 import.
- Tests: `Python/tests/unit/test_hyper6_forces.py` (GHAME refs, frozen aero/thrust vs C++ rtol=1e-12; FSPB sentinel unchanged; absent RCS zeros).

## 0.96.0 - HYPER6 maut dispatcher
- `Hyper6Control.execute` ports C++ `Hyper::control` decode `mauty=maut//10`, `mautp=maut%10`. Modes `{0, 24}` only (climb 24: yaw SAS + gamma + roll). Unknown including -1 → `ValueError`. Limit `|del*|` and `philimx` with CADAC sign. Writes `delacx`/`delecx`/`delrcx`/`ancomx`/`phicomx`. maut=0 returns without writing. No unused mauty/mautp branches. No Plane6 import.
- Tests: `Python/tests/unit/test_hyper6_maut.py` (maut=24 vs C++ rtol=1e-12; maut=0 no write; maut=-1 raises). Task 10/11 execute-is-pass tests now dispatch maut=24.

## 0.95.0 - HYPER6 gamma controller
- `Hyper6Control.control_gamma` ports C++ climb `mautp=4` pole-placement (`pgam`/`wgam`/`zgam`, `thtvdcomx`). Reads INS `qqcx`/`thtbdcx`/`thtvdcx`/`dvbec` (dvbec==0 → `dvbe`). `np.linalg.inv` for DP and (AA-BB*~GAINGAM). Returns `delecx`; writes `GAINGAM`/`gainff`; does not write `delecx`. No Plane6 import. `execute` still pass.
- Tests: `Python/tests/unit/test_hyper6_control_gamma.py` (climb pgam=4 wgam=2 zgam=0.7, thtvdcomx=0; finite elevator vs C++ rtol=1e-12; execute remains pass).

## 0.94.0 - HYPER6 roll and rate SAS
- Added `cadac.vehicles.hyper6.control.Hyper6Control` (`name="control"`). `define` registers C++ `def_control` (`maut`/`mroll`/`mfreeze`, `alimitx`, `dalimx`/`delimx`/`drlimx`/`philimx`, roll poles `wrcl`/`zrcl`, SAS `tp`/`zetlagr`, gamma poles `pgam`/`wgam`/`zgam`, INS heading/path `psivdcomx`/`thtvdcomx`, commands `delacx`/`delecx`/`delrcx` out+plot, states `yyd`/`yy`/`zzd`/`zz`). Does not define INS/aero names the helpers read (`phibdcx`/`ppcx`/`qqcx`/`rrcx`/`dvbec`, `dllp`/`dllda`/`dla`/`dlde`/`dma`/`dmq`/`dmde`/`dyb`/`dydr`/`dnb`/`dnr`/`dndr`). `execute` pass until maut dispatcher.
- Helpers port C++ `control_roll`/`control_roll_rate`/`control_pitch_rate`/`control_yaw_rate`: return commands; write `gkp`/`gkphi` (roll) and `zrate`/`grate`/`wnlagr` (yaw only). Reads INS `*c` names. Module-level `SMALL=1e-7` (not `cadac.constants`). Local CADAC sign (`<0 → -1` else `+1`). No Plane6/Flat6 import. No `control_gamma`.
- Tests: `Python/tests/unit/test_hyper6_control_roll.py` (climb `wrcl=8` `zrcl=0.9` `zetlagr=1.1`; frozen aero/rates vs C++ rtol=1e-12; execute remains pass).

## 0.93.0 - HYPER6 second-order actuators
- Added `cadac.vehicles.hyper6.actuator.Hyper6Actuator` (`name="actuator"`). `define` registers C++ `def_actuator` (`mact`, `dlimx`/`ddlimx`/`wnact`/`zetact`, `delax`/`delex`/`delrx` out+scrn/plot, elevon diags `elvlx`/`elvrx`/`elvlcx`/`elvrcx`, states `DXD`/`DX`/`DDXD`/`DDX`). Does not define control commands (`delacx`/`delecx`/`delrcx`). No `init_actuator`.
- `execute` ports `Hyper::actuator`: elevon mix `elvlcx=delecx+delacx`, `elvrcx=delecx-delacx`; `mact==0` position-limit only; `mact==2` second-order `actuator_scnd` (position/rate limits, stored-slope `integrate`); else including 1 → ValueError. Back-convert `delax=(elvlx-elvrx)/2`, `delex=(elvlx+elvrx)/2`. `dt=ctx.int_step`. Local CADAC sign (`<0 → -1` else `+1`); not `np.sign`; not `flat6._cadac_sign`. No Plane6 actuator import.
- Tests: `Python/tests/unit/test_hyper6_actuator.py` (climb `mact=2`, `delecx=1`, `dt=0.01`; `|delex|<=dlimx`; mact=1 raises; elevon mix; formulas vs C++ rtol=1e-12).

## 0.92.0 - HYPER6 propulsion mprop 0-2
- Added `cadac.vehicles.hyper6.propulsion.Hyper6Propulsion` (`name="propulsion"`). Constructor takes `Datadeck`. `define` registers C++ `def_propulsion` (including `mprop`, cowl/throttle/q-hold, `vmass`/`IBBB` and burn-out tensors, fuel state, `ca`/`spi`/`thrust`, rocket/exo slots, `mfreeze` saves). `throttle` default 0.05 as C++. Does not define env/aero names (`vmach`, `pdynmc`, `cd`/`cx`, `area`/`refa`, `alphax`, `time`, `rho`, `dvba`).
- `initialize` ports `Hyper::init_propulsion`: `vmass=vmass0`; GHAME `IBBB0`/`IBBB1`; `IBBB=IBBB0`; `vmass0_st`/`fmass0_st` 0. `execute`: `mprop` 0/1/2 only (climb is 2); else including 3/4 → ValueError (no rocket/LTG). Tables from `ghame6_prop_deck.asc`. Autothrottle q-hold and fuel/mass/`IBBB` interpolate as C++; stored-slope `integrate` of `fmasse`. Skip `mfreeze` latch if `mfreeze` absent.
- Tests: `Python/tests/unit/test_hyper6_propulsion.py` (climb `mprop=2`, vmach~3.3, qhold=200000; throttle in (0, thrtl_max]; mprop=0 thrust 0; mprop=4 raises; formulas vs C++ rtol=1e-12).

## 0.91.0 - HYPER6 GHAME aerodynamics
- Added `cadac.vehicles.hyper6.aero.Hyper6Aero` (`name="aerodynamics"`). Constructor takes `Datadeck`. `define` registers C++ `def_aerodynamics` (including `maero`, GHAME refs, `cx`/`cz`, table coeffs, dimensional der, `tralp`; `cd`/`cl` role `"dia"` as C++). Does not define kinematics/env/propulsion/actuator names (`alphax`, `vmach`, `pdynmc`, `dvba`, `vmass`, `IBBB`, `delax`/`delex`/`delrx`).
- `initialize` ports `Hyper::init_aerodynamics`: `refa=557.42`, `refb=24.38`, `refc=22.86`, termination `trmach`/`trdynm`/`trload`/`tralp`, `refa_st=7`, `caa=0.4`. `execute`: `maero==1` GHAME tables from ASC + body `cx`/`cz` + `aerodynamics_der` as C++. Else including 2 → ValueError (no transfer vehicle). Tests parse `ghame6_aero_deck.asc` (JSONC in Task 16).
- Tests: `Python/tests/unit/test_hyper6_aero.py` (climb `maero=1`, alphax=2.5, vmach from 1000 m/s at 10 km US76; `cx`/`cz` and der finite vs C++ rtol=1e-12; maero=2 raises).

## 0.90.0 - Round6 newton step
- `Round6Newton.execute` ports HYPER6 `Round6::newton`. `FSPB=FAPB/vmass`; `ABII = TBI.T@FSPB + TGI.T@GRAVG` (C++ `NEXT_ACC`); stored-slope `integrate` VBII then SBII. Then WGS84 `cad_geo84_in` / `cad_tdi84` / `cad_tgi84`; `VBED=TDI@(VBII-WEII@SBII)`; polar `psivdx`/`thtvdx` degrees; `TVD=mat2tr`. `vmass`/`FAPB`/`TBI`/`GRAVG` from store (not defined here). Skip `mfreeze` latch if `mfreeze` absent. Module-level `FOOT`/`NMILES` (not `cadac.constants`). No spherical earth / `gravity(alt)`.
- Tests: `Python/tests/unit/test_round6_newton_step.py` (Task 5 climb ICs, FAPB=0, dt=0.01; alt finite; SBII changes; replica NEXT_ACC matches ABII rtol=1e-12; nonzero FAPB → FSPB=[1,0,0]).

## 0.89.0 - Round6 newton initialization
- Added `cadac.eom.round6.Round6Newton` (`name="newton"`). Port of HYPER6 `Round6::def_newton` / `init_newton` **minit==0 only**. `define` registers C++ newton fields including newton-owned `FSPB`. Does not define kinematics/euler/environment names (`time`, `TBI`, `GRAVG`, Euler angles) or hyper waypoints. `ABII` role save (C++).
- Init: `SBII=cad_in_geo84(lonx*RAD,latx*RAD,alt,time)`; VBEB from `alpha0x`/`beta0x`/`dvbe`; `VBED=TBD.T@VBEB`; `VBII=TDI.T@VBED+WEII@SBII`; `TDI`/`TGI` from WGS84; flight-path `psivdx`/`thtvdx` degrees. Does not write `ABII` (stays zeros) or `FSPB`. `minit!=0` → ValueError. `execute` pass until newton step.
- Tests: `Python/tests/unit/test_round6_newton_init.py` (climb lonx=latx=10, alt=10000, dvbe=1000, alpha0x=2.5; `cad_geo84_in(SBII)` alt near 10000; VBII/VBED vs C++ rtol=1e-12).

## 0.88.0 - Round6 Euler equations
- Added `cadac.eom.round6.Round6Euler` (`name="euler"`). Port of HYPER6 `Round6::def_euler` / `init_euler` / `euler`. `define` registers C++ euler fields (`ppx`/`qqx`/`rrx` out+plot, `WBEB` diag, `WBIB`/`WBIBD` state, `WBII` out). Does not define plane/forces `IBBB`/`FMB` or kinematics `TBI`.
- Init: `WBEB=[ppx,qqx,rrx]*RAD` (store rates stay deg/s); `WBIB=WBEB+TBI@(0,0,WEII3)`; writes `WBIB` only (no `WBEBD`). Execute: `WACC_NEXT=inv(IBBB)@(FMB-skew(WBIB)@IBBB@WBIB)`; stored-slope `integrate` of `WBIB`; `WBII=TBI.T@WBIB`; `WBEB=WBIB-TBI@(0,0,WEII3)`; rates in deg/s on store. `dt=ctx.int_step`. No engine momentum (not Flat6).
- Tests: `Python/tests/unit/test_round6_euler.py` (zero FMB, identity TBI, ppx=10, dt=0.01; C++ WBIB replica rtol=1e-12; rates deg/s).

## 0.87.0 - Round6 kinematics DCM
- Added `cadac.eom.round6.Round6Kinematics` (`name="kinematics"`). Port of HYPER6 `Round6::def_kinematics` / `init_kinematics` / `kinematics` (DCM, not Flat6 quaternions). `define` registers C++ kinematics fields (`time`, `event_time`, `int_step_new`, `out_step_fact`, `TBD`/`TBI`/`TBID`, `ortho_error`, Euler `psibd*`/`thtbd*`/`phibd*`, `alppx`/`phipx`/`alphax`/`betax`/`alphaix`/`betaix`). Does not define newton `lonx`/`latx`/`alt`/`SBII`/`VBED`/`VBII`, euler `WBIB`, unused `ck`, or hyper `trcode`.
- Init: `time=ctx.sim_time`, `int_step_new=ctx.int_step`; `TBD=mat3tr(psibdx*RAD, thtbdx*RAD, phibdx*RAD)`; `TDI=cad_tdi84(lonx*RAD, latx*RAD, alt, time)`; `TBI=TBD@TDI`. Execute: stored-slope `integrate` of `TBID_NEW=(-skew(WBIB))@TBI`; orthonormalize as C++; Euler from TBD; aero/inertial incidence as C++. Local CADAC sign (`<0 → -1` else `+1`). Skip `trcode` if absent. Timing `ctx.int_step`/`out_fact` like Round3.
- Tests: `Python/tests/unit/test_round6_kinematics.py` (climb ICs `thtbdx=2.5`, `lonx=latx=10`, `alt=10000`; TBD finite; `WBIB=0` orthonormal `ortho_error`; TBD/TBI vs `mat3tr`/`cad_tdi84` rtol=1e-12).

## 0.86.0 - Round6 environment US76
- Added `cadac.eom.round6.Round6Environment` (`name="environment"`). `define` registers C++ `def_environment` fields used by the mair==0 execute path (`mair`, `press`, `rho`, `vsound`, `vmach`, `pdynmc`, `tempk`, freeze saves, `GRAVG`, `grav`, wind data/state, `VAED`, `dvba`). Does not define newton `alt`/`SBII`/`VBED`, kinematics `time`, or hyper `trcode`/`mfreeze`. Does not define Dryden-only fields.
- `execute` decodes `matmo=mair//100`, `mturb=(mair-matmo*100)//10`, `mwind=(mair-matmo*100)%10`. All-zero: US76 (`atmosphere76`) + `GRAVG=cad_grav84(SBII,time)`, `grav=||GRAVG||`, `VAED=0`, `dvba=||VBED||`. `vmach` not `mach`. Other mair including 100 → ValueError. Skip `trcode`/`mfreeze` latch if those names are absent. `initialize`/`terminate` pass.
- Tests: `Python/tests/unit/test_round6_environment.py` (alt=10000, SBII from `cad_in_geo84(10*RAD,10*RAD,10000,0)`, `||VBED||=1000`; rho/press vs `atmosphere76(10000)` rtol=1e-12; vmach finite; mair=100 and other nonzero triples raise).

## 0.85.0 - WGS84 earth helpers
- Added `cadac.math.wgs84`: HYPER6 WGS84 `cad_in_geo84`, `cad_geo84_in`, `cad_tdi84`, `cad_tgi84`, `cad_grav84` and callee `cad_geoc_in`. Module-level `GM`/`C20`/`FLATTENING`/`SMAJOR_AXIS`/`GW_CLONG`/`SMALL`. Does not change spherical `cadac.math.earth` or locked `cadac.constants`.
- Tests: `Python/tests/unit/test_wgs84.py` (equator roundtrip atol=1e-6; grav finite; `cad_tdi84(0,0,0,0)` `assign_loc` pin rtol=1e-12).

## 0.84.2 - Hyper5 control_altitude reads Round3 vbeg
- `Hyper5Control.control_altitude` reads store `vbeg` (Round3 Newton name), not `VBEG`. Same rule as `tgv`. mcontrol 6/16/36 no longer KeyError on a defined Hyper5.
- Tests: `test_mcontrol_6_execute_reads_round3_vbeg_on_hyper5`; Task 8 plants `vbeg`.

## 0.84.1 - HYPER5 e2e skip without golden
- Added `Python/tests/e2e/test_hyper5_pronav.py`: skip if `tests/e2e/goldens/hyper5/plot.csv` absent (file not created). Fixture tests: skip helper, sentinel `time=-1`, `alt` at t=0, shared-column intersection `rtol=1e-5` / `atol=max(1e-6, 5e-6*|g|)`. Live `run_scenario` on `cases/hyper5/input.jsonc` calls `require_golden` first.

## 0.84.0 - HYPER5 and TARGET3 from JSONC
- `cadac.vehicles.hyper5.vehicle.Hyper5` (`type="HYPER5"`, health=1). Constructor `(name, aero_deck, prop_deck, events=None)` with `prop_deck` allowed `None`. Modules: environment, aerodynamics, propulsion, forces, newton, seeker, guidance, control, intercept, targeting (define always; Demo 4.7 MODULES omits targeting execute). Simple Cruise3-style define loop.
- `run_scenario` maps `HYPER5`/`TARGET3`/`SATELLITE3`. HYPER5 requires aero; prop if `mprop!=0` or `prop_deck` present. TARGET3/SATELLITE3: `Target3(name, events)` / `Satellite3(name, events)` — no decks. CRUISE3/PLANE/PLANE6 still require both decks. Unknown types still ValueError (`"HYPER6"` stays the unknown-type sentinel).
- `run_loop` seeds combus from store `com_names` before the first execute (CADAC `loading_packet_init`) so Hyper seeker can read TARGET3 `lonx`/`latx` at t=0. Skip-missing modules and plot slot 0 unchanged. `_plot_columns` CRUISE3 path unchanged.
- Translated `CADAC_Simulations/HYPER5_250113/HYPER5/input.asc` + `hyper5_aero_deck.asc` to `Python/cases/hyper5/` (`end_time` 25, RR3X+Truck_t1, mprop=0, mcontrol=44, mguidance=66, mseeker=1, acq_range=6000). No e2e golden.
- Tests: `Python/tests/unit/test_hyper5_one_step.py` (0.05 s smoke alt finite health 1; committed case; deck policy; TARGET3/SATELLITE3 no decks). Combus seed: `test_run_loop_seeds_combus_from_store_before_first_execute`.

## 0.83.0 - HYPER5 satellite targeting
- `Hyper5Targeting.execute`: `mtargeting==0` return; `==1` ports C++ `targeting` / `targeting_satellite` / `targeting_grnd_ranges`; else including 2 ValueError. Writes `wp_lonx`/`wp_latx`/`wp_alt` from closest TARGET3 (guidance-owned; not defined here) plus `clost_tgt_slot`/`tgtng_sat_slot`. Skip cout/`out_count`.
- Identify SATELLITE3/TARGET3 by `Packet.type` (not CADAC `id.find`). Kinematics from packet.vars `lonx`/`latx`/`alt`/`sbii`; Hyper store `del_radius`/`lonx`/`latx`/`sbii`. `LARGE=1e10` module-level; not in `cadac.constants`. Local CADAC `angle()` (EPS; not Flat6). Port `angle(SBII,STII)` as written. No Plane5 import.
- Tests: `Python/tests/unit/test_hyper5_targeting.py` (one Hyper+Target+Satellite; mtargeting=1 waypoint vs target lon/lat/alt; mtargeting=2 raises; type not id; closest by ground range; rtol=1e-12, atol=1e-14). Task 16 noop: mtargeting=1 no longer raises.

## 0.82.1 - Satellite3 sat_thrust default 0
- `Satellite3Forces.define`: `sat_thrust` default 0 (C++ `satellite[4].init("sat_thrust",0)`). `sat_mass` stays 100. Demo ICs omit sat_thrust → FSPV[0]=0.

## 0.82.0 - HYPER5 SATELLITE3 forces
- Added `cadac.vehicles.hyper5.satellite`: `Satellite3Forces` (`name="forces"`), `Satellite3` (`type="SATELLITE3"`, health=1). Constructor `(name, events=None)` — no decks. Modules: Round3Environment, Satellite3Forces, Round3Newton (C++ MODULES order; forces before newton). No seeker. `com_names` from store fields with `"com"` in outputs. Not registered in cli (Task 20).
- Forces: C++ `def_forces` FSPV (vec out, skip-if-exists), sat_thrust (data, default 0), sat_mass (data, default 100). `execute` ports `Satellite::forces`: `FSPV=[sat_thrust/sat_mass, 0, 0]`. Round3 lowercase names. `initialize`/`terminate` pass. Does not import Hyper5Forces/Cruise3/Plane5.
- Tests: `Python/tests/unit/test_satellite3_forces.py` (sat_thrust=0, sat_mass=100 → FSPV[0]==0; sat_thrust=100 → FSPV[0]==1; rtol=1e-12, atol=1e-14).

## 0.81.0 - HYPER5 TARGET3 vehicle
- Added `cadac.vehicles.hyper5.target`: `Target3Forces` (`name="forces"`), `Target3Intercept` (`name="intercept"`), `Target3` (`type="TARGET3"`, health=1). Constructor `(name, events=None)` — no decks. Modules: Round3Environment, Target3Forces, Round3Newton, Target3Intercept (C++ MODULES order; forces before newton). `com_names` from store fields with `"com"` in outputs. Not registered in cli (Task 20).
- Forces: C++ `def_forces` FSPV (vec out, skip-if-exists), fwd_accel/side_accel (data), CORIO_V/CENTR_V (vec diag). `execute` ports `Target::forces`: TVG=tgv.T, TGI=tig.T, TEG=tge.T, WEIG=tge@weii@TEG, CORIO_V=TVG@WEIG@vbeg*2, CENTR_V=TVG@WEIG@WEIG@TGI@sbii, GRAV_G=[0,0,grav], ACC_V=CORIO+CENTR-GRAV_V, FSPV += fwd_accel/side_accel. Round3 lowercase names. Does not read unused C++ local thtvgx. Does not import Hyper5Forces/Cruise3/Plane5. `initialize`/`terminate` pass.
- Intercept: C++ `def_intercept` targ_health (int diag). `execute`: `targ_health = ctx.combus[ctx.vehicle_slot].status`. Skip cout. `initialize`/`terminate` pass.
- Tests: `Python/tests/unit/test_target3_forces.py`, `test_target3_intercept.py` (Demo 4.7 Truck lon/lat/alt; fwd=side=0 FSPV finite vs C++ rtol=1e-12, atol=1e-14; combus status 0 → targ_health 0).

## 0.80.0 - HYPER5 targeting stub
- Added `cadac.vehicles.hyper5.targeting.Hyper5Targeting` (`name="targeting"`). `define` registers C++ `def_targeting`: mtargeting (int, data, scrn/plot), del_radius (data), clost_tgt_slot/tgtng_sat_slot (int, out). Does not define guidance waypoints (wp_lonx/wp_latx/wp_alt) or newton/seeker names. `initialize`/`terminate` pass (no C++ `init_targeting`).
- `execute`: `mtargeting==0` return without writing (C++ return before satellite/target work). Else including 1 ValueError until Task 19. Protocol `vehicle.store`. No vehicle. No Plane5 import.
- Tests: `Python/tests/unit/test_hyper5_targeting_noop.py` (mtargeting=0 no raise/no write; 1/2/99/-1 raise; define-only C++ fields).

## 0.79.0 - HYPER5 Hyper intercept halt/hit
- Added `cadac.vehicles.hyper5.intercept.Hyper5Intercept` (`name="intercept"`). `define` registers C++ `def_intercept`: write (int, save, default 1), miss/hit_time/MISS_G (diag), time_m/SBTGM/STMEG/SBMEG (save), halt (int, data). Does not define newton/seeker/guidance names. `initialize`/`terminate` pass (no C++ `init_intercept`).
- `execute` ports `Hyper::intercept` without cout/`sys.exit`: `halt and write` or `alt<=0 and write` → write=0, `vehicle.health=0`, `ctx.combus[slot].status=0`. `mseeker==3` and `range_go<1000`: STEG from `ctx.combus[targ_com_slot].vars["sbeg"]`; closest-approach interpolation when `closing_speed<0 and write` (hit_time/MISS_G/miss, then same kill); always save previous SBTGM=`-STBG`, STMEG, SBMEG=`sbeg`, time_m. Always writes write and diagnostics. Does not set target packet status. Protocol `vehicle.store`. No vehicle. No Plane5 import.
- Tests: `Python/tests/unit/test_hyper5_intercept.py` (halt=1 write=1 kills; halt=0 no kill; ground alt<=0; interpolation vs C++ rtol=1e-12, atol=1e-14; previous-step vectors; write latch; SimContext combus Packet).

## 0.78.1 - Hyper5Seeker terminate pass
- `Hyper5Seeker.terminate` is `pass` like the Module protocol / DummyModule. Does not write. Test: `test_terminate_exists_and_is_pass`.

## 0.78.0 - HYPER5 seeker acquire and track
- Added `cadac.vehicles.hyper5.seeker.Hyper5Seeker` (`name="seeker"`). `define` registers C++ `def_seeker`: mseeker (int, data/save, scrn), acq_range (data), range_go (out, plot/scrn), STBG (vec, out, plot), WOEB (vec, out), closing_speed (out), time_go (out, plot/scrn), psisbx/thtsbx (out, plot/scrn), targ_com_slot (int, save), UTBB (vec, out), acquisition (int, init/save, scrn). Does not define mcontrol/mguidance (control/guidance). Does not define plant time/tig/vbeg/sbii/TBG/lonx/latx. `initialize` pass (no C++ `init_seeker`).
- `execute` ports `Hyper::seeker`: `mseeker==0` return without writing. `1` acquire if TARGET3 ground range `< acq_range` then mseeker=3 and acquisition=1 (Demo 4.7 acq_range=6000). Same execute then tracks if mseeker became 3. `3` track: STBG, range_go, WOEB=`TBG@_skew(UTBG)@VTBG*inv_dtb`, closing_speed, UTBB, time_go, psisbx/thtsbx. Else including 2/99 ValueError. Writes mseeker/acquisition/targ_com_slot; does not write mcontrol/mguidance. Skip cout.
- Targets: `packet.type=="TARGET3"` (not CADAC `id.find("t")`). Kinematics from `packet.vars` names `lonx`/`latx`/`vbeg`/`sbii`. Own lonx/latx from Hyper store. Ground range `REARTH*acos(dum)` as C++. Local CADAC `_skew` (not imported from flat6). Protocol `vehicle.store`. No vehicle. No Plane5 import.
- Tests: `Python/tests/unit/test_hyper5_seeker.py` (two packets Hyper+Target; range inside 6000 → mseeker=3 and track vs C++ rtol=1e-12, atol=1e-14; mseeker=0 no write; 2/99/-1 raise; type not id prefix; store lonx/latx not combus).

## 0.77.0 - HYPER5 mguidance dispatcher
- `Hyper5Guidance.execute` is the mguidance dispatcher for `{0,44,66,70}` only. `0` zeros local alcomx/ancomx and returns without writing (C++ return before `gets`). `44` APGV=`guidance_point()`; alcomx=APGV[1]/grav; ancomx=-APGV[2]/grav. `66` APNB=`guidance_pronav()`; same mapping. `70` phicx=`guidance_arc()`. Else including unused C++ 30/33/3/6/40/43/60/99 → ValueError. Then clip ancomx to [anneglimx, anposlimx] and alcomx to ±allimx; write phicx, ancomx, alcomx (control-owned; tests register limiters).
- `guidance_arc(vehicle)` ports `Hyper::guidance_arc` including `if(dwbh<0.2*rad_min)` (not commented 2*rad_min), CADAC sign, local CADAC `angle()` (EPS from constants; not added to `cadac.constants`), `asin` guard `denom!=0` plus `|argument|<=1` so Python matches C++ NaN-else. Returns phicx; writes SWBG, wp_grdrange, rad_min, wp_flag. Round3 names time/FSPV/grav/tig/dvbe/vbeg/sbii. Does not read unused C++ local psivgx. No `if(time>49)` debug. Protocol `vehicle.store`. No vehicle. No Plane5 import.
- Tests: `Python/tests/unit/test_hyper5_mguidance.py` (Demo 4.7 mguidance=66 finite clipped commands; 0 no write; 30/33/99/3/6/40/43/60 raise; 44 point; 70 arc Demo 5.12 phicx and zeroed loads; arc vs C++ rtol=1e-12, atol=1e-14; 0.2*rad_min wp_flag). Tasks 11–12 execute-is-pass now assert mguidance=0 no write.

## 0.76.0 - HYPER5 pro-nav guidance
- `Hyper5Guidance.define` adds C++ `def_guidance` pronav fields: pronav_gain, bias (data). Does not define seeker names (WOEB, UTBB, closing_speed, range_go) or TBG (control). Does not define unused C++ locals psisbx/thtsbx. Line/arc extras still omitted. `execute` still pass until Task 13 dispatcher.
- `guidance_pronav(vehicle)` ports `Hyper::guidance_pronav`: GRAV_G=[0,0,grav+bias]; APNB=`_skew(WOEB)@UTBB*(pronav_gain*closing_speed)-TBG@GRAV_G`. Local CADAC row-major cross-product `_skew` (same layout as Flat6; not imported). Does not read unused C++ locals range_go/psisbx/thtsbx. Returns APNB; does not write alcomx/ancomx. Protocol `vehicle.store`. No vehicle. No Plane5/Flat6 import.
- Tests: `Python/tests/unit/test_hyper5_guidance_pronav.py` (Demo 4.7 pronav_gain=3.5, bias=5; frozen seeker vectors; APNB vs C++ rtol=1e-12, atol=1e-14; unused locals unregistered; execute pass). Plant grav/TBG/WOEB/UTBB/closing_speed/range_go registered by the test.

## 0.75.0 - HYPER5 point guidance
- Added `cadac.vehicles.hyper5.guidance.Hyper5Guidance` (`name="guidance"`). `define` registers C++ `def_guidance` point+dispatcher fields: mguidance (int, data, scrn), wp_lonx/wp_latx/wp_alt, point_gain (data), wp_sltrange (diag, scrn/plot, default 999999), VBEO (vec, diag), wp_grdrange (diag, scrn/plot, default 999999), SWBG (vec, out), rad_min (diag), wp_flag (int, diag). C++ `dia` → `"diag"`. Does not define pronav/line/arc extras or philimx. `initialize` pass (no C++ `init_guidance`). `execute` pass until Task 13 dispatcher.
- `guidance_point(vehicle)` ports `Hyper::guidance_point`: SWII=`cadine(wp_lonx*RAD, wp_latx*RAD, wp_alt, time)`; SWBG=`tig.T@(SWII-sbii)`; polar_from_cart/mat2tr LOS; wp_grdrange=hypot; VBEO=TOG@vbeg; APGV steering with point_gain and gravity terms on **thtvgx** (deg); rad_min=`dvbe**2/(grav*tan(philimx*RAD))` with dvbe=||vbeg||; wp_flag CADAC sign(VH·SH) inside 2*rad_min else 0 (`<0 → -1` else `+1`, never 0). Writes listed diagnostics; returns APGV. Round3 names lowercase (`time`/`grav`/`tig`/`thtvgx`/`vbeg`/`sbii`). No C++ `time>54` debug. Protocol `vehicle.store`. No vehicle. No Plane5Guidance import.
- Tests: `Python/tests/unit/test_hyper5_guidance_point.py` (Demo 4.6 waypoint lon/lat/alt offset finite APGV; wp_flag 0 outside, +1 closing, -1 fleeting; CADAC sign zero-dot +1; one-step vs C++ replica; execute pass; rtol=1e-12, atol=1e-14). Plant time/grav/tig/thtvgx/vbeg/sbii/philimx registered by the test.

## 0.74.1 - Hyper5Control reads Round3 tgv
- `Hyper5Control.execute` reads store `tgv` (Round3 Newton name), not `TGV`. No Round3 alias. Hyper-owned `TBV`/`TBG` unchanged. Tests plant `tgv`.

## 0.74.0 - HYPER5 mcontrol dispatcher
- `Hyper5Control.define` adds remaining C++ `def_control` dispatcher/lateral fields: mcontrol (int, data, scrn), TBV/TBG (mat, out), alcomx (data, scrn/plot), allimx/gcp (data), alx (diag, plot), alphacx/phimvcx (data). Skip-if-exists. Bank/load/altitude/heading fields unchanged.
- `control_lateral(vehicle, alcomx)` ports `Hyper::control_lateral`: TBV=`cadtbv(phimvx*RAD, alphax*RAD)`, FSPB=TBV@FSPV, anx=-FSPB[2]/grav, clip alcomx ±allimx, **phic=atan2(alcomx, anx)** (not Plane5 gcp), phicx=phic*DEG, alx=FSPV[1]/grav. Writes alx; returns phicx; does not write phicx/TBV. Protocol `vehicle.store`. No vehicle. No Plane5Control import.
- `execute` is the mcontrol dispatcher for allowed modes `{0,3,4,6,16,36,40,44}` only (`mcontrol 03` is int 3). Separate `if`s as C++. Locals phimvx/alphax start 0. `0` zeros then still TBV/TBG/`gets()`. Else including -1/1/10/11/46 ValueError. Then `TBV=cadtbv(phimvx*RAD, alphax*RAD)`, `TBG=TBV@tgv.T`. Writes phicx, TBV, TBG, alphax, phimvx, ancomx.
- Tests: `Python/tests/unit/test_hyper5_mcontrol.py` (Demo 4.7 mcontrol=44 alcomx=0.5 finite phimvx/alphax; mcontrol=-1 raises; mcontrol=0 zeros; atan2 vs gcp; TBG; mcontrol=3 alphacx; rtol=1e-12, atol=1e-14). Tasks 6–8 execute-is-pass now assert dispatcher. Plant tgv/FSPV/grav registered by tests.

## 0.73.0 - HYPER5 heading and flight-path control
- `Hyper5Control.define` adds C++ `def_control` fields used by `control_heading` / `control_flightpath`: gain_thtvg, gain_psivg, psivgcx, thtvgcx, avx. alphax/anx/alpposlimx/alpneglimx already from Task 7. Skip-if-exists. Does not register mcontrol/TBV/TBG/alcomx/lateral. Bank, load, altitude fields unchanged. `execute` still pass.
- `control_heading(vehicle, psivgcx)` ports `Hyper::control_heading`: reads geographic **psivgx** (deg), not Flat3 psivlx. Wrap: if fabs(psivgcx)<=135 then psivgx_comp=psivgx; else if psivgx*psivgcx>=0: psivgx; else wrap `360-psivgx*sign` with C++ `if(psivgx>=0) sign=1 else sign=-1`. Returns bank command; does not write phimvx/phicx. Protocol `vehicle.store`. No vehicle. No Plane5Control import.
- `control_flightpath(vehicle, thtvgcx, phimvx)` ports `Hyper::control_flightpath`: reads **thtvg in radians** (Round3 `round3[18]`); command `thtvgcx*RAD`. avx=gain_thtvg*(thtvgcx*RAD-thtvg); anx=avx/cos(phimvx*RAD); alphax=anx*mass*grav/(pdynmc*area*cla); clip alpposlimx/alpneglimx. Writes anx,avx; returns alphax; does not write alphax.
- Tests: `Python/tests/unit/test_hyper5_control_heading.py` (Demo 5.1 gain_psivg=2; heading wrap `|psivgcx|>135` opposite-sign; Demo 4.7 alpposlimx=6/alpneglimx=-4 alphax clip vs C++ replica; thtvg radians not degrees; execute pass via existing bank/load/altitude tests; rtol=1e-12, atol=1e-14). Plant psivgx/thtvg/pdynmc/grav/mass/area/cla registered by the test.

## 0.72.0 - HYPER5 altitude control
- `Hyper5Control.define` adds C++ `def_control` fields used by `control_altitude`: altdlim, gh, gv (data), altd (diag, plot), altcom (data, plot). anposlimx/anneglimx already from Task 7. Skip-if-exists. Does not register alt/grav/VBEG (newton/environment). Does not register mcontrol, TBV/TBG, heading, lateral. Bank and load fields unchanged. `execute` still pass.
- `control_altitude(vehicle, altcom, phimvx)` ports `Hyper::control_altitude`: ealt=gh*(altcom-alt) clipped ±altdlim; altd=-VBEG[2]; ancomx=(gv*(ealt-altd)/grav+1)*(1/cos(phimvx*RAD)); clip [anneglimx, anposlimx]. Writes altd; returns ancomx; does not write ancomx. Protocol `vehicle.store`. No vehicle. No Plane5Control import. Does not use VBEL.
- Tests: `Python/tests/unit/test_hyper5_control_altitude.py` (Demo 5.1 fly-out gh=0.2, gv=0.3, altdlim=50, altcom=24000, anposlimx=2, anneglimx=-1; one-step vs C++ replica; rate-limiter; banked RAD; ancomx clip; execute pass; rtol=1e-12, atol=1e-14). Plant alt/grav/VBEG registered by the test.

## 0.71.0 - HYPER5 load-factor control
- `Hyper5Control.define` adds C++ `def_control` fields used by `control_load`: anposlimx, anneglimx, gacp, ta, alphax (out, scrn/plot), alpposlimx, alpneglimx, xi, xid, alp, alpd, anx (diag, scrn/plot), qq, tip, ancomx (data, scrn/plot). Skip-if-exists. Does not register mcontrol, TBV/TBG, altitude, heading, lateral. Bank fields unchanged. `execute` still pass.
- `control_load(vehicle, ancomx, int_step)` ports `Hyper::control_load`: TBV=`cadtbv(phimv,alpha)`, FSPB=TBV@FSPV, clip ancomx to [anneglimx, anposlimx], anx=-FSPB[2]/grav, tip=dvbe*mass/(pdynmc*area*cla/RAD+thrust), P-I if ta>0 else xi=0 (gr starts 0), incidence lag, clip returned alpx. Writes xi,xid,alp,alpd,anx,qq,tip. Does not write alphax. Protocol `vehicle.store`. No vehicle. No Plane5Control import.
- Tests: `Python/tests/unit/test_hyper5_control_load.py` (Demo 4.7 gacp=10, ta=0.8, anposlimx=2, anneglimx=-2, alpposlimx=6, alpneglimx=-4; one-step vs C++ replica; ta<=0; ancomx/alpx clips; execute pass; rtol=1e-12, atol=1e-14). Plant FSPV/grav/mass/dvbe/pdynmc/thrust/area/cla registered by the test. Bank tests still pass.

## 0.70.0 - HYPER5 bank-angle control
- Added `cadac.vehicles.hyper5.control.Hyper5Control` (`name="control"`). `define` registers C++ `def_control` fields used by `control_bank`: phimvx (out, scrn/plot), phicx (data, scrn/plot), phix (state, plot), phixd (state), philimx, tphi. Does not register mcontrol, load, altitude, heading, TBV/TBG. `initialize` pass (no C++ `init_control`). `execute` pass until Task 10 dispatcher.
- `control_bank(vehicle, phicx, int_step)` ports `Hyper::control_bank`: clip phicx to ±philimx (local), `phixd_new=(phicx-phix)/tphi`, stored-slope `integrate`, writes phix/phixd, returns phix. Does not write phimvx/phicx. Protocol `vehicle.store`. No vehicle. No Plane5Control import.
- Tests: `Python/tests/unit/test_hyper5_control_bank.py` (Demo 4.7 philimx=70, tphi=1, int_step=0.05; one- and two-step lag vs C++ replica; limiter phicx=90 and -90; execute pass; rtol=1e-12, atol=1e-14).

## 0.69.0 - HYPER5 Hyper forces FSPV
- Added `cadac.vehicles.hyper5.forces.Hyper5Forces` (`name="forces"`). `define` registers C++ `def_forces` `FSPV` only (vec, out, plot). Skip-if-name-exists (Round3 newton does not define FSPV). Does not define pdynmc/cl/cd/area/thrust/mass/alphax/phimvx/time (plant/control; tests register those). `initialize` pass (no C++ `init_forces`).
- `execute` ports `Hyper::forces`: `phimv=phimvx*RAD`, `alpha=alphax*RAD`; `fspv1=(-pdynmc*area*cd+thrust*cos(alpha))/mass`, `fspv2=sin(phimv)*(pdynmc*area*cl+thrust*sin(alpha))/mass`, `fspv3=-cos(phimv)*(pdynmc*area*cl+thrust*sin(alpha))/mass`. Protocol `vehicle.store`. No vehicle. No Cruise3/Plane5 imports or shared helper.
- Tests: `Python/tests/unit/test_hyper5_forces.py` (pdynmc=72000, area=11.6986, cd=0.05, cl=0.2, thrust=0, mass=1352, alphax=-1.5, phimvx=0 and 90 vs C++ formulas rtol 1e-12, atol 1e-14; FSPV-only define; skip-if-exists).

## 0.68.0 - HYPER5 propulsion mprop 0-3
- Added `cadac.vehicles.hyper5.propulsion.Hyper5Propulsion` (`name="propulsion"`). Constructor takes Datadeck or `None`. `define` registers C++ `def_propulsion` (phi_const, tlag, phis/phisd, mprop, aintake, phi, phi_max/min, qhold, mass/mass0, cin, tq, fmass0, fmasse/fmassd, thrst_stoch, spi, thrust, mass_flow, fmassr, thrst_req). C++ role `dia` maps to `"diag"`. Does not define rho/pdynmc/mach/dvbe/ca/area/alphax/time (plant; unused C++ local time not read). `initialize` sets `mass=mass0`.
- `execute` ports `Hyper::propulsion`: `mprop` not in `{0,1,2,3}` (including -1) ValueError. Locals `phi`/`spi`/`cin`/… start at 0. `mprop>0`: cin look_up `cin_vs_alphax_mach`; first spi look_up `spi_vs_mach_phi_alphax(mach, phi, alphax)` with that local phi (0) before 1/2/3 branches. mprop=1: `thrust=0.0676*phi_const*spi*AGRAV*rho*dvbe*cin*aintake`. mprop=2: q-hold, tlag integrate of phis, clip phi only (phis unclipped), second spi look_up with `phi/0.0676`. mprop=3: C++ `phi;` no-op — does not read saved store phi; look_up/`thrust` as written (local phi stays 0). Fuel integrate inside `mprop>0`; `fmassr<=0` → `mprop=0`; then `mprop==0` zeros fmassd/thrust. Protocol `vehicle.store`. No forces/vehicle. No Cruise3/Plane5 imports. Parsed `hyper5_prop_deck.asc` in tests.
- Tests: `Python/tests/unit/test_hyper5_propulsion.py` (mprop=0 no look_up with None/boom deck; mprop=1 vs C++ formula rtol 1e-12; mprop=-1/4/99 raise; mprop 2 lag/clips/stored-slope/`phi/0.0676`; mprop=3 keep-phi-as-written; fuel cutoff).

## 0.67.0 - HYPER5 Roadrunner aerodynamics
- Added `cadac.vehicles.hyper5.aero.Hyper5Aero` (`name="aerodynamics"`). Constructor takes Datadeck. `define` registers C++ `def_aerodynamics` (cl, cd, cl_ov_cd, area default 0, cla, cn, ca). Does not define mach/alphax (environment/control) or unused C++ local time. C++ role `dia` maps to `"diag"`. `initialize` pass (no C++ `init_aerodynamics`).
- `execute` ports `Hyper::aerodynamics`: 2D look_up `cn_rr3x_vs_alphax_mach` / `ca_rr3x_vs_alphax_mach`; `cd=cn*sin(alpha)+ca*cos(alpha)`; `cl=cn*cos(alpha)-ca*sin(alpha)`; `cla` from ±2 deg (`cna=(cnp-cnn)/4`, `caa=(cap-can)/4`, `cla=cna*cos(alpha)-caa*sin(alpha)`). `alpha=alphax*RAD`. Protocol `vehicle.store`. Parsed `hyper5_aero_deck.asc` in tests (JSONC Task 20). No Cruise3/Plane5 imports. No propulsion/forces/vehicle.
- Tests: `Python/tests/unit/test_hyper5_aero.py` (mach=4, alphax=2, area=11.6986 IC vs look_up then C++ formulas rtol 1e-12, atol 1e-14).

## 0.66.0 - Skip missing modules; plot vehicle 0
- `run_loop` skips module names absent on a vehicle (`named.get` / continue); no KeyError when HYPER5 Target shares MODULES with Hyper.
- `make_plot_on_step` records plot rows only for `vehicle_slot == 0`; still advances `plot_time` once per tick on the last slot. Single-vehicle CRUISE3/PLANE/PLANE6 unchanged. `_plot_columns` CRUISE3 special case kept.
- Tests: `test_skip_module_absent_on_vehicle`; `test_plot_rows_only_vehicle_slot_0`.

## 0.65.0 - CADAC cadine lon/lat/alt to inertial
- Added `cadine(lon_rad, lat_rad, alt_m, time)` to `cadac.math.earth` next to `cadtei`/`cadtge`/`cadsph`. Ports HYPER5 C++ `cadine`: spherical radius `alt+REARTH`, celestial longitude `lon+WEII3*time`, inertial position `[rad*clat*clon, rad*clat*slon, rad*slat]`. ndarray `(3,)`. Constants from `cadac.constants`. No scipy. No guidance.
- Tests: `Python/tests/unit/test_cadine.py` (equator t=0 → `[REARTH,0,0]`; celestial-longitude formula rtol 1e-12). `cadtei`/`cadtge`/`cadsph` numerics unchanged.

## 0.64.1 - FALCON6 e2e skip-if-missing golden
- Added `Python/tests/e2e/test_falcon6_gamma.py`: `pytest.skip` if `tests/e2e/goldens/falcon6/plot.csv` is absent (file not created).
- If golden exists: `run_scenario` on `Python/cases/falcon6/input_gamma.jsonc`; compare `hbe` at t=0 and `hbe`/`vmach` at shared times (Flat6 names, not `alt`/`mach`). CSV rtol=1e-5, atol=max(1e-6, 5e-6*|g|); skip sentinel time=-1.
- Covering tests: skip helper, sentinel, t=0 hbe tolerances, shared-column intersection (`hbe`/`vmach` on both; `mach` is not `vmach`). Cruise3/HYPER3/PLANE production unchanged. FALCON5 e2e still skips without its golden.

## 0.64.0 - FALCON6 PLANE6 from JSONC
- Added `cadac.vehicles.plane6.vehicle.Plane6` (`type="PLANE6"`, health=1). Modules in gamma ASC order plus guidance: environment, kinematics, aerodynamics, propulsion, guidance, forces, control, actuator, euler, newton. `define` skip-if-exists (FALCON5 TBV / Cruise3 FSPV); last execute writer owns the plotted value. Guidance `define` still runs if MODULES omits it.
- `run_scenario` type map adds `PLANE6 -> Plane6`. CRUISE3 and PLANE kept. aero_deck and prop_deck still required. `_plot_columns`: CRUISE3 still `PLOT_COLUMNS`; PLANE6 uses the existing non-CRUISE3 flagged path (no HYPER3 special case).
- Translated `CADAC_Simulations/FALCON6_250201/FALCON6/input_gamma.asc` + `f16_aero_deck.asc` / `f16_prop_deck.asc` to `Python/cases/falcon6/`. Unknown-type tests retargeted from `"PLANE6"` to `"HYPER6"`.
- Tests: `Python/tests/unit/test_plane6_one_step.py` (0.1 s smoke, `hbe` near 1000, flagged columns not CRUISE3, committed end_time 20). No Task 17 e2e golden.

## 0.63.0 - F-16 guidance stub
- Added `cadac.vehicles.plane6.guidance.Plane6Guidance` (`name="guidance"`). `define` registers C++ `def_guidance` (mguid, line_gain, nl_gain_fact, decrement, swel1/2/3, psiflx, thtflx, dwb, nl_gain, VBEO, VBEF, dwbh, SWBL, turn_min, wp_flag). Does not define newton/control names (time, halt, grav, SBEL, VBEL, dvbe, psivlx, thtvlx, philimx, phicomx, ancomx, alcomx). `initialize` pass (no C++ `init_guidance`).
- `execute` ports C++ `if(mguid==0) return` without writing phicomx/ancomx/alcomx (control-owned) or diagnostics. mguid!=0 ValueError. Does not port `guidance_line` / waypoint / cout / halt. Protocol `vehicle.store`. No vehicle. Cruise3/HYPER3/PLANE untouched.
- Tests: `Python/tests/unit/test_plane6_guidance_noop.py` (mguid 0 no raise/no write; 30/33 raise; define-only C++ fields).

## 0.62.0 - F-16 forces
- Added `cadac.vehicles.plane6.forces.Plane6Forces` (`name="forces"`). `define` registers C++ `def_forces` only: FAPB and FMB vec out. Does not define pdynmc/thrust/refa/refb/refc/cxt/cyt/czt/clt/cmt/cnt (tests register), FSPB/vmass (newton), or unused C++ local time. `initialize` pass (no C++ `init_forces`).
- `execute` ports `Plane::forces`: FAPB=[pdynmc*refa*cxt+thrust, pdynmc*refa*cyt, pdynmc*refa*czt]; FMB=[pdynmc*refa*refb*clt, pdynmc*refa*refc*cmt, pdynmc*refa*refb*cnt]. No FSPB. Protocol `vehicle.store`. No guidance/vehicle. Cruise3/HYPER3/PLANE untouched.
- Tests: `Python/tests/unit/test_plane6_forces.py` (frozen aero/thrust vs CADAC rtol 1e-12; define-only FAPB/FMB; initialize pass; FSPB/vmass/time not written or required).

## 0.61.0 - F-16 maut dispatcher
- `Plane6Control.execute` ports C++ `Plane::control` for `input_gamma.asc` / `input_roll.asc` only: maut 0 returns without writing commands; 1 roll-only (`mauty=0`,`mautp=1`); 24 yaw SAS + gamma then roll (`mauty=2`,`mautp=4`); else including -1 ValueError. Decode `mauty=maut//10`, `mautp=maut%10`. Does not implement mauty 3/4 or mautp 2/3/5. `mroll` 0 clamps `phicomx` by `philimx` with CADAC sign then `control_roll`; 1 `control_roll_rate`; else ValueError. Limit `|del*|` by `d*limx` with CADAC sign. Stores delacx/delecx/delrcx/ancomx/phicomx. `dt` unused except C++ signature. Gamma omits mroll (define default 0). Cruise3/HYPER3/PLANE untouched. No forces/vehicle.
- Tests: `Python/tests/unit/test_plane6_maut.py` (frozen aero/kinematics/rates/dvbe; maut 24 vs CADAC rtol 1e-12; maut -1/unknown raise; maut 0 no write; maut 1 roll-only; mroll 1; philimx/surface limiters CADAC sign). Task 11/12 execute-is-pass tests now assert dispatcher.

## 0.60.0 - F-16 gamma controller
- Added `Plane6Control.control_gamma(vehicle, thtvlcomx) -> delecx`. Ports C++ `Plane::control_gamma`: pole-placement DP/DD (`pgam`/`wgam`/`zgam`), `GAINGAM=inv(DP)@DD` shape (3,), `DUM33=AA-outer(BB,GAINGAM)`, `gainff=-1/(HH·(inv(DUM33)@BB))` with HH=[0,0,1], then `thtc/qqf/thtblf/thtvlf` in rad and `delecx=delec*DEG`. Stores GAINGAM and gainff. Does not write delecx. Does not read unused C++ locals `time`/`pdynmc`. C++ `if(dvbe==0)dvbe=dvbe` no-op omitted. CADAC row-major AA/BB/DP; `np.linalg.inv` as with IBBB. RAD/DEG from constants. `execute` still pass (no maut dispatcher). Cruise3/HYPER3/PLANE untouched.
- Tests: `Python/tests/unit/test_plane6_control_gamma.py` (input_gamma poles pgam=10 wgam=3 zgam=0.5, thtvlcomx=1 deg, frozen nonzero dvbe/dla/dmde; finite elevator vs CADAC rtol 1e-12; no time/pdynmc; execute pass).

## 0.59.0 - F-16 roll and rate SAS
- Added `cadac.vehicles.plane6.control.Plane6Control` (`name="control"`). `define` registers full C++ `def_control` (maut/mroll/mfreeze, limiters, wrcl/zrcl/tp/zetlagr, delacx/delecx/delrcx, gkp/gkphi, zrate/grate/wnlagr, GAINFP/GAINGAM vec (3,), isetc2 real init). Does not define kinematics/aero names these functions read (phiblx, ppx, dllp, dllda, dla, …, dvbe, qqx, rrx). `initialize` pass (no C++ `init_control`). `execute` pass until Task 13 maut dispatcher.
- Methods port C++ as written: `control_roll` pole-placement gkp/gkphi, RAD/DEG; `control_roll_rate` kp=(1/tp+dllp)/dllda; `control_pitch_rate` zrate/aa/bb, radix clamp 0, |dmde|<SMALL then SMALL*sign; `control_yaw_rate` similar with dyb/dydr/dnb/dnr/dndr, stores zrate/grate/wnlagr. Local `_sign` (`<0 → -1` else `+1`); SMALL=1.e-7 module-level, not in `cadac.constants`. RAD/DEG from constants. Returns commands; does not write delacx/delecx/delrcx. No control_gamma, maut dispatcher, forces, vehicle. Cruise3/HYPER3/PLANE untouched.
- Tests: `Python/tests/unit/test_plane6_control_roll.py` (frozen store vs CADAC rtol 1e-12; zero-error still computes gains; radix clamp; SMALL dmde/dndr; CADAC sign 0→+1; execute pass).

## 0.58.0 - F-16 second-order actuators
- Added `cadac.vehicles.plane6.actuator.Plane6Actuator` (`name="actuator"`). `define` registers C++ `def_actuator` (mact/dlimx/ddlimx/wnact/zetact, delax/delex/delrx, DXD/DX/DDXD/DDX). Does not define delacx/delecx/delrcx. DXD/DX/DDXD/DDX are vec (3,). `initialize` is pass (no C++ `init_actuator`).
- `execute` ports `Plane::actuator`: mact 0 copies ACTCX then position-limits with CADAC sign (`<0 → -1`, else `+1`); 2 calls `actuator_scnd`; else ValueError (C++ case 1 not implemented). Local `_sign`; no `np.sign`; no `flat6._cadac_sign`. `dt=ctx.int_step`. Protocol `vehicle.store`. No control/forces/vehicle. Cruise3/HYPER3/PLANE untouched.
- `actuator_scnd` all deg: position stop zeros same-sign rate; rate limit sets iflag; CADAC `integrate` of DX then DDX; `DDXD_NEW=wnact*wnact*edx-2.*zetact*wnact*DXD`; iflag zeros DDXD if same-sign. Returns DX as ACTX.
- Tests: `Python/tests/unit/test_plane6_actuator.py` (input_gamma mact=2 delecx=1 dt=0.001 lags and |delex|<=dlimx; full vs CADAC rtol 1e-12; mact 0 limit; mact 1/other raise; position/rate stops; CADAC sign 0→+1; stored-slope).

## 0.57.0 - F-16 propulsion
- Added `cadac.vehicles.plane6.propulsion.Plane6Propulsion` (`name="propulsion"`). Constructor takes Datadeck. `define` registers C++ `def_propulsion` (mprop/vmachcom/throttle/gmach, thrustf/thrust/thrust_req, mfreeze_prop, powerd/power, power_com/tpower, idle/mil/max). Does not define vmach/pdynmc/alphax/refa/cdrag/hbe/time/mfreeze. No fuel-flow fields; C++ neglects mass change and never looks up `ff_vs_thrust_alt_mach`. FOOT=3.280834 and NT=4.448 are module-level, not in `cadac.constants`.
- `initialize` is pass (no C++ `init_propulsion`; power/powerd start 0 from define). `execute` ports `Plane::propulsion`: mprop 1 manual throttle; 2 Mach hold `throttle=clip(gmach*(vmachcom-vmach),[0,0.77])` plus `thrust_req=cdrag*pdynmc*refa/cos(alphax*RAD)`; else thrust=0 without integrating power. `propulsion_thrust` power lag via CADAC `integrate`; 2D idle/mil/max look_up(vmach, hbe*FOOT)*NT; blend at power 50. Skip mfreeze latch if `mfreeze` not on the store. Protocol `vehicle.store`. No actuator/control/forces/vehicle. Cruise3/HYPER3/PLANE untouched.
- Tests: `Python/tests/unit/test_plane6_propulsion.py` (parsed `f16_prop_deck.asc`; mprop=2 vmach=0.58 throttle in (0,1]; full vs CADAC rtol 1e-12; clips; afterburner; power blend; mprop 0/else no integrate; stored-slope; mfreeze skip/latch; FOOT/NT).

## 0.56.0 - F-16 aero derivatives
- `Plane6Aero.execute` now calls `aerodynamics_der()` at the end, as C++ `Plane::aerodynamics`. Finite-diff cz/cm/cn ±1.5 deg; local `cla=-cza` and `cmde` not stored (store `cla`/`cmde` stay 0). Stored `cma`/`clnb` include CG shift. Dimensional pitch/lateral/roll derivs and pitch/yaw rigid-mode roots as C++. `stmarg=-cma/cla` if local `cla`. Uses stored `clde`/`cyb`/`cydr`/`cllda`/`cllp`/`cmq`/`clndr`/`clnr` from `aerodynamics()`.
- Tests: `Python/tests/unit/test_plane6_aero_der.py` (Task 7 FC finite; `dla=duml*(-cza)`, `dma=dumm*cma/RAD`; full der vs CADAC rtol 1e-12; CG shift; store `cla`/`cmde` stay 0). Task 7 `test_unassigned_table_locals_stay_zero` keeps aero-unassigned zeros, no longer asserts der fields stay 0. Cruise3/HYPER3/PLANE untouched. No propulsion/vehicle.

## 0.55.0 - F-16 aerodynamics tables
- Added `cadac.vehicles.plane6.aero.Plane6Aero` (`name="aerodynamics"`). Constructor takes Datadeck. `define` registers C++ `def_aerodynamics` (refa/refb/refc, force/moment coeffs, prepared derivs, gmax/gminx, termination, vmass/IBBB/eng_ang_mom, xcg/xcgr). Does not define time/alphax/betax/vmach/pdynmc/dvba/ppx/qqx/rrx/delax/delex/delrx.
- `initialize` ports `Plane::init_aerodynamics`: refa=27.87, refb=9.14, refc=3.45, vmass=9496, IBBB diag 12875/75673/85551 with Ixz=Izx=-1331.4, eng_ang_mom=70000, trmach=0.8, trdynm=10e3, trload=3, tralppx=21, tralpnx=-6, trbetx=5, trcode=0, tmcode=0. xcg/xcgr/alplimpx/alplimnx stay define zeros.
- `execute` ports `Plane::aerodynamics` (not `_der`): parsed `f16_aero_deck.asc`; cxt/cyt/czt/clt/cmt/cnt as C++; clt uses local cllr=0 (clr look_up unused); clda sign flip; gmax/gminx from cz at alplimpx/alplimnx; cdrag/clift/clovercd; prepared derivs; unassigned locals stored 0; diagnostic `cl` is rolling-moment look_up; trcode=4 if gmax<trload. Protocol `vehicle.store`. No JSONC under cases/. Cruise3/HYPER3/PLANE untouched.
- Tests: `Python/tests/unit/test_plane6_aero.py` (alpha=1 elev=0 cxt/czt; full coeffs vs CADAC formulas rtol 1e-12; cllr not clr; cl vs clift; trcode).

## 0.54.0 - Flat6 newton
- Added `cadac.eom.flat6.Flat6Newton` (`name="newton"`) in the same `flat6.py` as environment/kinematics/euler. `define` registers C++ `def_newton` (time exec scrn/plot/com, halt, VBEBD/VBEB/SBELD/SBEL state, sbel1/2/3, SBELM, groundrange, FSPB, VBEL, dvbe in/out plot, alpha0x/beta0x, hbe, psivlx/thtvlx, alx/anx/ayx, ATB, mfreeze_newt/dvbef). Does not define TBL (kinematics), grav (environment), WBEB (euler), FAPB (forces), vmass/mfreeze (plane).
- `initialize` ports `Flat6::init_newton`: VBEB=[calp*cbet, sbet, salp*cbet]*dvbe with alpha0x/beta0x in degrees * RAD; VBEL=TBL.T@VBEB; flight path psivl=0 if vbel1=vbel2=0 else atan2, thtvl=atan2(-vbel3, hypot); SBEL from sbel1/2/3; hbe=-SBEL[2]; SBELM=SBEL. Does not write VBEBD/SBELD/time.
- `execute` ports `Flat6::newton`: time=ctx.sim_time; ATB=skew(WBEB)@VBEB; FSPB=FAPB/vmass; VBEBD_NEW=FSPB-ATB+TBL@GRAVL; stored-slope `integrate` of VBEB then SBEL (slope VBEL=TBL.T@VBEB); dvbe=||VBEL||; hbe=-SBEL[2]; anx/ayx/alx (mat2tr); groundrange with DELSBEL vertical zeroed. Skip mfreeze latch if `mfreeze` not on the store. `dt=ctx.int_step`. No aero/vehicle. Cruise3/HYPER3/PLANE untouched.
- Tests: `Python/tests/unit/test_flat6_newton.py` (sbel3=-1000, dvbe=180, dt=0.001: init hbe==1000, one step near 1000; TBL transpose; stored-slope; ATB skew; groundrange; mfreeze skip/latch).

## 0.53.0 - Flat6 Euler equations
- Added `cadac.eom.flat6.Flat6Euler` (`name="euler"`) in the same `flat6.py` as environment/kinematics. `define` registers C++ `def_euler` (ppx/qqx/rrx init/out plot, WBEB/WBEBD state). Does not define IBBB/eng_ang_mom (plane) or FMB (forces).
- `initialize` ports `Flat6::init_euler`: `WBEB=[ppx,qqx,rrx]*RAD` (store rates in deg/s). Does not write WBEBD.
- `execute` ports `Flat6::euler`: `L_ENGINE=[eng_ang_mom,0,0]`; `WACC_NEXT=inv(IBBB)@(FMB-skew(WBEB)@(IBBB@WBEB+L_ENGINE))` with CADAC skew (row-major cross-product matrix); stored-slope `integrate`; `WBEBD=WACC_NEXT`; `ppx,qqx,rrx=WBEB*DEG`. `dt=ctx.int_step`. `np.linalg.inv` for 3x3 IBBB. No scipy. No newton. Cruise3/HYPER3/PLANE untouched.
- Tests: `Python/tests/unit/test_flat6_euler.py` (zero FMB, F-16 IBBB, ppx=10, dt=0.001 vs C++ replica; gyroscopic qqx change; spherical I holds rates; stored-slope second step).

## 0.52.0 - Flat6 kinematics
- Added `cadac.eom.flat6.Flat6Kinematics` (`name="kinematics"`) in the same `flat6.py` as environment. `define` registers C++ `def_kinematics` (ck default 50, quaternion states, TBL/TLB, Euler, alphax/betax/alpp/phip, erq/etbl). Does not define WBEB/dvba/VBAL (euler/environment) or plane trcode/tralppx/tralpnx/trbetx.
- `initialize` ports `Flat6::init_kinematics`: half-angle quats with `sin(psiblx/(2.*DEG))` etc; `TBL=mat3tr(psiblx/DEG, thtblx/DEG, phiblx/DEG)`. Euler in degrees on store. Does not write alphax.
- `execute` ports `Flat6::kinematics`: stored-slope `integrate` of four quaternion derivatives (ck*erq orthogonalizing); TBL from nine quat assigns starting zeros; etbl from diagonal of TLB@TBL (not C++ `UBL[0]` which assigns `num_col=1`); Euler extract with |tbl13|<1 vs `PI/2*sign`, cpsi/cphi clamp to `(1-EPS)*sign`; CADAC sign `<0 → -1` else `+1`; VBAB=TBL@VBAL, alpha=atan2, beta=asin, alpp/phip as C++ (EPS, PI). Skip plane trcode termination if those names are absent. No euler/newton. No RK4.
- Tests: `Python/tests/unit/test_flat6_kinematics.py` (thtblx=1, VBAL=TBL.T@VBEB from alpha0x=1, beta0x=0, dvbe=180, WBEB=0 → TBL finite, alphax≈1; stored-slope; etbl diagonal; trcode skip/5/6/7). Cruise3/HYPER3/PLANE untouched.

## 0.51.0 - Flat6 environment
- Added `cadac.eom.flat6.Flat6Environment` (`name="environment"`). `define` registers C++ `def_environment` mwind==0 outputs (mwind, press, rho, vsound, grav, vmach, pdynmc, tempk, VAEL, dvba, VBAL). Does not define hbe/VBEL (newton) or plane mfreeze/mguid/trcode. `execute` ports FALCON6 `Flat6::environment` mwind==0: US76+gravity(hbe), VAEL=0, VBAL=VBEL, dvba=||VBEL||, vmach, pdynmc. mwind!=0 ValueError. No ISO62. No kinematics/euler/newton.
- Tests: `Python/tests/unit/test_flat6_environment.py` (hbe=1000, ||VBEL||=180, mwind=0 vs atmosphere76/gravity; vmach not mach; mwind 1/2 raises). Cruise3/HYPER3/PLANE untouched.

## 0.50.0 - CADAC mat3tr Euler DCM
- Added `mat3tr(psi, tht, phi)` to `cadac.math.frames` next to `mat2tr`: FALCON6/HYPER3 `utility_functions.cpp` element-by-element (zeros then all nine `assign_loc`). Angles in radians. Not a scipy/ZYX helper. `cadtbv` unchanged.
- Tests: `Python/tests/unit/test_mat3tr.py` (identity at 0,0,0; `mat3tr(0,0,0.1)[1,2]==cos(0)*sin(0.1)`; all nine vs C++). Cruise3/HYPER3/PLANE numerics untouched. No Flat6 EOM. lookup.py untouched.

## 0.49.1 - FALCON5 e2e golden skip gate
- Added `Python/tests/e2e/test_falcon5_turning.py`: `pytest.skip` if `tests/e2e/goldens/falcon5/plot.csv` is absent (file not created).
- If golden exists: `run_scenario` on `Python/cases/falcon5/input_turning_to_IP.jsonc`; compare `alt` at t=0 and plot columns present in both (PLANE plot-flagged names, not HYPER3 23 columns). CSV rtol=1e-5, atol=max(1e-6, 5e-6*|g|); skip sentinel time=-1.
- Covering tests: skip helper, sentinel, t=0 alt tolerances, shared-column intersection. Cruise3/HYPER3 unchanged.

## 0.49.0 - FALCON5 PLANE from JSONC
- Added `cadac.vehicles.plane5.vehicle.Plane5` (`type="PLANE"`, health=1). Modules in turning-to-IP ASC order: environment, kinematics, aerodynamics, propulsion, guidance, control, forces, newton, intercept.
- TBV skip-if-exists on `Plane5Control` and `Flat3Newton` (Cruise3 FSPV pattern). Control owns the Field; both still `set` TBV in execute.
- `run_scenario` type map `PLANE -> Plane5` (CRUISE3 kept). Require aero_deck and prop_deck for PLANE. Unknown types still ValueError.
- `plot_row(store, columns=None)` defaults to HYPER3 `PLOT_COLUMNS`. PLANE uses plot-flagged fields (vecs as `NAME1/2/3`); `time` and `alt` in plot_rows. CRUISE3/HYPER3 e2e unchanged.
- Translated `input_turning_to_IP.asc` + Falcon5 aero/prop decks to `Python/cases/falcon5/` (`end_time` 160, type `PLANE`).
- Tests: `Python/tests/unit/test_plane5_one_step.py` (1.0 s: time≈1, alt finite heading toward 3000; plot columns; committed 160 s case). No FALCON6. No sys.exit.

## 0.48.0 - FALCON5 intercept stop_run
- Added `cadac.vehicles.plane5.intercept.Plane5Intercept` (`name="intercept"`). `define` registers C++ `def_intercept` `stop_run` only (int, data, default 0). Does not define write/mguidance/wp_flag/SWBL/time/psivlx/thtvlx.
- `execute` ports FALCON5 `Plane::intercept`: if write, then mguidance 30 or 40 with wp_flag==-1 computes horizontal miss and clears write; mguidance 33 with wp_flag==-1 computes 3D miss and clears write. If stop_run==1 under those triggers: `vehicle.health=0` and `ctx.combus[ctx.vehicle_slot].status=0` (C++ `exit(1)`; no sys.exit). stop_run==0 still clears write, does not kill. Always writes `write` back. Skip cout.
- Tests: `Python/tests/unit/test_plane5_intercept.py` (SimContext combus Packet status=1; 30/40/33 stop; stop_run 0/2 no kill; write=0 and wp_flag!=-1 and other mguidance no stop). Cruise3/HYPER3 untouched. No vehicle.py.

## 0.47.0 - FALCON5 line guidance
- `Plane5Guidance.define` adds C++ `def_guidance` line fields: line_gain, nl_gain_fact (default 1), decrement, psiflx, thtflx (data); nl_gain, VBEF (dia). Point fields unchanged.
- `guidance_line(vehicle)` ports FALCON5 `Plane::guidance_line`: TFL=mat2tr(psiflx*RAD, thtflx*RAD); SWBL=SWEL-SBEL; polar/mat2tr LOS as point; VBEO=TOL@VBEL; VBEF=TFL@VBEL; nl_gain=nl_gain_fact*(1-exp(-wp_sltrange/decrement)) unsimplified; algv1=grav*sin(thtvlx*RAD); algv2=line_gain*(-vbeo2+nl_gain*vbef2); algv3=line_gain*(-vbeo3+nl_gain*vbef3)-grav*cos(thtvlx*RAD). Same wp_flag/write/rad_min as point (CADAC sign). Writes write, wp_sltrange, nl_gain, VBEO, VBEF, wp_grdrange, SWBL, rad_min, wp_flag; returns ALGV.
- `execute` keeps 0 and 40; adds 30 (ALGV=guidance_line(), alcomx=ALGV[1]/grav, ancomx stays 0) and 33 (also ancomx=-ALGV[2]/grav). Then same clips and phicx/ancomx/alcomx writes. Other mguidance still ValueError (no 3/43). Cruise3/HYPER3 untouched.
- Tests: `Python/tests/unit/test_plane5_guidance_line.py` (turning-to-IP line ICs line_gain=1.5, nl_gain_fact=0.4, decrement=800, psiflx=180, thtflx=-30 for 33; one-step vs C++ replica; nl_gain exponential not constant; 30/33 clips; wp_flag; unknown 3/43/99). Point unknown list no longer includes 30/33.

## 0.46.0 - FALCON5 point guidance
- Added `cadac.vehicles.plane5.guidance.Plane5Guidance` (`name="guidance"`). `define` registers C++ `def_guidance` fields used by point + dispatcher (mguidance, swel1/2/3, point_gain, wp_sltrange, VBEO, wp_grdrange, SWBL, rad_min, write, wp_flag). Does not register line_gain/nl_gain_fact/decrement/psiflx/thtflx/nl_gain/VBEF.
- `guidance_point(vehicle)` ports FALCON5 `Plane::guidance_point`: SWEL-SBEL, `polar_from_cart`/`mat2tr` LOS TM, wp_grdrange=hypot, VBEO=TOL@VBEL, APGV steering with `point_gain` and gravity terms, rad_min=dvbe**2/(grav*tan(philimx*RAD)), wp_flag CADAC sign(VH·SH) inside 2*rad_min else 0 (sign never 0). Writes write, wp_sltrange, VBEO, wp_grdrange, SWBL, rad_min, wp_flag; returns APGV.
- `execute`: mguidance==0 local zeros and return without writing (mprop==0 pattern); ==40 APGV=guidance_point(), alcomx=APGV[1]/grav, clip ancomx/alcomx, write phicx (unchanged), ancomx, alcomx; other mguidance ValueError (Task 14 adds 30/33). Cruise3/HYPER3 untouched.
- Tests: `Python/tests/unit/test_plane5_guidance_point.py` (swel=[5000,2000] origin heading 0 finite commands; wp_flag 0 outside, +1 closing, -1 fleeting; CADAC sign zero-dot +1; mguidance 0 no write; unknown raises; one-step vs C++ replica; clip; write latch). Plant SBEL/VBEL/grav/thtvlx/philimx plus control ancomx/alcomx/limits/phicx registered by tests.

## 0.45.0 - FALCON5 mcontrol dispatcher
- `Plane5Control.define` adds lateral/dispatcher fields from C++ `def_control` used by `control_lateral` / `Plane::control` (mcontrol int, TBV 3x3 zeros mat out, alcomx, allimx, gcp, alx). Heading/altitude/load/bank fields remain.
- `control_lateral(vehicle, alcomx)` ports FALCON5 `Plane::control_lateral`: TBV=`cadtbv(phimvx*RAD, alphax*RAD)`, FSPB=TBV@FSPV, clip alcomx to ±allimx, sign=1 if anx>=0 else -1 (anx=-FSPB[2]/grav), phic=gcp*sign/(fabs(anx)+.001)*alcomx, phicx=phic*DEG, alx=FSPV[1]/grav. Writes alx; returns phicx; does not write phicx/TBV.
- `execute` is the mcontrol dispatcher for turning-to-IP modes only: 46 lateral+bank+altitude+load; 44 lateral+bank+load with store ancomx. Then TBV=`cadtbv(phimvx*RAD, alphax*RAD)`; writes phicx, TBV, alphax, phimvx, ancomx. Any other mcontrol (0, 3, 6, 16, …) raises ValueError. Cruise3/HYPER3 untouched.
- Tests: `Python/tests/unit/test_plane5_mcontrol.py` (turning_to_IP alcomx=0.5, gcp=2, allimx=1; mcontrol=46 finite ancomx/phimvx; unknown raises; one-step vs C++ replica; clip; negative anx sign; `.001`; 44 no altitude). Bank/load/altitude/heading execute-as-bank-wrap tests now call `control_bank`. Plant FSPV/grav plus chained-controller fields registered by tests.

## 0.44.0 - FALCON5 heading and flight-path control
- `Plane5Control.define` adds heading/FPA fields from C++ `def_control` used by `control_heading`/`control_flightpath` (gain_thtvg, gain_psivg, psivlcx, thtvgcx, avx). alphax/anx/alpposlimx/alpneglimx already from Task 9. Does not register mcontrol/TBV/alcomx/lateral.
- `control_heading(vehicle, psivlcx)` ports FALCON5 `Plane::control_heading`: south singularity if fabs(psivlcx)<=135 then psivgx_comp=psivlx; else if same sign: psivlx; else wrap 360-psivlx*sign. psivlx is degrees. Returns bank command; does not write phimvx/phicx. Gains from mcontrol_11: gain_psivg=12.0.
- `control_flightpath(vehicle, thtvgcx, phimvx)` ports FALCON5 `Plane::control_flightpath`: avx=gain_thtvg*(thtvgcx*RAD-thtvl); anx=avx/cos(phimvx*RAD); alphax=anx*mass*grav/(pdynmc*area*cla); clip alpposlimx/alpneglimx. thtvl is radians; thtvgcx is degrees. Writes anx,avx; returns alphax; does not write alphax. gain_thtvg=30. `execute` remains the bank wrap.
- Tests: `Python/tests/unit/test_plane5_control_heading.py` (mcontrol_11 gains; one-step vs C++ replica; |psivlcx|>135 opposite-sign wrap; same-sign no wrap; alphax clip; banked RAD; thtvl radians; execute still bank-only). Plant psivlx/thtvl/pdynmc/grav/mass/area/cla registered by tests.

## 0.43.0 - FALCON5 altitude control
- `Plane5Control.define` adds altitude-hold fields from C++ `def_control` used by `control_altitude` (altdlim, gh, gv, altd, altcom). anposlimx/anneglimx already from Task 9. Does not register mcontrol/TBV/alcomx/heading/lateral.
- `control_altitude(vehicle, altcom, phimvx)` ports FALCON5 `Plane::control_altitude`: ealt=gh*(altcom-alt) clipped to ±altdlim; altd=-VBEL[2]; ancomx=(gv*(ealt-altd)/grav+1)*(1/cos(phimvx*RAD)) with cadac.constants.RAD; clip to [anneglimx, anposlimx]. Writes altd; returns ancomx; does not write ancomx. No 1/cos guard at 90° bank. `execute` remains the bank wrap.
- Tests: `Python/tests/unit/test_plane5_control_altitude.py` (turning_to_IP gh=0.3, gv=1.0, altdlim=50, altcom=3000, anposlimx=3, anneglimx=-1; one-step vs C++ replica; rate-limiter |gh*(altcom-alt)|>altdlim; banked RAD; no ancomx write; execute still bank-only). Plant alt/grav/VBEL registered by tests.

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
