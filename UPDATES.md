# Updates

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
