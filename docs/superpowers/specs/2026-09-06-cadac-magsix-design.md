# CADAC MAGSIX Rotor (DNU Magnus) — design

Date: 2026-09-06
Status: approved (chat). Rotor EOM slice. Depends on kernel + deck-optional factory (HYPER5/HYPER6). Family API is added here if missing.

Parent: `docs/superpowers/specs/2026-09-04-cadac-python-design.md` — EOM table: **Rotor is reserved, not a Flat6 flag.**

## Goal

Add unique EOM layer `Rotor` (class `Rotor : Cadac` in C++) and JSONC type `ROTOR` with `family="magsix"` on translated vehicles. First case: MAGSIX `input.asc` (attitude RECT.MR1, `ENDTIME 0.35`, `int_step 0.0001`, `nonlinear=0`). Second case: `input_trajectoryMR1.asc` (trajectory-only, `ENDTIME 50`, `int_step 0.001`). Numerics regression-close to `CADAC_Simulations/MAGSIX_231111/MAGSIX/`. No aero/prop decks — coefficients live in `params`.

## Non-goals

Wind `mwind!=0`. Multi-run `input_multi_trajectoryMR1.asc`. Markov / Monte Carlo. `sys.exit` / `system("pause")`. Reusing `Flat6*` / `Flat3*` / `Round6*` / `Round3*` classes. Adding `RPM` / `RHO_SL` to locked `cadac.constants`. C++ leftover `eng_ang_mom=rotor[213]` (that slot is `cnr3`; unused — do not port). Kernel `terminate` dispatch (C++ `term_trajectory` is console-only; Python `terminate` is a no-op).

## Layout

```
Python/src/cadac/eom/rotor.py              # RotorEnvironment, RotorTrajectory, RotorAttitude
Python/src/cadac/vehicles/planar/magsix/
  __init__.py
  vehicle.py                               # Rotor type="ROTOR"; composes the three EOM modules
Python/cases/magsix/                       # translated input.asc + input_trajectoryMR1.asc
Python/tests/unit/test_rotor_*.py
Python/tests/e2e/test_magsix_attitude.py
Python/tests/e2e/test_magsix_trajectory.py
```

Replace reserved `eom/rotor.py` docstring stub if present. Do not reuse Flat6/Flat3/Round6 classes. Shared kernel only: `integrate`, `atmosphere76`, `gravity`, `mat2tr`.

## Family and JSONC type

Type `ROTOR` is unique (not a Flat6 flag). Still set `family="magsix"` on translated MAGSIX vehicles.

Register **both**:

- `_VEHICLE_TYPES["ROTOR"] = Rotor`
- `_VEHICLE_FAMILIES[("magsix", "ROTOR")] = Rotor`

Factory lookup (AIM5 scenario-family loader — **keep / add idempotently**; do not invent a second loader):

- `VehicleSpec.family` optional (`None` if omitted).
- `load_scenario`: vehicle `"family"` else scenario-level `"family"` else `None`. Vehicle key wins.
- `family` set → **only** `_VEHICLE_FAMILIES[(family, type)]` (do not fall back to `_VEHICLE_TYPES`).
- `family is None` → `_VEHICLE_TYPES[type]`.
- Unknown → `ValueError` that includes the type token and, when set, the family token.
- `translate_scenario_asc(src, dst_dir, family=None)`. When `family` is a string, stamp it on every vehicle object; when `None`, omit the key (existing cases unchanged).

Constructor: `Rotor(name, events=None)` — no decks. Add `"ROTOR"` to `_NO_DECK_TYPES`.

When registering `ROTOR`, **always** retarget unknown-type tests that still use `"AIM5"` or `"ROTOR"` as the unregistered token to `"NO_SUCH_TYPE"` (AIM5 now uses `NO_SUCH_TYPE` too). Add `test_unknown_type_no_such_type_still_raises`. Do not leave `"AIM5"` as the sentinel. Do not overwrite `tests/unit/test_vehicle_family.py`.

## DNU time (MAGSIX-specific)

C++ MAGSIX executive `sim_time`, `int_step`, `plot_step`, `ENDTIME` are **dynamic normalized time (DNT)**, not seconds. Translate those numbers as-is into JSONC (`end_time` 0.35 / 50, `int_step` 0.0001 / 0.001).

`RotorTrajectory` interprets `ctx.sim_time` / `ctx.int_step` as DNT:

- DNU states (`velocityx`, `gamma`, `omegax`, attitude angles) integrate with `dt = ctx.int_step`.
- Position `SBEL` integrates with `dt = ctx.int_step * tau` (seconds).
- Store `sim_time` = `ctx.sim_time` (DNT). **Plot-flag** `sim_time` (and `time`) so e2e can align on `sim_time`.
- Store `time` = `tau * ctx.sim_time` (C++ uses the **current** `tau`, not an integral of `d(tau)`). Plot sampling stays on executive DNT (`ctx.sim_time`), matching C++ `plot_step`.

`RPM = 9.5493` and `RHO_SL = 1.225` are module-level in `eom/rotor.py` from MAGSIX `global_constants.hpp`. Do not add them to `cadac.constants`. Use existing `AGRAV`, `RAD`, `DEG`, `G`, `EARTH_MASS`, `REARTH`, `R`.

## Rotor EOM

Port `MAGSIX/{environment,trajectory,attitude}.cpp`. Atmosphere is **US76** (`atmosphere76(hbe)`), gravity `gravity(hbe)` = `G*EARTH_MASS/(REARTH+hbe)**2`. Not ISO 62. Not WGS84 / Round6 `GRAVG`.

### Environment (`RotorEnvironment.name=="environment"`)

Port `Rotor::def_environment` / `environment`. Reads `hbe`, `VBEL` (trajectory-owned). `mwind==0` only (both first cases): `VAEL=0`, `VBAL=VBEL`, `dvba=||VBAL||`, US76 at `hbe`, `vsound=sqrt(1.4*R*tempk)`, `vmach=abs(dvba/vsound)`, `pdynmc=0.5*rho*dvba*dvba`, `grav=gravity(hbe)`. Other `mwind` → `ValueError`. Define C++ wind data/state names (`dvae`, `VAELS`, …) so JSONC unknown-param checks stay honest; do not execute wind filters.

Does not define `hbe` / `VBEL` (trajectory). New class — do not subclass `Flat6Environment` even though `mwind==0` algebra is close.

### Trajectory (`RotorTrajectory.name=="trajectory"`)

Port `def_trajectory`, `init_trajectory`, `trajectory`. Aero/mass coefficients are **data fields**, not decks: `cd`, `cmdw`, `clw`, `cma`, `mass`, `ref_area`, `ref_length`, `moi_spin`.

**`moi_spin` vs `moi_spinx`:** C++ comments are swapped. Input `moi_spin` is physical spin inertia **kg·m²** (RECT.MR1 `0.004`). Execute computes DNU `moi_spinx = moi_spin / (ref_length**2 * mu**2 * mass)` and writes `moi_spinx` for attitude. Port the names and that conversion; do not “fix” the comments.

**Init (C++ `init_trajectory`):**

- `gamma_ss = atan(cd*cmdw/(clw*cma))` — Python **scalar rad** (C++ 3-arg `init` is a vector quirk; `.gets` writes a scalar).
- `velocity_ss = sqrt(2*AGRAV*mass*|sin(gamma_ss)|/(RHO_SL*ref_area*cd))` — sea-level `RHO_SL`, not local `rho`.
- `omega_ss = -velocity_ss*cma/(ref_length*cmdw)`.
- `SBEL = (sbel1, sbel2, -hbe)` — ignore `sbel3` as C++.
- `TVL = mat2tr(psivlx*RAD, thtvlx*RAD)`; `VBEL = TVL.T @ (dvbe, 0, 0)`.
- `velocityx = dvbe/velocity_ss`; `gamma = thtvlx*RAD`.
- `tau = 2*mass/(rho*ref_area*velocity_ss)` with `rho` from `atmosphere76(hbe)` at launch.
- `omegax = (omega_rpm/RPM)*tau`.
- No console I/O.

**Execute (C++ Table 12.1 eqs. 1–3), order as C++:**

```
tau = 2*mass/(rho*ref_area*velocity_ss)
mu  = 2*mass/(rho*ref_area*ref_length)
moi_spinx = moi_spin/(ref_length**2 * mu**2 * mass)

velocityxd_new = -cd*velocityx**2 - tau*grav*sin(gamma)/velocity_ss
gammaxd_new    = clw*omegax/mu - tau*grav*cos(gamma)/(velocity_ss*velocityx)
omegaxd_new    = cma*velocityx**2/(mu*moi_spinx) + cmdw*velocityx*omegax/(mu**2*moi_spinx)
```

Stored-slope `integrate` each, then `dvbe=velocityx*velocity_ss`, `thtvlx=gamma*DEG`, `omega=omegax/tau`, `omega_rpm=omega*RPM`. Rebuild `VBEL` from `mat2tr(psivlx*RAD, thtvlx*RAD)`. `SBEL = integrate(VBEL, SBELD, SBEL, int_step*tau)`. `hbe=-SBEL[2]`. `tpsp_ratio=omega*ref_length/dvbe`. `time=tau*sim_time`.

**Ground impact:** `hbe < hbg` → `vehicle.health = 0` and `ctx.combus[slot].status = 0`. No `sys.exit`. Trajectory-only RECT.MR1 at `ENDTIME` 50 DNT reaches the ground (C++ marks the packet dead and keeps the loop).

`terminate`: no-op (no `sys.exit`).

### Attitude (`RotorAttitude.name=="attitude"`)

Port `def_attitude`, `init_attitude`, `attitude`. First case `nonlinear=0`. Trajectory case omits this module from JSONC `modules` (uncoupled). The vehicle still **composes and defines** attitude (zeros / defaults); `run_loop` skips init/exec when the name is not in the scenario list. Plot rows may include attitude names (`phix`, …) at those defaults — do not require them absent.

CADAC execute order (port exactly): integrate `beta` first; `phidd_new` uses the **new** `beta`; then roll (`phidd`→`phid`, then `phi` with equal slopes); `psidd_new` uses the **post-update** `phid`; then yaw.

**Init:** `beta=betax*RAD`, `phi=phix*RAD`, `phid=ppx*RAD*tau`, `psi=psix*RAD`, `psid=rrx*RAD*tau`. `ppx` defaults 0 (absent from `input.asc`).

**Execute:** `moi_transx = moi_trans/(ref_length**2 * mu**2 * mass)`. `nonlinear in {0, 1}` only; else `ValueError`. C++ adds `+ nonlinear * (cubic terms)` — port both branches. Cubic data default 0 (`cyb3`, `clwb3`, `clp3`, `clwb2p`, `clwbp2`, `cnb3`, `cnr3`, `cnb2r`, `cnbr2`).

Sideslip: stored-slope integrate of `betad_new` (C++ eq. 4).

Roll/yaw **second-derivative then angle** as C++ (Table 12.1 eqs. 5–8), including the CADAC update: after integrating `phidd` into `phid`, `phi` is integrated with **equal new/old slopes** (`phid_new == phid` post-update ⇒ `phi += phid * int_step`). Same for `psi`. Do not substitute a “correct” trapezoid on the pre-update rate.

Outputs: `betax=beta*DEG`, `phix=phi*DEG`, `ppx=phid*DEG/tau`, `psix=psi*DEG`, `rrx=psid*DEG/tau`.

Do not read `rotor[213]` as angular momentum.

## First case (attitude RECT.MR1)

Translate `CADAC_Simulations/MAGSIX_231111/MAGSIX/input.asc` → `Python/cases/magsix/input.jsonc`. `family="magsix"`. No decks. `end_time` 0.35. `int_step` 0.0001. `plot_step` 0.005.

MODULES: `environment` def,exec; `trajectory` def,init,exec,term; `attitude` def,init,exec.

Exact ICs / params:

| name | value |
|---|---|
| sbel1, sbel2 | 0 |
| hbe | 1000 |
| hbg | 0 |
| dvbe | 16.6 |
| psivlx | 0 |
| thtvlx | -77 |
| omega_rpm | 850 |
| nonlinear | 0 |
| betax | 0 |
| phix | 3 |
| psix | 0 |
| rrx | -40 |
| mass | 1.5 |
| moi_spin | 0.004 |
| moi_trans | 0.0268 |
| ref_area | 0.0468 |
| ref_length | 0.0625 |
| cd | 1.31 |
| cmdw | -0.45 |
| clw | 2.51 |
| cma | 0.508 |
| cyb | -3.82 |
| clwb | -0.357 |
| clp | -5.82 |
| cnb | -0.737 |
| cnr | -13.8 |

Smoke: short DNT run (`end_time` 0.01); `hbe` near 1000; `phix` finite.

## Second case (trajectory RECT.MR1)

Translate `input_trajectoryMR1.asc` → `Python/cases/magsix/input_trajectoryMR1.jsonc`. `family="magsix"`. MODULES: `environment`, `trajectory` only (attitude still defined on the vehicle, not executed). `end_time` 50. `int_step` 0.001. `plot_step` 0.01. Params: same trajectory subset (no `nonlinear` / attitude coeffs / `moi_trans`). Smoke: `end_time` 0.1 DNT so the rotor has not hit `hbg`; `hbe` near 1000. Do not assert `phix` absent.

## Testing

Unit: `rtol=1e-12` vs CADAC formulas. TDD each task. Files `test_rotor_*.py`.

E2E attitude: `tests/e2e/test_magsix_attitude.py`. Skip if `tests/e2e/goldens/magsix/plot.csv` absent. Align rows on `sim_time` when both CSVs have it, else `time`. Sentinel `time=-1`. Compare **all** shared plot-flagged columns (not only `hbe`/`dvbe`). Require `hbe`, `dvbe`, and `sim_time` in the shared set (MAGSIX has no `alt`). Translated attitude case locks `plot_step` 0.005. `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`.

E2E trajectory: `tests/e2e/test_magsix_trajectory.py`. Skip if `tests/e2e/goldens/magsix/trajectory/plot.csv` absent. Same tolerances and all-shared-column compare. Translated trajectory case locks `plot_step` 0.01.

Regression: HYPER3 e2e, HYPER5/HYPER6 units, FALCON5/FALCON6 units. Unknown `"NO_SUCH_TYPE"` still raises.

## Process

Grok `cursor-grok-4.6-high` implementer + reviewer per task; whole-plan review; no Fast/Kimi; controller does not patch physics.

## Python style

Named state; CADAC numeric order; same module name `environment` binds Rotor vs Flat6/Round6 by vehicle type.
