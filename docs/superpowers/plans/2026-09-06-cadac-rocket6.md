# CADAC ROCKET6 Hyper (Round6 SLV) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add family-dispatched JSONC type `("rocket6","HYPER6")` Rocket6 SLV; unit-test each C++ module used by `input.asc`; e2e vs CADAC CSV when a golden exists.

**Architecture:** Reuse `cadac.eom.round6` + `cadac.math.wgs84`. New `cadac.vehicles.rocket6` ports `CADAC_Simulations/ROCKET6_250122/ROCKET6/*.cpp` Hyper modules. Do not import `cadac.vehicles.hyper6.*`. Never overwrite `_VEHICLE_TYPES["HYPER6"]`.

**Tech Stack:** Python >= 3.11, numpy, pytest. Work in `Python/`.

**Spec:** `docs/superpowers/specs/2026-09-06-cadac-rocket6-design.md`

**Depends on:** HYPER6 plan complete (Round6 EOM + WGS84 + Hyper6 climb factory).

## Global Constraints

- Spec + parent design spec
- Do not regress HYPER3 e2e, HYPER5 units, FALCON5/FALCON6 units, HYPER6 climb units
- Reuse Round6 kinematics/euler/newton; patch environment only for `mair` 0 and 12
- Family set: `_VEHICLE_FAMILIES` only. Family None: `_VEHICLE_TYPES`. `VehicleSpec.family` is canonical (scenario default, vehicle wins). Do not wipe `_VEHICLE_FAMILIES`; `setdefault(("rocket6","HYPER6"), Rocket6)`. Never assign `_VEHICLE_TYPES["HYPER6"]`
- Do not retarget unknown-type tests (`AIM5`, `NO_SUCH_TYPE`)
- E2E golden if present is C++ with MONTE off and Dryden drive 0, not raw `input.asc` plot.csv
- No `cadac.vehicles.hyper6` imports
- No `sys.exit`; LTG unreachable end-state → `ValueError`
- Monte Carlo / Markov sampling out of scope: GAUSS→mean, MARKOV→0, RAYL→first, Dryden `gauss_value=0`, INS `gauss`/Cholesky draws 0
- JSONC first case `input.asc` Vandenberg `end_time` 190, `mair` 12
- E2E skip if `Python/tests/e2e/goldens/rocket6/plot.csv` absent
- Grok `cursor-grok-4.6-high` implementer + reviewer; TDD; no Fast/Kimi
- Unit rtol=1e-12, atol=1e-14; CSV e2e rtol=1e-5, atol=max(1e-6, 5e-6*|g|)
- CADAC sign `<0 → -1` else `+1` local; not `np.sign`

## File map

- Modify: `Python/src/cadac/io/scenario.py`, `Python/src/cadac/io/translate.py`, `Python/src/cadac/cli.py`, `Python/src/cadac/eom/round6.py`, `Python/src/cadac/math/wgs84.py`, `Python/src/cadac/math/frames.py`
- Create: `Python/src/cadac/vehicles/rocket6/{__init__,vehicle,aero,propulsion,gps,startrack,ins,guidance,control,rcs,tvc,forces,intercept}.py`
- Tests: `test_rocket6_*.py`, `test_wgs84.py` (kepler), `test_frames.py` or extend existing frame tests, `test_round6_environment.py` (mair=12)
- Case: `Python/cases/rocket6/` from `input.asc` + `aero_deck_SLV.asc` + `weather_deck_Wallops.asc`
- C++: `environment.cpp`, `aerodynamics.cpp`, `propulsion.cpp`, `tvc.cpp`, `rcs.cpp`, `control.cpp`, `forces.cpp`, `ins.cpp`, `gps.cpp`, `startrack.cpp`, `guidance.cpp`, `intercept.cpp`, `utility_functions.cpp` (`cad_kepler`, `cart_from_pol`, `angle`)

---

### Task 1: Family dispatch and weather/stoch translate

**Files:**
- Modify: `Python/src/cadac/io/scenario.py`, `Python/src/cadac/io/translate.py`, `Python/src/cadac/cli.py`
- Test: `Python/tests/unit/test_rocket6_family.py`, `Python/tests/translate/test_rocket6_asc.py`

**Interfaces:**
- Produces: `VehicleSpec.family: str | None` is the **canonical** resolved family. `VehicleSpec.weather_deck: Path | None = None`. JSONC `"family"` may sit on the scenario and/or a vehicle; `load_scenario` sets `VehicleSpec.family` from the vehicle key if present, else the scenario key, else `None` (vehicle wins). `translate_scenario_asc(src, dst_dir, family=None)` writes **scenario-level** `"family"` when the kwarg is set; do not stamp every vehicle unless that vehicle already had a family. Vehicle parser: `WEATHER_DECK` → `weather_deck`; `GAUSS name mean sigma` → `params[name]=mean`; `RAYL name first` → `params[name]=first`; `MARKOV name sigma bcor` → `params[name]=0`. Idempotent: create `cli._VEHICLE_FAMILIES` only if it does not already exist; never assign `_VEHICLE_FAMILIES = {}` over a live dict. `_build_vehicle`: if `spec.family is not None`, `cls = _VEHICLE_FAMILIES.get((spec.family, spec.type))` else `_VEHICLE_TYPES.get(spec.type)`. Miss → `ValueError` including path. Do not fall back. Do not register Rocket6 yet. Do not change `_VEHICLE_TYPES["HYPER6"]`. Do not retarget `AIM5` or `NO_SUCH_TYPE` tests.

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

from cadac.cli import _VEHICLE_TYPES, _build_vehicle
from cadac.io.scenario import VehicleSpec, load_scenario
from cadac.io.translate import translate_scenario_asc


def test_hyper6_without_family_stays_hyper6():
    assert "HYPER6" in _VEHICLE_TYPES
    assert _VEHICLE_TYPES["HYPER6"].__name__ == "Hyper6"


def test_family_unknown_does_not_use_type_table(tmp_path: Path):
    spec = VehicleSpec(
        type="HYPER6",
        name="SLV",
        aero_deck=None,
        prop_deck=None,
        params={},
        events=[],
        family="rocket6",
    )
    try:
        _build_vehicle(tmp_path / "x.jsonc", spec)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "rocket6" in str(exc)
        assert "HYPER6" in str(exc)


def test_translate_family_and_stoch(tmp_path: Path):
    src = tmp_path / "in.asc"
    src.write_text(
        "TITLE t\nOPTIONS y_plot n_csv\nMODULES\n\tkinematics\tdef,exec\nEND\n"
        "TIMING\n\tint_step 0.001\nEND\nVEHICLES 1\n"
        "\tHYPER6 SLV\n"
        "\t\tmair  012\n"
        "\t\tWEATHER_DECK  weather_deck_Wallops.asc\n"
        "\t\tAERO_DECK aero_deck_SLV.asc\n"
        "\t\tRAYL dvae  5\n"
        "\t\tGAUSS ucbias_error  0  3\n"
        "\t\tMARKOV pr1_noise  0.25  0.002\n"
        "\tEND\nENDTIME 190\nSTOP\n",
        encoding="utf-8",
        newline="\n",
    )
    translate_scenario_asc(src, tmp_path, family="rocket6")
    cfg = load_scenario(tmp_path / "in.jsonc")
    v = cfg.vehicles[0]
    assert v.type == "HYPER6"
    assert v.family == "rocket6"  # scenario default resolved onto VehicleSpec
    assert v.params["mair"] == 12
    assert v.params["dvae"] == 5
    assert v.params["ucbias_error"] == 0
    assert v.params["pr1_noise"] == 0
    assert v.weather_deck == tmp_path / "weather_deck_Wallops.jsonc"
    assert v.aero_deck == tmp_path / "aero_deck_SLV.jsonc"
    assert v.prop_deck is None


def test_vehicle_family_wins_over_scenario(tmp_path: Path):
    path = tmp_path / "both.jsonc"
    path.write_text(
        '{"title":"t","options":{},"modules":[],"timing":{"int_step":0.01},'
        '"end_time":0,"family":"scenario_default",'
        '"vehicles":[{"type":"HYPER6","name":"SLV","family":"rocket6","params":{}}]}',
        encoding="utf-8",
        newline="\n",
    )
    v = load_scenario(path).vehicles[0]
    assert v.family == "rocket6"


def test_scenario_family_used_when_vehicle_omits_it(tmp_path: Path):
    path = tmp_path / "scen.jsonc"
    path.write_text(
        '{"title":"t","options":{},"modules":[],"timing":{"int_step":0.01},'
        '"end_time":0,"family":"rocket6",'
        '"vehicles":[{"type":"HYPER6","name":"SLV","params":{}}]}',
        encoding="utf-8",
        newline="\n",
    )
    assert load_scenario(path).vehicles[0].family == "rocket6"
```

Also: `translate_scenario_asc(src, dst)` without `family` writes no `family` key (existing HYPER3 translate tests still pass). Pin that `test_hyper6_one_step.py` / `test_plane5_one_step.py` / `test_cruise3_one_step.py` still use `AIM5` (and any `NO_SUCH_TYPE` test stays `NO_SUCH_TYPE`).

- [ ] **Step 2:** FAIL `pytest tests/unit/test_rocket6_family.py tests/translate/test_rocket6_asc.py -v`
- [ ] **Step 3:** Add `family` / `weather_deck` with defaults so existing `VehicleSpec(...)` positional tests keep working. Parse stoch tokens before generic `params[token]=_parse_number(parts[1])`. If `_VEHICLE_FAMILIES` is already defined, leave it; do not assign a fresh `{}`.
- [ ] **Step 4:** PASS plus `pytest tests/unit/test_hyper6_one_step.py tests/unit/test_cruise3_one_step.py tests/translate/test_asc_scenario.py -q`
- [ ] **Step 5: Commit** `feat: CADAC family dispatch and weather-deck translate`

---

### Task 2: cad_kepler and polar helpers

**Files:**
- Modify: `Python/src/cadac/math/wgs84.py`, `Python/src/cadac/math/frames.py`
- Test: `Python/tests/unit/test_wgs84.py`, `Python/tests/unit/test_frames.py` (create if missing)

**Interfaces:**
- Produces: `cad_kepler(sbii, vbii, tgo) -> (spii, vpii, flag)` port ROCKET6 `utility_functions.cpp` Morth `cad_kepler` using module `GM` and `SMALL`. `flag==1` leaves outputs unused (return input copies + flag). Do not port `cad_kepler1`. `cart_from_pol(magnitude, azimuth, elevation) -> (3,)` as C++ `Matrix::cart_from_pol`. `angle(vec1, vec2)` as C++ `angle()` (`EPS` clamp).

- [ ] **Step 1:** Circular equatorial: `sbii=(REARTH+400e3,0,0)`, circular `vbii` with `v=sqrt(GM/r)`, `tgo` small → `flag==0`, `||spii||` near r, rtol 1e-12 on a replica of the C++ `fk`/`gk` formulas. `cart_from_pol(1,0,0)==[1,0,0]`. `angle([1,0,0],[0,1,0])==pi/2`.
- [ ] **Step 2–5:** implement, pass, commit `feat: CADAC kepler and polar helpers`

---

### Task 3: Round6 environment mair=12

**Files:** Modify `Python/src/cadac/eom/round6.py`; `Python/tests/unit/test_round6_environment.py`

**Interfaces:**
- Consumes: `atmosphere76`, `cad_grav84`, optional weather `Datadeck`
- Produces: `Round6Environment(weather_deck=None)`. Keep `initialize`: `dvba=store.get("dvbe")` as C++ `init_environment` (do not delete). `define` adds C++ wind/Dryden fields. `execute`: decode mair. `(0,0,0)` unchanged (works with `ctx is None`). `(0,1,2)`: US76; require a weather Datadeck else `ValueError`; `dvw=look_up("speed",alt)`, `psiwdx=look_up("direction",alt)`; smooth `VAEDS` as C++; Dryden with `gauss_value=0`; `VAED=VTAD+VAEDS`; `dvba=||VBED-VAED||`. Other mair → `ValueError`. Need `ctx.int_step` only when mwind/mturb nonzero.

- [ ] **Step 1:** Keep `test_mair0_us76_*` and `test_mair_100_raises`. Change `test_other_nonzero_mair_raises` to exclude 12; add `test_mair_12_tabular_wind_zero_dryden` with a 1D speed/direction deck, `twind=1`, `dt=0.01`, `TBD=I`, `alppx=phipx=0`, `turb_sigma=0` → `VAED` matches C++ smoother of table wind; `gauss_value` 0. `mair=10` still raises. Add `test_mair_12_without_weather_deck_raises`. Add `test_initialize_sets_dvba_from_dvbe`.
- [ ] **Step 2–4:** implement; PASS `pytest tests/unit/test_round6_environment.py tests/unit/test_round6_*.py tests/unit/test_hyper6_one_step.py -q`
- [ ] **Step 5:** commit `feat: Round6 environment tabular wind and Dryden`

---

### Task 4: Rocket6 aerodynamics (SLV stages + der)

**Files:**
- Create: `Python/src/cadac/vehicles/rocket6/__init__.py`, `Python/src/cadac/vehicles/rocket6/aero.py`
- Test: `Python/tests/unit/test_rocket6_aero.py`

**Interfaces:**
- Produces: `Rocket6Aero(aero_deck)` name `"aerodynamics"`. Port `Hyper::init_aerodynamics`, `aerodynamics`, `aerodynamics_der` from ROCKET6 `aerodynamics.cpp`. `maero` in `{11,12,13}` else `ValueError`. Table names `ca0slv{N}_vs_mach` etc. as C++. Tests parse ASC via `parse_asc_deck` / `Datadeck.from_tables` (JSONC in Task 17).

- [ ] **Step 1:** `maero=13`, `vmach=0.5`, `alppx=2`, `phipx=0`, `mprop=3`, finite `cx`/`cz`/`clm`; `maero=1` raises. Replica `cx==-ca` with `ca=ca0+caa*alppx+ca0b`.
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 SLV aerodynamics`

---

### Task 5: Rocket6 propulsion (analytic mprop 0/3/4)

**Files:**
- Create: `Python/src/cadac/vehicles/rocket6/propulsion.py`
- Test: `Python/tests/unit/test_rocket6_propulsion.py`

**Interfaces:**
- Produces: `Rocket6Propulsion` name `"propulsion"`. No deck. Port `Hyper::propulsion`. `mprop` in `{0,3,4}` else `ValueError`. `thrust=spi*fuel_flow_rate*AGRAV` for 3/4. `integrate` fuel. Shutdown `fmassr<=0` sets `mprop=0`. When `mprop==0`, **every execute** zeros `thrust`, `fmasse`, `fmassr`, and `fmassd` as C++ (BECO event sets `mprop=0` only and does not write `fmasse`). `init` no-op. Skip `mfreeze` if absent.

- [ ] **Step 1:** insertion stage-1 numbers `spi=279.2`, `fuel_flow_rate=514.1`, `mprop=3`, `dt=0.001` → `thrust==spi*fuel_flow_rate*AGRAV`; `mprop=0` after a burn with leftover `fmasse` → `thrust==0` and `fmasse==0` and `fmassr==0` on that step; `mprop=2` raises; after enough steps `fmassr<=0` implies `mprop==0`.
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 analytic rocket propulsion`

---

### Task 6: Rocket6 TVC (mtvc 0/2)

**Files:**
- Create: `Python/src/cadac/vehicles/rocket6/tvc.py`
- Test: `Python/tests/unit/test_rocket6_tvc.py`

**Interfaces:**
- Produces: `Rocket6Tvc` name `"tvc"`. Port `tvc` / `tvc_scnd`. `mtvc==0` return (FPB stays defined zeros). `mtvc==2` second-order. Else `ValueError`. CADAC sign local. `dt=ctx.int_step`.

- [ ] **Step 1:** `mtvc=0` no raise; `mtvc=2`, `delecx=1`, `gtvc=1`, `thrust>0`, `dt=0.001` → `|etax|<=tvclimx`, `FPB[0]` finite; `mtvc=1` raises.
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 second-order TVC`

---

### Task 7: Rocket6 RCS (Schmitt 20/21/22)

**Files:**
- Create: `Python/src/cadac/vehicles/rocket6/rcs.py`
- Test: `Python/tests/unit/test_rocket6_rcs.py`

**Interfaces:**
- Produces: `Rocket6Rcs` name `"rcs"`. Port `rcs`, `rcs_schmitt`, `rcs_prop`. `mrcs_force` must be 0 else `ValueError`. `mrcs_moment` in `{0,20,21,22}` else `ValueError`. Decode type/mode as C++. Mode 1 Euler, 0 roll-only, 2 `UTBC` vector. CADAC sign local.

- [ ] **Step 1:** `rcs_schmitt` replica: increasing through dead_zone/hysteresis vs C++ table of (input_new, input, dz, hy, output). `mrcs_moment=21` finite `FMRCS`; `mrcs_moment=11` raises; `mrcs_force=1` raises.
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 Schmitt RCS`

---

### Task 8: Rocket6 control (maut 0/53)

**Files:**
- Create: `Python/src/cadac/vehicles/rocket6/control.py`
- Test: `Python/tests/unit/test_rocket6_control.py`

**Interfaces:**
- Produces: `Rocket6Control` name `"control"`. Port `control`, `control_normal_accel`, `control_yaw_accel`. `maut` in `{0,53}` else `ValueError`. `define` C++ `def_control`. Limit `|delecx|<=delimx`, `|delrcx|<=drlimx`. `mprop==0` skips accel calls (C++ `if(mprop)`).

- [ ] **Step 1:** frozen `pdynmc`, `dla`/`dmde`/`FSPCB` vs C++ `delecx` rtol 1e-12; `maut=53` no error; `maut=24` raises; `maut=0` finite limited commands.
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 acceleration autopilot`

---

### Task 9: Rocket6 forces

**Files:**
- Create: `Python/src/cadac/vehicles/rocket6/forces.py`
- Test: `Python/tests/unit/test_rocket6_forces.py`

**Interfaces:**
- Produces: `Rocket6Forces` name `"forces"`. Port `Hyper::forces`. `FAPB[i]=pdynmc*refa*{cx,cy,cz}`. `FMB[i]=pdynmc*refa*refd*{cll,clm,cln}` — `refd` only, not GHAME `refb`/`refc`. Writes `FAPB`/`FMB` only. Missing RCS/TVC vectors → zero.

- [ ] **Step 1:** frozen aero/thrust, `mtvc=0`, `mprop=3` → `FAPB[0]==pdynmc*refa*cx+thrust`; `FMB[1]==pdynmc*refa*refd*clm`; `mtvc=2` adds `FPB`; `FSPB` not written.
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 forces FAPB/FMB`

---

### Task 10: Rocket6 INS mins=0

**Files:**
- Create: `Python/src/cadac/vehicles/rocket6/ins.py`
- Test: `Python/tests/unit/test_rocket6_ins_ideal.py`

**Interfaces:**
- Produces: `Rocket6Ins` name `"ins"`. `define` C++ `def_ins` (instrument vectors default **0**, not `gauss()`). `mins==0`: copy truth into computed names as C++ `if(mins==0)` plus shared geographic/Euler block. `init` no-op for mins=0. Task 10 `execute` supports `mins==0` only; `mins!=0` → `ValueError` until Task 11.

- [ ] **Step 1:** `mins=0` copies `SBII` to `SBIIC`, `TBI` to `TBIC`; `mins=2` raises.
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 ideal INS`

---

### Task 11: Rocket6 INS mins=1 (zero draws)

**Files:** Modify `ins.py`; `Python/tests/unit/test_rocket6_ins_errors.py`

**Interfaces:**
- Port `ins` else-branch, `ins_gyro`, `ins_accl`, `ins_grav`. Cholesky/`gauss` draws = 0 so `init` writes zero `ESBI`/`EVBI`/`RICI`. GPS `mgps==3` applies `SXH`/`VXH` and sets `mgps=2` if those names exist else skip. Star `mstar==3` applies `URIC` and sets `mstar=2` if present. Skip `mroll` if absent.

- [ ] **Step 1:** zero instruments, `mins=1`, one step: `SBIIC==SBII` within 1e-12 (zero errors); gyro replica `WBICB=WBIB` when all E* = 0; `ins_grav` vs C++ `GM` formula rtol 1e-12. `mins=1` no longer raises.
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 space-stabilized INS`

---

### Task 12: GPS constellation and quadriga

**Files:**
- Create: `Python/src/cadac/vehicles/rocket6/gps.py`
- Test: `Python/tests/unit/test_rocket6_gps_quadriga.py`

**Interfaces:**
- Produces: module-level `SV_INIT` 24×2 Yuma 787 table, `rsi=26560000`, `incl=0.95986`, `wsi=sqrt(GM/rsi**3)` copied from `gps_sv_init`. Functions `gps_sv_init()`, `gps_quadriga(...)` port C++ (use `angle` from frames, `LARGE=1e10` module-level). `<4` visible → caller sets `mgps=1`. No `cout`. `define`/`execute` can be pass until Task 13 except `define` C++ `def_gps`.

- [ ] **Step 1:** `sv_data[0]==(5.63,-1.600)`; Vandenberg `SBII=cad_in_geo84(-120.49*RAD,34.68*RAD,100,0)` at `time=0`, `almanac_time=80000`, `del_rearth=2317000` → 4 slots in 1..24, finite `gdop`.
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 GPS Yuma quadriga`

---

### Task 13: GPS filter mgps 0/1/2/3

**Files:** Modify `gps.py`; `Python/tests/unit/test_rocket6_gps_filter.py`

**Interfaces:**
- Produces: `Rocket6Gps` name `"gps"`. Port `Hyper::gps` with sequential `if` fall-through in **one** `execute` (C++ `if(mgps==1){... mgps=2;} if(mgps==2){...}`). Do **not** use `elif` (that delays extrapolate by one step). Instance state for `PP (8,8)`, `PHI`, `FF`, `sv_init_data`. `mgps` in `{0,1,2,3}` else `ValueError`. Markov/Gauss params used as stored values (already 0/mean). No console. numpy inverse for Kalman gain matching C++ `(HH@PP@HH.T+RR)`.

- [ ] **Step 1:** `mgps=0` return; **one** `execute` starting `mgps=1` → store `mgps==2` **and** `PP` already extrapolated this call (`PP[0,0]` changed from the init diagonal, or `std_pos` written); `PP` init diagonal `(ppos*(1+factp))**2` is visible if you inspect mid-init in a helper. `mgps=4` raises; one `mgps=3` frozen geometry → `SXH` finite. Pin: `elif` after init would leave `std_pos==0` / no extrapolate on that same call — that must fail.
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 GPS EKF`

---

### Task 14: Rocket6 startrack

**Files:**
- Create: `Python/src/cadac/vehicles/rocket6/startrack.py`
- Test: `Python/tests/unit/test_rocket6_startrack.py`

**Interfaces:**
- Produces: `Rocket6Startrack` name `"startrack"`. Port catalog, `star_triad`, `startrack` with sequential `if` fall-through in **one** `execute` (C++ `if(mstar==1){... if(alt>startrack_alt) mstar=2;} if(mstar==2){...}`). Do **not** use `elif` (that delays the wait/step block by one step). `mstar` in `{0,1,2,3}` else `ValueError`. Uses `cart_from_pol`, `cad_geo84_in` only for diagnostics. No console names.

- [ ] **Step 1:** catalog row 0 ≈ Sirius `(-.179457,.947482,-.264715)`; `mstar=0` no raise; **one** `execute` with `mstar=1` and `alt>startrack_alt` → `mstar==2` (and wait-clock logic has run this call if `star_acq`/`starfix_epoch` require it); `mstar=1` at `alt=0` stays 1 (C++ does not set 2 below `startrack_alt`); `mstar=4` raises; triad volume in (0,1].
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 star tracker`

---

### Task 15: Rocket6 guidance (mguide 0/5 LTG)

**Files:**
- Create: `Python/src/cadac/vehicles/rocket6/guidance.py`
- Test: `Python/tests/unit/test_rocket6_guidance.py`

**Interfaces:**
- Produces: `Rocket6Guidance` name `"guidance"`. `mguide==0` zero `UTBC` return. `mguide==5` port `guidance_ltg` + `_tgo` `_igrl` `_trate` `_trate_rtgo` `_pdct` `_crct`. `cad_kepler` for `_pdct`. `x==1` → `ValueError` not `sys.exit`. Writes `mprop`/`beco_flag`. Else `ValueError`. No BECO print.

- [ ] **Step 1:** `mguide=0` `UTBC==0`; `mguide=6` raises; `mguide=5` with insertion LTG inputs and frozen INS state, one `ltg_step` → finite `UTBC` unit-ish (`||UTBC||` near 1 after skip_flag clears, or 0 during skip — pin C++ skip_flag behavior: first 9 calls leave UTIC unset/zero).
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 linear-tangent guidance`

---

### Task 16: Rocket6 intercept

**Files:**
- Create: `Python/src/cadac/vehicles/rocket6/intercept.py`
- Test: `Python/tests/unit/test_rocket6_intercept.py`

**Interfaces:**
- Produces: `Rocket6Intercept` name `"intercept"`. Port `intercept` without `cout`/`exit`. `alt<=0` and `write`: `write=0`, `vehicle.health=0`, `ctx.combus[ctx.vehicle_slot].status=0` if combus present. `modes` diagnostic.

- [ ] **Step 1:** `alt=100` health stays 1; `alt=-1` health 0; never calls `sys.exit`.
- [ ] **Step 2–5:** implement, pass, commit `feat: ROCKET6 ground-impact intercept`

---

### Task 17: Rocket6 vehicle + translate insertion

**Files:**
- Create: `Python/src/cadac/vehicles/rocket6/vehicle.py`
- Modify: `Python/src/cadac/cli.py` (register family; Rocket6 factory: aero required, prop optional, weather passed into `Round6Environment`)
- Translate: `input.asc` → `Python/cases/rocket6/` via `translate_scenario_asc(..., family="rocket6")`; call `deck_asc_to_jsonc` on `aero_deck_SLV.asc` and `weather_deck_Wallops.asc` into that directory
- Test: `Python/tests/unit/test_rocket6_one_step.py`

**Interfaces:**
- Produces: `Rocket6.type=="HYPER6"`, `Rocket6.family=="rocket6"`. Constructor `(name, aero_deck, events=None, weather_deck=None, prop_deck=None)`. Modules in insertion ASC order: kinematics, environment, propulsion, aerodynamics, gps, startrack, ins, guidance, control, rcs, tvc, forces, newton, euler, intercept. Environment constructed with `weather_deck`. Skip-if-exists on name collisions. `_VEHICLE_FAMILIES.setdefault(("rocket6","HYPER6"), Rocket6)` — do not replace the dict. After family lookup, `_build_vehicle` uses the Rocket6 constructor: require `aero_deck`; `prop_deck` optional; pass weather Datadeck into `Round6Environment`. Do **not** fall through to the Hyper6 both-decks constructor. `end_time` 190. Smoke: 0.1 s, `alt` near 100. `AIM5` / `NO_SUCH_TYPE` tests untouched.

- [ ] **Step 1:** smoke FAIL then PASS; family JSONC type HYPER6 runs Rocket6; no-family HYPER6 still Hyper6; AIM5 still raises; case dir contains `aero_deck_SLV.jsonc` and `weather_deck_Wallops.jsonc`.
- [ ] **Step 2–5:** wire, pass, commit `feat: run ROCKET6 from JSONC insertion`

---

### Task 18: ROCKET6 e2e golden (optional file)

**Files:** `Python/tests/e2e/test_rocket6_insertion.py`

**Interfaces:** Skip without `tests/e2e/goldens/rocket6/plot.csv`. If that file is present, it is a C++ plot from a **deterministic** run: `MONTE` off (`nmonte==0` GAUSS/RAYL/MARKOV path) **and** Dryden `gauss_value` forced to 0. It is **not** a raw `input.asc` `plot.csv` (that file used `MONTE 1` and live `rand()`). Compare plot-flagged columns (`alt`/`vmach` if both present). Sentinel time=-1. Copy the structure of `tests/e2e/test_hyper6_climb.py`. Bump `UPDATES.md` subsubver when this task is executed in a session that maintains project-docs.

- [ ] **Step 1:** skip without golden; with golden, alt at t=0; docstring or comment states the golden is MONTE-off + Dryden-drive-0, not raw `input.asc` plot.csv
- [ ] **Step 2–5:** implement, pass, commit `test: ROCKET6 e2e gate (skip without golden)`

---

## Self-review

- Family/weather translate, kepler/polar, env mair=12, aero, prop, TVC, RCS, control, forces, INS 0, INS 1, GPS quadriga, GPS EKF, startrack, LTG, intercept, vehicle, e2e: each has a task
- Non-goals ballistic, mtvc 1/3, maut other than 0/53, mguide other than 0/5, matmo 1/2, live rand/Markov: no tasks that implement them (they `ValueError` or are zeroed)
- `_VEHICLE_TYPES["HYPER6"]` never overwritten; `_VEHICLE_FAMILIES` not wiped; `AIM5` / `NO_SUCH_TYPE` tests not retargeted
- No `sys.exit`; no `hyper6` imports
- Type names: `Rocket6*` modules; `Round6Environment(weather_deck=None)` default preserved; `initialize` still sets `dvba=dvbe`
- GPS/startrack sequential `if` (same-call fall-through); forces use `refd`; `mprop==0` zeros `fmasse` every step
- E2E golden is MONTE-off + Dryden-drive-0, not raw `input.asc` plot.csv
