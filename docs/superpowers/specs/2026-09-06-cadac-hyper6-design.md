# CADAC HYPER6 Hyper (Round6) — design

Date: 2026-09-06
Status: approved (chat). Second slice plan 2. Depends on HYPER5 plan complete (kernel multi-vehicle skip + deck-optional factory already shipped).

Parent: `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
Sibling: `docs/superpowers/specs/2026-09-06-cadac-hyper5-design.md`

## Goal

Add Round6 EOM and JSONC type `HYPER6`. First case: `input_climb.asc` (1 vehicle, 60 s, GHAME, `mprop=2`, `maut=24`, `mins=0`). Numerics regression-close to `CADAC_Simulations/HYPER6_250125/HYPER6/`.

## Non-goals

HYPER6 Satellite, Radar, Ground0. GPS, startrack, seeker, datalink, RCS, intercept. `mins!=0` INS errors. `mair` wind/turbulence/NASA-extended atmosphere. `maero=2` transfer vehicle. `mprop` 3/4 rocket and LTG. `minit=1` automated satellite-relative ICs. Markov / Monte Carlo. ROCKET6 (same Round6 layer, different vehicle — later).

## Layout

```
Python/src/cadac/eom/round6.py     # Environment, Kinematics, Euler, Newton
Python/src/cadac/vehicles/round6/hyper6/
  vehicle.py      # Hyper6 type="HYPER6"
  aero.py
  propulsion.py
  actuator.py
  control.py
  forces.py
  guidance.py
  ins.py
Python/cases/hyper6/              # translated input_climb.asc + ghame6 decks
Python/tests/unit/test_round6_*.py test_hyper6_*.py
Python/tests/e2e/test_hyper6_climb.py
```

Do not reuse `Flat6*` or `Plane6*` classes. Shared kernel only: `integrate`, `look_up`, `mat3tr`, `cadtei`/`cadsph`/`cadtge`, `atmosphere76`, `gravity`.

## Round6 EOM

Port `HYPER6/{environment,kinematics,euler,newton}.cpp`. Atmosphere is **US76**, not ISO 62.

**Environment:** `mair = |matmo|mturb|mwind|`. Climb default 0 → US76, no turb, no wind, `VAED=0`, `dvba` from geographic speed. Writes `press`, `rho`, `vsound`, `vmach`, `pdynmc`, `tempk`, `GRAVG`, `grav`. Other `mair` → `ValueError`. Skip `mfreeze` latch if `mfreeze` absent.

**Kinematics:** timing (`time=ctx.sim_time`, `int_step_new`/`out_step_fact` like Round3). TBD (body wrt geodetic) and TBI (body wrt inertial) as C++ — DCM/state derivative, not Flat6 quaternions. Euler angles `psibdx`/`thtbdx`/`phibdx` degrees on the store. `alphax`/`betax`/`alppx`/`phipx` from air-relative velocity as C++.

**Euler:** `WBEB` from `ppx`/`qqx`/`rrx`; integrate with inertia and `FMB` as C++ `Round6::euler`.

**Newton:** `minit=0` geographic ICs (`lonx`, `latx`, `alt`, `dvbe`, `alpha0x`/`beta0x`). Inertial `SBII`/`VBII`/`ABII` stored-slope `integrate`. `FSPB` newton-owned (`FAPB/vmass`). Geographic lon/lat/alt, `psivdx`/`thtvdx`, `dvbe`. `minit!=0` → `ValueError`.

## Hyper6 climb modules

CADAC MODULES: `kinematics`, `environment`, `aerodynamics`, `propulsion`, `ins`, `guidance`, `control`, `actuator`, `forces`, `newton`, `euler`.

**Aerodynamics:** `maero==1` GHAME tables (`ghame6_aero_deck.asc`) + `_der` as C++ `Hyper::aerodynamics`. Else `ValueError`.

**Propulsion:** `mprop==2` hyper autothrottle q-hold as C++. `0` none; `1` hyper manual. Else `ValueError` (no 3/4). Fuel/mass as C++.

**INS:** `mins==0` ideal: copy truth into computed `SBIIC`/`VBIIC`/`TBIC`/Euler/flight-path as C++ `if(mins==0)` paths (init is no-op). `mins!=0` → `ValueError`. Control/guidance that read `*c` INS names must see these copies.

**Guidance:** `mguide==0` return (climb default). Else `ValueError`. No LTG / line / pronav in this plan.

**Control:** `maut=|mauty|mautp|`. Climb `maut=24`. Also `0`. Else `ValueError`. Port HYPER6 `control.cpp` helpers used by those modes (roll, SAS rates, gamma). Limit `|del*|` as C++.

**Actuator:** `mact` 0 (limit only) and 2 (second order `actuator_scnd`). Else `ValueError`. CADAC sign `<0 → -1` else `+1`. Local helper; not `np.sign`; not `flat6._cadac_sign`.

**Forces:** body `FAPB`/`FMB` from aero + thrust as C++ `Hyper::forces`. Do not write `FSPB`.

## First case

Translate `input_climb.asc` + `ghame6_aero_deck.asc` + `ghame6_prop_deck.asc` to `Python/cases/hyper6/`. `end_time` 60. Event `time>10` sets `thtvdcomx=10`.

Smoke: short run (`0.1 s`) `alt` near 10000.

## Testing

Unit: `rtol=1e-12` vs CADAC formulas. TDD each task.

E2E: `tests/e2e/test_hyper6_climb.py`. Skip if `tests/e2e/goldens/hyper6/plot.csv` absent. Else CSV tolerances on Hyper plot columns (`alt`/`vmach` or plot-flagged names present in both); sentinel `time=-1`; `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)`.

Regression: HYPER3 e2e, HYPER5 units, FALCON5/FALCON6 units.

## Process

Same as HYPER5 spec: Grok `cursor-grok-4.6-high` implementer + reviewer per task; whole-plan review; no Fast/Kimi; controller does not patch physics.

## Python style

Named state; CADAC numeric order; same module name binds Round6 vs Round3/Flat6 by vehicle type.
