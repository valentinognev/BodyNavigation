# CADAC Python library — design

Date: 2026-09-04
Status: approved (chat). First slice: kernel + HYPER3 + FALCON5 + FALCON6.

## Goal

One installable Python package `cadac` under `Python/` that runs Zipfel CADAC++ simulations from JSONC inputs, in Python style (named state, numpy, module protocol), with numerics regression-close to C++.

## Non-goals (first slice)

MAGSIX Rotor, Round6 (HYPER6/ROCKET6), Ground0/Flat0 radars, AIM5/AGM6/SAM6/SRAAM6 vehicle modules, Monte Carlo / `markov_noise`, CADAC Studio `.asc` plot merge beyond CSV.

## Layout

```
Python/
  pyproject.toml
  src/cadac/
    __init__.py          # run_scenario, translate_asc
    constants.py
    kernel/              # integrate, executive, events, combus, module protocol, state
    io/                  # jsonc, scenario, deck, plot, translate_asc
    tables/              # Table, Datadeck, look_up
    env/                 # us76, iso62, gravity
    math/                # frames (mat2tr, mat3tr, polar), earth (cadtei, cadsph, cadtge)
    eom/
      round3.py          # first slice
      flat3.py           # first slice
      flat6.py           # first slice
      round6.py          # reserved, empty except docstring
      ground0.py         # reserved
      flat0.py           # reserved
      rotor.py           # reserved (MAGSIX; not Flat6)
    vehicles/
      cruise3/           # HYPER3, type CRUISE3
      plane5/            # FALCON5, type PLANE
      plane6/            # FALCON6, type PLANE6
  tests/unit tests/translate tests/e2e
  cases/hyper3|falcon5|falcon6
  tools/translate_asc.py
```

Package name: `cadac`. Python >= 3.11. Dependencies: `numpy`, `pytest`. No scipy for EOM.

## Runtime I/O

Runtime reads **JSONC only**. CADAC `.asc` is an offline translation source.

Public API:

```python
from cadac import run_scenario
result = run_scenario(path)  # Path to scenario JSONC
# result.plot: pandas-optional; first slice uses list[dict] or numpy structured arrays
# writes plot.csv next to the case when options.plot and options.csv are true
```

CLI: `cadac run <scenario.jsonc>` and `cadac translate-asc <input.asc> -o <out.jsonc>`.

Unknown JSONC `params` keys are errors. Missing decks, unknown vehicle `type`, `int_step <= 0`, table name missing: fail with path and key. No `system("pause")`.

## Scenario JSONC

Sections: `title`, `options` (object of booleans; omitted = false), `modules` (list of `{name, phases}`), `timing`, `end_time`, `vehicles`.

Vehicle: `type`, `name`, optional `aero_deck` / `prop_deck` (paths relative to the scenario file), `params` (CADAC field names), `events` (ordered).

Event: `{ "when": {"time": {">": 10}} | {"var": "wp_flag", "op": "=", "value": -1}, "set": { ... } }`.
Ops: `<`, `=`, `>`. Int watch vars compare as int; floats use raw `==` / `<` / `>` (CADAC, no epsilon).

Comments: JSONC `//` on the same keys as CADAC `//` lines. `cadac.io.jsonc.loads` strips `//` line comments and `/* */`; then `json.loads`. No trailing commas. `//` inside JSON strings is out of scope (CADAC comments are never inside strings).

`options` keys map from CADAC `y_scrn`/`n_scrn` to `scrn: true/false`. Known: `scrn`, `events`, `plot`, `doc`, `csv`, `tabout`, `merge`, `comscrn`, `traj`.

Module `phases`: subset of `def`, `init`, `exec`, `term`. `def` always runs at build. Calling **order is the modules list**.

## Table JSONC

```jsonc
{ "title": "...", "tables": [{ "name": "...", "dim": 1|2|3, "x1": [...], "x2": [...], "x3": [...], "values": nested }] }
```

`values` shape: 1D `len(x1)`; 2D `[len(x1)][len(x2)]`; 3D `[len(x1)][len(x2)][len(x3)]`. Translator flattens CADAC’s split-breakpoint 2D/3D layout into these arrays. Runtime never parses `.asc` decks.

## look_up (CADAC Datadeck)

Binary `find_index`: largest index `i` with `list[i] <= value`, clamped to `[0, n-1]`. If `value >= list[max]` return `max`; if `value <= list[0]` return `0`.

1D: if `loc == n-1`, return `data[n-1]` (constant upper). Else linear interpolate; **slope extrapolation below min** (loc=0, interpolate toward index 1). `dx > EPS` (`EPS=1e-10`) else `dumx=0`.

2D/3D: bilinear/trilinear as in `Datadeck::interpolate`. Constant **upper** extrapolation per axis (`ind_hi = ind_lo` when `ind_lo == dim-1`). Slope **lower** extrapolation. Data packing: 2D `data[i1 * n2 + i2]`; 3D as C++ `HYPER3/utility_functions.cpp`.

## Integrator

```
y_new = y + (dydx_new + dydx) * int_step / 2
```

Scalar and `ndarray` (broadcast). Caller stores `dydx = dydx_new`. EOM must keep CADAC derivative **order** (e.g. Round3: integrate `vbii` from `abii`, then `sbii` from `vbii_new` and old `vbii`).

No RK4, no scipy ODE.

## Constants (verbatim CADAC)

`REARTH=6370987.308`, `WEII3=7.292115e-5`, `RAD=0.0174532925199432`, `DEG=57.2957795130823`, `AGRAV=9.80675445`, `G=6.673e-11`, `EARTH_MASS=5.973e24`, `R=287.053`, `PI=3.1415927`, `EPS=1e-10`.

US76 uses its own `rearth=6369.0` km inside `atmosphere76` — do not substitute `REARTH`.

## Environment

- **ISO 62** (Round3 / HYPER3): `alt < 11000`: `k=288.15-0.0065*alt`, `press=101325*(k/288.15)**5.2559`; else `k=216.0`, `press=22630*exp(-0.00015769*(alt-11000))`. `rho=press/(R*k)`, `vsound=sqrt(1.4*R*k)`, `mach=abs(dvbe/vsound)`, `pdynmc=0.5*rho*dvbe**2`.
- **US76** (Flat3 / Flat6): copy `atmosphere76` from HYPER3 `utility_functions.cpp` (same as FALCON).
- **Gravity** (all three EOM): `grav = G*EARTH_MASS/(REARTH+alt)**2`.

Do not merge ISO 62 and US76.

## State and modules

Named attributes, CADAC names (`newton.lonx`, `aerodynamics.alphax`). Vectors `shape (3,)`, matrices `(3,3)`. No `Variable[i]` arrays.

Each field: `role` in `{data, state, diag, out, exec, init, save}`, `units`, `module`, `outputs` subset of `{scrn, plot, com}`. Lookup for events and JSONC params is **by name**. Duplicate names on one vehicle are errors.

```python
class Module(Protocol):
    name: str
    def define(self, vehicle) -> None: ...
    def initialize(self, vehicle, ctx: SimContext) -> None: ...
    def execute(self, vehicle, ctx: SimContext) -> None: ...
    def terminate(self, vehicle, ctx: SimContext) -> None: ...
```

`SimContext`: `sim_time`, `int_step` (mutable for HYPER3 `int_step_new`), `event_time`, `out_fact`, `combus`, `vehicle_slot`.

Vehicle = composition: EOM stack + vehicle modules. No Cadac vtable stubs.

## Events

One event armed (`nevent` walks the list). On fire: apply `set` by name, `event_epoch=True`, `event_time=0`, advance. `event_time += int_step` every step after the vehicle body (including dead vehicles).

## Executive

```
init: load JSONC → construct vehicles by type → define → initialize (phases with init) → t=0 outputs
while sim_time <= end_time + int_step:
    for vehicle i:
        evaluate next event
        if event_epoch: event_time = 0
        if combus[i].status == 1:
            for module in scenario.modules: execute
            save health; publish packet; restore health
        event_time += int_step
        maybe scrn/plot if abs(t_out - sim_time) < int_step/2 + EPS
    maybe traj/comscrn
    sim_time += int_step
```

`status`: 1 alive, 0 dead, -1 hit. Packet: `{name, type, status, vars}` named combus subset.

## EOM taxonomy (library-wide)

| Layer | Dynamics | First-slice impl |
|---|---|---|
| Round3 | spherical 3DOF translation | yes |
| Flat3 | flat 3DOF / Zipfel 5DOF | yes |
| Flat6 | flat 6DOF | yes |
| Round6 | spherical 6DOF | reserved |
| Ground0 | fixed site (round) | reserved |
| Flat0 | radar site (flat) | reserved |
| Rotor | MAGSIX DNU Magnus | reserved — not a Flat6 flag |

JSONC `type` tokens: `CRUISE3`, `PLANE`, `PLANE6` in first slice. Later: `AIM5`, `AIRCRAFT3`, `TARGET3`, `SATELLITE3`, `HYPER5`, `HYPER6`, `MISSILE6`, `ROCKET5`, `RADAR0`, `ROTOR`. `TARGET3` and `RADAR0` are ambiguous across programs — bind `(family, type)` when those platforms are added.

Same JSONC module **name** (`environment`, `newton`, `kinematics`) binds a **different class** per vehicle type.

## First-slice vehicles

**CRUISE3 / Round3:** modules `environment`, `aerodynamics`, `propulsion`, `forces`, `newton`.  
Aero: `cl = cla0 + cla*alphax`, `cd = cd0 + ckk*(cl-cl0)**2` with 1D Mach tables.  
Prop: `mprop` 0/1/2; `thrust = spi*0.029*throttle*AGRAV*rho*dvbe*ca*acowl` (`mprop==1`); autothrottle as HYPER3 `cruise_modules.cpp`; fuel `fmasse` via `integrate`.  
Forces: FSPV as HYPER3 `Cruise::forces`.  
Newton: inertial `sbii`/`vbii`/`abii`, geographic `sbeg`/`vbeg`, `cadtei`/`cadsph`/`cadtge`, `WEII` = `(0,0,WEII3)`.

**PLANE / Flat3:** modules `environment`, `kinematics` (timing only), `aerodynamics`, `propulsion`, `guidance`, `control`, `forces`, `newton`, `intercept`. US76. Newton: `NEXT_ACC = TBL.T @ FSPV + [0,0,grav]`, then integrate vel/pos as FALCON5. Control/guidance: port FALCON5 `plane_modules.cpp` functions (`control_heading`, `control_flightpath`, `control_bank`, `control_load`, `control_lateral`, `control_altitude`, `guidance_line`, `guidance_point`) with `mcontrol` / `mguidance` dispatchers.

**PLANE6 / Flat6:** modules `environment` (US76 + wind flags), `kinematics` (DCM/α/β), `aerodynamics` (+ `_der`), `propulsion`, `forces`, `control`, `actuator`, `euler`, `newton`. Port FALCON6 module `.cpp` files. Guidance module exists so JSONC may list it; unused modes not invented.

## Frames (CADAC)

- `polar_from_cart(v)`: `d=||v||`, `az=atan2(v[1],v[0])`, elev from `atan2(-v[2], hypot(v[0],v[1]))` with CADAC vertical special cases.
- `mat2tr(psivg, thtvg)`: CADAC `mat2tr` element assignment (not a generic ZYX helper).
- `mat3tr(psi, tht, phi)`: CADAC `mat3tr`.
- `cadtei(t)`: identity with `xi=WEII3*t` in the 2×2 top-left.
- `cadsph(sbie)`: lon/lat/alt, multi-valued lon as C++.
- `cadtge(lon, lat)`: CADAC matrix.

## Testing

Unit: `rtol=1e-12`, `atol=1e-14` vs CADAC formulas.  
Deck ASC vs JSONC: arrays equal.  
E2E vs CADAC CSV plot columns: `rtol=1e-5`, `atol=max(1e-6, 5e-6*|g|)` (CSV print precision). Time grid matches `plot_step`.

Goldens: `tests/e2e/goldens/`. HYPER3: copy `CADAC_Simulations/HYPER3_250114/HYPER3/plot1.csv`. FALCON5/FALCON6: skip e2e with `pytest.importorskip` / `pytest.mark.skipif` until a CADAC `plot.csv` is present; unit tests of modules still required.

TDD on every task. pytest.

## Implementation process

Grok implementer (`cursor-grok-4.6-high`) per plan task; Grok task reviewer on the diff; Grok whole-branch review at the end. No Composer Fast, no Kimi 3, no Fast mode. Controller does not patch physics. Isolated worktree; no commit to main without user ask.

## Python style (binding)

Named state, numpy `@`, dataclasses/attrs for configs, pathlib, no C++ index comments as the data model. CADAC **numeric** algorithms and **update order** are copied, not “improved.”
