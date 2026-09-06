# Modeling and Simulation of Aerospace Vehicles (Zipfel)

## Idea
Workspace for Peter H. Zipfel's CADAC++ simulations (3/5/6-DOF aerospace vehicles) and a unified Python library that re-implements them in Python style, not as a line-for-line C++ transcription.

Mandatory reading: `UPDATES.md`.

## Architecture
**C++ source of truth:** `CADAC_Simulations/` — twelve standalone CADAC++ programs. Each copies executive, `Variable` tables, `Matrix`/`Datadeck`, atmosphere, and I/O. Hierarchy is `Cadac` → `Round3`/`Flat3`/`Round6`/`Flat6` → vehicle (Cruise, Plane, Missile, …). Multi-vehicle runs use a `combus` packet bus.

| Folder | DOF / earth | Vehicles |
|---|---|---|
| HYPER3 | 3DOF round | Cruise |
| AIM5 | 5DOF flat | Aim + Aircraft |
| CRUISE5 | 5DOF round | Cruise + Satellite + Target |
| FALCON5 | 5DOF flat | Plane |
| HYPER5 | 5DOF round | Hyper + Satellite + Target |
| FALCON6 | 6DOF flat | Plane |
| HYPER6 | 6DOF round (+ 3DOF) | Hyper + Radar + Satellite + Ground |
| MAGSIX | 6DOF rotor | Rotor |
| ROCKET6 | 6DOF round | Rocket |
| SAM6 | 6DOF flat (+ 3DOF) | Missile + Aircraft + Rocket + Radar |
| SRAAM6 | 6DOF flat (+ 3DOF) | Missile + Target |
| AGM6 | 6DOF flat (+ 3DOF) | Missile + Aircraft + Target |

**Python target:** `Python/` — installable package `cadac`. Runtime JSONC (scenarios + decks); `cadac.io.translate.deck_asc_to_jsonc` converts CADAC `.asc` decks. Named numpy state; CADAC `integrate` / `look_up` / atmospheres. `run_loop` skips module names missing on a vehicle and seeds combus from store `com_names` before the first execute; plot CSV records vehicle slot 0 only. First slice: kernel + HYPER3 (`CRUISE3`/Round3) + FALCON5 (`PLANE`/Flat3) + FALCON6 (`PLANE6`/Flat6). HYPER5 is runnable from JSONC (`Python/cases/hyper5/` Demo 4.7): `Hyper5` (`type="HYPER5"`, health=1; aero required, prop if `mprop!=0`) plus registered `TARGET3` / `SATELLITE3` (no decks). HYPER6 is runnable from JSONC (`Python/cases/hyper6/` climb): `Hyper6` (`type="HYPER6"`, health=1; both aero and prop decks required). Radar/Satellite/Ground0 are not registered. Optional JSONC `"family"` (vehicle key else scenario key) selects `_VEHICLE_FAMILIES[(family, type)]` with no type-table fallback; `HYPER6` without family still maps to `Hyper6`. `translate_scenario_asc(..., family=)` writes scenario-level `"family"` only; vehicle parser maps `WEATHER_DECK` and `GAUSS`/`RAYL`/`MARKOV`. E2E `tests/e2e/test_hyper5_pronav.py` skips if `tests/e2e/goldens/hyper5/plot.csv` is absent. HYPER5 Python modules: aero (Roadrunner 3X), propulsion (mprop 0–3), forces (FSPV), control (`control_bank`, `control_load`, `control_altitude`, `control_heading`, `control_flightpath`, `control_lateral`, `mcontrol` in `{0,3,4,6,16,36,40,44}`), guidance (`guidance_point`, `guidance_pronav`, `guidance_arc`; `mguidance` in `{0,44,66,70}`), seeker (`mseeker` in `{0,1,3}`; Demo 4.7 `acq_range=6000`), intercept (`halt` / ground `alt<=0` / seeker closest-approach; `vehicle.health=0` and combus status 0; no `sys.exit`), targeting (`mtargeting==0` return; `==1` satellite visibility / closest TARGET3 waypoint; else ValueError), `Target3` (`type="TARGET3"`; Round3 env+newton + Target Coriolis/centrifugal forces + intercept `targ_health` from combus; constructor `(name, events=None)`, no decks), and `Satellite3` (`type="SATELLITE3"`; Round3 env/newton + `Satellite3Forces`; `sat_thrust` default 0, `sat_mass` default 100; `FSPV=[sat_thrust/sat_mass,0,0]`; constructor `(name, events=None)`, no decks, no seeker). Round3 ISO 62 later. Seven EOM layers in the spec; MAGSIX Rotor is not Flat6.

**Design / plans (agents):**
1. `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
2. `docs/superpowers/plans/2026-09-04-cadac-kernel-hyper3.md` (execute first)
3. `docs/superpowers/plans/2026-09-04-cadac-falcon5.md`
4. `docs/superpowers/plans/2026-09-04-cadac-falcon6.md`

Implementation: Grok subagents + Grok reviewers, TDD, isolated worktree.

## Reading order for agents
1. Read this `README.md` (mandatory if present).
2. Read `UPDATES.md` (mandatory) for the change history and current state before working.
3. Do not transcribe C++ arrays/pointers; map CADAC concepts onto Python types.
