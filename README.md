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

**Python target:** `Python/` — installable package `cadac`. Runtime JSONC (scenarios + decks); `cadac.io.translate.deck_asc_to_jsonc` converts CADAC `.asc` decks. Named numpy state; CADAC `integrate` / `look_up` / atmospheres. `run_loop` skips module names missing on a vehicle and seeds combus from store `com_names` before the first execute; plot CSV records vehicle slot 0 only. Optional `VehicleSpec.family`: family set → `_VEHICLE_FAMILIES[(family, type)]` only (no fallthrough to `_VEHICLE_TYPES`). Unknown-type sentinel: `"NO_SUCH_TYPE"`. Runnable: kernel + HYPER3 (`CRUISE3`/Round3) + FALCON5 (`PLANE`/Flat3) + FALCON6 (`PLANE6`/Flat6) + HYPER5 (`HYPER5`/`TARGET3`/`SATELLITE3`, Demo 4.7) + HYPER6 (`HYPER6` climb; Radar/Satellite/Ground0 not registered) + AIM5 (`AIM5` global+family, `AIRCRAFT3` family `"aim5"` only; `Python/cases/aim5/` hori). AIM5 reuses Flat3; `Aim5Flat3Newton` maps JSONC `sael*`/`dvae` onto `sbel*`/`dvbe`. E2E `tests/e2e/test_aim5_hori.py` skips if `tests/e2e/goldens/aim5/plot.csv` is absent. Seven EOM layers in the spec; MAGSIX Rotor is not Flat6.

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
