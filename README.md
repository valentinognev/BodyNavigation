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

**Python target:** `Python/` — installable package `cadac`. Runtime JSONC (scenarios + decks); `cadac.io.translate.deck_asc_to_jsonc` converts CADAC `.asc` decks. Named numpy state; CADAC `integrate` / `look_up` / atmospheres. `run_loop` skips module names missing on a vehicle and seeds combus from store `com_names` before the first execute; plot CSV records vehicle slot 0 only. Optional `VehicleSpec.family`: family set → `_VEHICLE_FAMILIES[(family, type)]` only (no fallthrough to `_VEHICLE_TYPES`). Unknown-type sentinel: `"NO_SUCH_TYPE"`. Runnable: kernel + HYPER3 (`CRUISE3`/Round3) + FALCON5 (`PLANE`/Flat3) + FALCON6 (`PLANE6`/Flat6) + HYPER5 (`HYPER5`/`TARGET3`/`SATELLITE3`, Demo 4.7) + HYPER6 (`HYPER6` climb; Radar/Satellite/Ground0 not registered) + AIM5 (`AIM5` global+family, `AIRCRAFT3` family `"aim5"` only; `Python/cases/aim5/` hori) + CRUISE5 (`CRUISE3`/`TARGET3`/`SATELLITE3` family `"cruise5"`; `Python/cases/cruise5/` input_1) + MAGSIX (`ROTOR`/Rotor EOM, not Flat6; no decks) + ROCKET6 (`HYPER6` family `"rocket6"` SLV insertion; aero required, weather deck for `mair=12`). `HYPER6` without family still maps to `Hyper6`. AIM5 reuses Flat3; `Aim5Flat3Newton` maps JSONC `sael*`/`dvae` onto `sbel*`/`dvbe`. Translate writes scenario-level and per-vehicle `"family"`; parses `WEATHER_DECK` and `GAUSS`/`RAYL`/`MARKOV`.

**Not yet in Python:** SAM6, SRAAM6, AGM6.

**Design / plans (agents):**
1. `docs/superpowers/specs/2026-09-04-cadac-python-design.md`
2. `docs/superpowers/plans/2026-09-04-cadac-kernel-hyper3.md` (done)
3. `docs/superpowers/plans/2026-09-04-cadac-falcon5.md` (done)
4. `docs/superpowers/plans/2026-09-04-cadac-falcon6.md` (done)
5. `docs/superpowers/plans/2026-09-06-cadac-hyper5.md` (done)
6. `docs/superpowers/plans/2026-09-06-cadac-hyper6.md` (done)
7. `docs/superpowers/plans/2026-09-06-cadac-aim5.md` (done) — `input_hori.asc`; introduces family dispatch
8. `docs/superpowers/plans/2026-09-06-cadac-cruise5.md` (done) — `input_1.asc`; family `cruise5`
9. `docs/superpowers/plans/2026-09-06-cadac-magsix.md` (done) — Rotor EOM; `input.asc` attitude
10. `docs/superpowers/plans/2026-09-06-cadac-rocket6.md` (done) — family `rocket6` + type `HYPER6` SLV
11. `docs/superpowers/plans/2026-09-06-cadac-sam6.md` — Flat0 + family `sam6`
12. `docs/superpowers/plans/2026-09-06-cadac-sraam6.md` — `input_1v1.asc`; family `sraam6`
13. `docs/superpowers/plans/2026-09-06-cadac-agm6.md` — free flight; family `agm6`

Implementation: Grok subagents + Grok reviewers, TDD, isolated worktree. Family API landed with AIM5; later Task 1 is idempotent.

## Reading order for agents
1. Read this `README.md` (mandatory if present).
2. Read `UPDATES.md` (mandatory) for the change history and current state before working.
3. Do not transcribe C++ arrays/pointers; map CADAC concepts onto Python types.
