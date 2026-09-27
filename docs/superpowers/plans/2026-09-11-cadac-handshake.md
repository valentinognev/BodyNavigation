# CADAC aero mapper + handshake + MISDC glue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. After **each** task: Grok non-fast reviewer (`cursor-grok-4.6-high`). After the last task: whole-plan reviewer.

**Goal:** Shared coefficient payload, per-family CADAC table schemas, auto-map with mapped/merged/missing preview, CADAC handshake session API, MISDC POST-complete after a successful MDT run.

**Architecture:** `cadac.aero_map` lives in the `cadac` package (tests without a browser). Workbench FastAPI sessions store payload until CADAC UI confirms (plan 5). MISDC adds a handshake client that POSTs CADAC when `?cadacSession=` is present. No iframe.

**Tech Stack:** Python 3.11, numpy, pytest, FastAPI. MISDC: existing FastAPI + React. Do not change MDT Fortran.

**Spec:** `docs/superpowers/specs/2026-09-11-cadac-workbench-ui-design.md` (Handshake, Mapper). **Plan 3 of 5.** Requires plan 2 API. Next: `docs/superpowers/plans/2026-09-11-aid-web.md`. Plan 5 wires CADAC Launch/preview UI.

## Global Constraints

- Do not invent F-16 `PLANE6` tables (`cx_vs_elev_alpha`, …)
- Do not invent NaN/Inf coefficients; they are **missing**
- Confirm allowed only if every **required** `look_up` name is mapped or merged
- GHAME HYPER mapper: Import-file of an already CADAC deck only (no MDT/AID polar → GHAME)
- Do not break MISDC existing `/namelist/*` and `/run` e2e
- Implementer + reviewer: `cursor-grok-4.6-high`
- TDD. No commit unless asked. `UPDATES.md` in **both** repos when that repo changes (`0.170.N` here; MISDC uses its own UPDATES)

## File map

- Create: `Python/src/cadac/aero_map/__init__.py`, `payload.py`, `schema.py`, `map.py`, `tables.py`
- Test: `Python/tests/unit/test_aero_payload.py`, `test_aero_schema.py`, `test_aero_map_sraam.py`, `test_aero_map_falcon5.py`, `test_aero_map_falcon6.py`, `test_aero_map_aim5.py`
- Modify: `workbench/api/cadac_web/app.py` (session routes)
- Test: `workbench/api/tests/test_handshake_api.py`
- MISDC: `api/misdat_io/handshake.py`, `app.py`; `web/src/handshake.ts`, `web/vite.config.ts` (no CADAC proxy required — sibling POSTs CADAC :8001 directly)
- MISDC tests: `api/tests/test_handshake.py`

---

### Task 1: Payload dataclass + for006 → payload

**Files:**
- Create: `Python/src/cadac/aero_map/payload.py`
- Test: `Python/tests/unit/test_aero_payload.py`

**Interfaces:**
- Produces:

```python
@dataclass
class AeroPayload:
    source: str  # "misdc" | "aid" | "file"
    solver: str  # "mdt" | "datcom" | "tornado" | "avl" | "flow5" | "cadac_deck"
    axes: dict   # {"mach": list[float], "alpha": list[float], "beta": list[float]}
    tables: dict[str, list]  # "cn"|"cm"|"ca"|"cl"|"cd" → 2D [n_mach][n_alpha] or 1D
    ref: dict    # sref, lref, xcg floats optional

def payload_from_mdt_rows(rows: list[dict]) -> AeroPayload:
    """MISDC parse_for006 rows with keys alpha, mach, cn, cm, ca (optional cl, cd, beta)."""
```

Grid: unique sorted mach, unique sorted alpha. `tables["cn"][i_mach][i_alpha]`. Missing cell → skip that name entirely if any NaN? Spec: NaN/Inf stay missing — omit that (mach,α) from the axis **or** leave out the whole table if any required cell is non-finite. Rule: drop non-finite points; if a mach slice is empty, drop that mach.

- [ ] **Step 1:**

```python
from cadac.aero_map.payload import payload_from_mdt_rows

def test_mdt_rows_grid():
    rows = [
        {"mach": 0.8, "alpha": 0.0, "cn": 0.1, "cm": -0.01, "ca": 0.3},
        {"mach": 0.8, "alpha": 4.0, "cn": 0.5, "cm": -0.02, "ca": 0.31},
        {"mach": 1.2, "alpha": 0.0, "cn": 0.12, "cm": -0.011, "ca": 0.4},
        {"mach": 1.2, "alpha": 4.0, "cn": 0.6, "cm": -0.03, "ca": 0.41},
    ]
    p = payload_from_mdt_rows(rows)
    assert p.source == "misdc"
    assert p.solver == "mdt"
    assert p.axes["mach"] == [0.8, 1.2]
    assert p.axes["alpha"] == [0.0, 4.0]
    assert p.tables["cn"][0][1] == 0.5

def test_nan_cn_omits_cn_table():
    rows = [{"mach": 0.5, "alpha": 0.0, "cn": float("nan"), "cm": 0.0, "ca": 0.2}]
    p = payload_from_mdt_rows(rows)
    assert "cn" not in p.tables
    assert "ca" in p.tables
```

- [ ] **Step 2–4: TDD**

- [ ] **Step 5: Commit prepare**

Also `payload_from_aid(coeff: dict) -> AeroPayload` in the same task if AID dict has `alpha`, `CL`, `CD`, `Cm` lists (1-D α at one Mach). Store `axes["mach"]` from `coeff.get("MACH")` or `[coeff.get("mach", 0.3)]`.

```python
def test_aid_lists():
    p = payload_from_aid({"alpha": [-2.0, 0.0, 4.0], "CL": [0.0, 0.2, 0.6], "CD": [0.02, 0.02, 0.04], "Cm": [0.0, -0.01, -0.03], "MACH": 0.2})
    assert p.source == "aid"
    assert p.tables["cl"][0][1] == 0.2
```

---

### Task 2: Required look_up schemas

**Files:**
- Create: `Python/src/cadac/aero_map/schema.py`
- Test: `Python/tests/unit/test_aero_schema.py`

**Interfaces:**
- Produces: `required_tables(family: str | None, vtype: str) -> tuple[str, ...]`

Exact tuples (unique names, order as first look_up in aero.py):

| (family, type) | names |
|---|---|
| (`sraam6`, `MISSILE6`) | `ca0_vs_mach`, `caa_vs_mach`, `cad_vs_mach`, `caoff_vs_mach`, `cyp_vs_mach_alpha`, `cndq_vs_mach_alpha`, `cn0_vs_mach_alpha`, `cnp_vs_mach_alpha`, `cllap_vs_mach_alpha`, `cllp_vs_mach_alpha`, `clldp_vs_mach_alpha`, `clm0_vs_mach_alpha`, `clmp_vs_mach_alpha`, `clmq_vs_mach`, `clmdq_vs_mach_alpha`, `clnp_vs_mach_alpha` |
| (`agm6`, `MISSILE6`) | `ca0_vs_mach`, `caa_vs_mach`, `cad_vs_mach`, `cndq_vs_mach`, `clmdq_vs_mach`, `clmq_vs_mach`, `cllap_vs_mach`, `clldp_vs_mach`, `cllp_vs_mach`, `cn0_vs_mach_alpha`, `cnp_vs_mach_alpha`, `clm0_vs_mach_alpha`, `clmp_vs_mach_alpha`, `cyp_vs_mach_alpha`, `clnp_vs_mach_alpha` |
| (`aim5`, `AIM5`) | `cl_aim_vs_alpha_mach`, `cd_aim_on_vs_alpha_mach`, `cd_aim_off_vs_alpha_mach` |
| (`None`, `PLANE`) | `cl_30MAC_vs_mach_alphax`, `cd_30MAC_vs_mach_alphax`, `cl_35MAC_vs_mach_alphax`, `cd_35MAC_vs_mach_alphax`, `cl_40MAC_vs_mach_alphax`, `cd_40MAC_vs_mach_alphax` |
| (`None`, `PLANE6`) | `cx_vs_elev_alpha`, `cxq_vs_alpha`, `cyr_vs_alpha`, `cyp_vs_alpha`, `cz_vs_alpha`, `czq_vs_alpha`, `cl_vs_beta_alpha`, `cldr_vs_beta_alpha`, `clda_vs_beta_alpha`, `clr_vs_alpha`, `clp_vs_alpha`, `cm_vs_elev_alpha`, `cmq_vs_alpha`, `cn_vs_beta_alpha`, `cnda_vs_beta_alpha`, `cndr_vs_beta_alpha`, `cnr_vs_alpha`, `cnp_vs_alpha` |
| (`cruise5`, `CRUISE3`) | `cd0_vs_mach`, `cl0_vs_mach`, `cla_vs_mach`, `ckk_vs_mach`, `cla0_vs_mach` |

Unknown pair → empty tuple (mapper no-ops; Import CADAC deck only).

- [ ] **Step 1: assert the AIM5 and PLANE tuples**

- [ ] **Step 2–4**

- [ ] **Step 5: Commit prepare**

---

### Task 3: Map + merge preview (SRAAM / AGM / AIM5)

**Files:**
- Create: `Python/src/cadac/aero_map/map.py`, `tables.py`
- Test: `Python/tests/unit/test_aero_map_sraam.py`, `test_aero_map_aim5.py`

**Interfaces:**
- Consumes: `AeroPayload`, `required_tables`, existing deck dict `{title, tables: [{name, dim, x1, x2?, x3?, values}]}`
- Produces:

```python
@dataclass
class PreviewRow:
    name: str
    status: str  # "mapped" | "merged" | "missing"

@dataclass
class MapResult:
    rows: list[PreviewRow]
    deck: dict  # full aero_deck JSONC body
    can_confirm: bool  # True iff no row status == "missing"

def map_payload(
    family: str | None,
    vtype: str,
    payload: AeroPayload,
    template: dict | None,
) -> MapResult:
```

SRAAM/AGM mapping from payload:
- `cn0_vs_mach_alpha`: dim 2, x1=mach, x2=alpha, values=cn grid
- `clm0_vs_mach_alpha`: cm grid
- `ca0_vs_mach`: dim 1, values = ca at alpha==0 if that column exists else mean over alpha

AIM5:
- `cl_aim_vs_alpha_mach`: dim 2, x1=alpha, x2=mach, values from `cl` else `cn` (transpose vs SRAAM)
- `cd_aim_on_vs_alpha_mach` and `cd_aim_off_vs_alpha_mach`: from `cd` else `ca`; both copies; status `mapped` (copied is still mapped)

All other required names: copy table with same `name` from `template["tables"]` → `merged`; else `missing`.

`can_confirm` False if any missing.

- [ ] **Step 1:**

```python
def test_sraam_maps_cn_and_merges_rest():
    payload = payload_from_mdt_rows([...])  # 2x2 finite cn,cm,ca
    template = json.loads(Path("tests/unit/fixtures/sraam_template.jsonc").read_text())
    # or load Python/cases/sraam6/sraam6_aero_deck.jsonc
    result = map_payload("sraam6", "MISSILE6", payload, template)
    by = {r.name: r.status for r in result.rows}
    assert by["cn0_vs_mach_alpha"] == "mapped"
    assert by["clm0_vs_mach_alpha"] == "mapped"
    assert by["ca0_vs_mach"] == "mapped"
    assert by["clmq_vs_mach"] == "merged"
    assert result.can_confirm is True

def test_missing_without_template():
    result = map_payload("sraam6", "MISSILE6", payload, None)
    assert result.can_confirm is False
    assert any(r.status == "missing" for r in result.rows)

def test_aim5_cl_from_cn():
    # payload has cn, ca, no cl/cd
    result = map_payload("aim5", "AIM5", payload, None)
    assert {r.name: r.status for r in result.rows}["cl_aim_vs_alpha_mach"] == "mapped"
```

Use real `Python/cases/sraam6/sraam6_aero_deck.jsonc` as template (do not mutate it).

- [ ] **Step 2–4**

- [ ] **Step 5: Commit prepare**

---

### Task 4: FALCON5 / CRUISE5 / FALCON6 merge-only

**Files:**
- Modify: `Python/src/cadac/aero_map/map.py`
- Test: `Python/tests/unit/test_aero_map_falcon5.py`, `test_aero_map_falcon6.py`

**Interfaces:**
- FALCON5 `PLANE`: `cl_30MAC_vs_mach_alphax` dim 2 x1=mach x2=alpha from `cl`; `cd_30MAC` from `cd`; **copy the same polar** into 35MAC and 40MAC (status `mapped` for all six)
- CRUISE5: `cd0_vs_mach` = CD at α nearest 0; `cl0_vs_mach` = CL at α nearest 0; `cla_vs_mach` = dCL/dα (deg) from two nearest α to 0 (`(cl[j]-cl[i])/(a[j]-a[i])` if |Δα|>0 else missing). `ckk_vs_mach`, `cla0_vs_mach` merge
- FALCON6: **no mapped rows** from AID polar. All required names merged from template or missing. `can_confirm` True only with full F-16 template

- [ ] **Step 1:**

```python
def test_falcon5_copies_mac():
    result = map_payload(None, "PLANE", aid_payload, None)
    assert result.can_confirm is True  # six tables all mapped
    names = {t["name"] for t in result.deck["tables"]}
    assert "cl_40MAC_vs_mach_alphax" in names

def test_falcon6_polar_cannot_confirm_without_template():
    result = map_payload(None, "PLANE6", aid_payload, None)
    assert result.can_confirm is False

def test_falcon6_template_merge_confirms():
    template = jsonc load Python/cases/falcon6/f16_aero_deck.jsonc
    result = map_payload(None, "PLANE6", aid_payload, template)
    assert result.can_confirm is True
    assert all(r.status == "merged" for r in result.rows)
```

- [ ] **Step 2–4**

- [ ] **Step 5: Commit prepare**

SAM6: map `cn0`/`clm0`/`ca0` onto `cn0_vs_mach,betax,alphax`, `clm0_vs_mach,betax,alphax`, `ca0_vs_mach,betax,alphax` with `beta=[0.0]` as x2 when payload has no beta axis; merge all other required SAM6 names from template. Test: `test_aero_map_sam6.py`.

ROCKET6: `map_payload(..., slv: int = 1)` writes `ca0slv{slv}_vs_mach`, `cn0slv{slv}_vs_mach_alpha`, `clm0slv{slv}_vs_mach_alpha` from ca/cn/cm; merge other `*slv*` names from template. Test: `test_aero_map_rocket6.py`.

---

### Task 5: CADAC handshake session API

**Files:**
- Modify: `workbench/api/cadac_web/app.py`
- Create: `workbench/api/cadac_web/handshake.py`
- Test: `workbench/api/tests/test_handshake_api.py`

**Interfaces:**
- `POST /handshake/sessions` body `{ "vehicle": str, "family": str | null, "type": str, "program": str, "stem": str }` → `{ "id": uuid, "callback": "http://127.0.0.1:8001/handshake/sessions/{id}/complete" }`
- `POST /handshake/sessions/{id}/complete` body = AeroPayload json → store; `{ "ok": true }`
- `GET /handshake/sessions/{id}` → `{ "status": "open"|"complete"|"expired", "payload"?: dict, "preview"?: MapResult as json, "can_confirm": bool }`
  - On GET after complete: load vehicle aero_deck from the case JSONC as template; `map_payload`; include preview rows
- `POST /handshake/sessions/{id}/confirm` writes `aero_deck.jsonc` next to the case (path from vehicle `aero_deck` field) **only if** `can_confirm`. Else 400
- Sessions expire 3600 s
- CORS already allows 5173/5175

- [ ] **Step 1:**

```python
def test_session_complete_preview_sraam(tmp_path):
    # use real sraam6 case; complete with mdt payload; GET can_confirm true
```

Do not overwrite committed `sraam6_aero_deck.jsonc` in the confirm test — copy case to tmp and point CASES_ROOT via env `CADAC_CASES` if you add that override in `paths.py`. **Add `CADAC_CASES` env** in this task so tests do not mutate the library.

- [ ] **Step 2–4**

- [ ] **Step 5: Commit prepare**

---

### Task 6: MISDC complete client

**Repo:** `/home/valentin/Projects/FlightSimulation/MDT/MISDC2026`

**Files:**
- Create: `api/misdat_io/handshake.py` — `def mdt_rows_to_payload(rows) -> dict` (duplicate grid logic **or** depend on cadac — **do not** add cadac as MISDC dep). Reimplement the small grid in MISDC using the same field names as `AeroPayload` JSON. Keep a unit test that a fixture of 4 rows matches the JSON shape CADAC `AeroPayload` expects (`source`, `solver`, `axes`, `tables`, `ref`).
- Modify: `api/misdat_io/app.py` — `POST /run` body optional `cadacCallback: str | None`. After successful parse_for006, if callback set, `httpx.post(callback, json=payload)` (ignore callback errors; still return MDT result)
- Modify: `web/src/api.ts` `runMdt` pass `cadacCallback` from `?cadacSession=`
- Create: `web/src/cadacSession.ts` — `cadacCallbackUrl(): string | null` reads `cadacSession` query; callback `http://127.0.0.1:8001/handshake/sessions/${id}/complete`
- Test: `api/tests/test_handshake.py` mock httpx; `web/src/cadacSession.test.ts`

**Interfaces:**
- Consumes: `parse_for006` rows
- Produces: POST body compatible with CADAC `AeroPayload`

- [ ] **Step 1: MISDC pytest** fixture rows → JSON keys `tables.cn` 2-D

- [ ] **Step 2–4**

- [ ] **Step 5: Commit prepare in MISDC repo** (separate git). Update MISDC `UPDATES.md` and `README.md` handshake query param.

Do not change MDT binary or e2e corpus assertions.

---
