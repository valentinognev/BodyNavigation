# CADAC round-trip UI (Launch / preview / confirm / Import) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. After **each** task: Grok non-fast reviewer (`cursor-grok-4.6-high`). After the last task: whole-plan reviewer.

**Goal:** CADAC workbench Launch MISDC/AID per vehicle routing, poll handshake preview, Confirm write aero_deck, and Import file through the same mapper.

**Architecture:** Uses plan 2 editor + plan 3 session API + plan 4 AID web + MISDC query param. No embedding.

**Tech Stack:** Existing workbench React/Zustand + FastAPI.

**Spec:** `docs/superpowers/specs/2026-09-11-cadac-workbench-ui-design.md` (Launch routing, Handshake UI, Import, errors). **Plan 5 of 5.** Requires plans 2–4.

## Global Constraints

- Routing table is the spec’s (program/family, type) table — copy it into `workbench/web/src/launch.ts` as data, not comments
- GHAME HYPER: no Launch (buttons hidden)
- FALCON6 Launch AID allowed; Confirm merge-only (API already enforces)
- Sibling down: message, CADAC keeps working (`fetch` Launch URL optional HEAD; if fail, toast “start MISDC on :5173” / “start AID on :5175”)
- Do not overwrite library decks in tests (`CADAC_CASES` temp)
- Implementer + reviewer: `cursor-grok-4.6-high`
- TDD. No commit unless asked

## File map

- Create: `workbench/web/src/launch.ts`, `launch.test.ts`, `PreviewModal.tsx`, `preview.test.ts`
- Modify: `workbench/web/src/forms/Vehicles.tsx` (or vehicle form), `api.ts`, `store.ts`
- Modify: `workbench/api/cadac_web/app.py` `POST /import/aero`
- Test: `workbench/api/tests/test_import_aero.py`, `workbench/web/src/launch.test.ts`

---

### Task 1: Launch routing helper

**Files:**
- Create: `workbench/web/src/launch.ts`, `launch.test.ts`

**Interfaces:**

```ts
export type Sibling = "misdc" | "aid" | null;
export function siblingFor(program: string, family: string | undefined, vtype: string): Sibling;
export function launchUrl(sibling: "misdc" | "aid", sessionId: string): string;
```

`launchUrl("misdc", id)` → `http://127.0.0.1:5173/?cadacSession=id`  
`launchUrl("aid", id)` → `http://127.0.0.1:5175/?cadacSession=id`

Exact routing (tests must include each row):

| program | family | type | sibling |
|---|---|---|---|
| aim5 | aim5 | AIM5 | misdc |
| sraam6 | sraam6 | MISSILE6 | misdc |
| agm6 | agm6 | MISSILE6 | misdc |
| sam6 | sam6 | MISSILE6 | misdc |
| sam6 | sam6 | ROCKET5 | misdc |
| rocket6 | rocket6 | HYPER6 | misdc |
| falcon5 | undefined | PLANE | aid |
| falcon6 | undefined | PLANE6 | aid |
| aim5 | aim5 | AIRCRAFT3 | aid |
| sam6 | sam6 | AIRCRAFT3 | aid |
| agm6 | agm6 | AIRCRAFT3 | aid |
| cruise5 | cruise5 | CRUISE3 | aid |
| hyper3 | undefined | CRUISE3 | null |
| hyper5 | undefined | HYPER5 | null |
| hyper6 | undefined | HYPER6 | null |
| magsix | magsix | ROTOR | null |
| sraam6 | sraam6 | TARGET3 | null |

`program` is catalog id (`hyper3`, `falcon6`, …). Compare case-insensitive.

- [ ] **Step 1: Vitest table above**

- [ ] **Step 2:** `cd workbench/web && npm test -- src/launch.test.ts` FAIL

- [ ] **Step 3: implement `siblingFor` / `launchUrl`**

- [ ] **Step 4: PASS**

- [ ] **Step 5: Commit prepare**

---

### Task 2: Launch button → create session → open tab

**Files:**
- Modify: vehicle form, `api.ts` `createSession`, `store.ts`
- Test: `workbench/web/src/launchFlow.test.ts` (mock `window.open` and `fetch`)

**Interfaces:**
- Click Launch MISDC/AID on a vehicle row: `POST /handshake/sessions` with `{ vehicle: name, family, type, program, stem }` then `window.open(launchUrl(sibling, id), "_blank")`
- If `siblingFor` is null, do not render the button
- If `fetch` createSession fails, toast the error; do not open a tab
- Store `activeSessionId` for Task 3 poll

- [ ] **Step 1: test sibling null hides buttons (pure function already); test `buildLaunchAction` returns url after mocked POST**

```ts
export async function startHandshake(args): Promise<{ id: string; url: string }> {
  const r = await fetch("/handshake/sessions", { method: "POST", body: JSON.stringify(args) });
  const { id } = await r.json();
  return { id, url: launchUrl(sibling, id) };
}
```

- [ ] **Step 2–4**

- [ ] **Step 5: Commit prepare**

---

### Task 3: Preview modal + Confirm

**Files:**
- Create: `workbench/web/src/PreviewModal.tsx`
- Modify: store poll `GET /handshake/sessions/{id}` every 2 s while `status==="open"` (stop on complete/expired)
- Test: `workbench/web/src/preview.test.ts`

**Interfaces:**
- When `status==="complete"`, show table: CADAC name, status mapped/merged/missing
- Confirm button enabled iff `can_confirm`
- Confirm → `POST /handshake/sessions/{id}/confirm` then reload vehicle decks / toast
- Expired: toast, no write
- Missing rows: red class; Confirm disabled

- [ ] **Step 1:**

```ts
import { confirmEnabled, rowClass } from "./preview";
it("disabled when missing", () => {
  expect(confirmEnabled({ can_confirm: false, rows: [{ status: "missing" }] })).toBe(false);
});
it("enabled when all mapped or merged", () => {
  expect(confirmEnabled({ can_confirm: true, rows: [{ status: "mapped" }, { status: "merged" }] })).toBe(true);
});
```

- [ ] **Step 2–4:** modal UI

- [ ] **Step 5: Commit prepare**

---

### Task 4: Import file (same mapper)

**Files:**
- Modify: `workbench/api/cadac_web/app.py` `POST /import/aero`
- Modify: vehicle form Import control
- Test: `workbench/api/tests/test_import_aero.py`

**Interfaces:**
- Body: `{ "program", "stem", "vehicleName", "payload": AeroPayload }` **or** `{ "for006": str }` for MISDC listing
- `for006` path: `misdat` parser is in another repo — **do not import MISDC**. Accept only handshake JSON payload in v1 (CADAC `AeroPayload`). Optional: user pastes JSON. File picker reads JSON.
- Runs `map_payload` + returns preview; `POST /import/aero/confirm` writes deck if `can_confirm`
- Tests use SRAAM template in `CADAC_CASES` tmp + MDT-like payload from plan 3 fixtures

- [ ] **Step 1: pytest preview mapped cn0, confirm writes dest aero_deck.jsonc in tmp**

- [ ] **Step 2–4**

- [ ] **Step 5: Commit prepare.** README: Launch/Import documented. `UPDATES.md` `0.171.0` or continue `0.170.N` — use next subver **feature** `0.171.0` when round-trip lands.

Sibling-down: `startHandshake` catch TypeError/network → toast `Start MISDC (http://127.0.0.1:5173) or AID (http://127.0.0.1:5175)`. Test the message helper.

---
