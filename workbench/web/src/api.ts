import type { StoreApi } from "zustand/vanilla";
import { parseCatalog, type Catalog } from "./catalog";
import { scenarioToJson, type Scenario } from "./scenario";
import type { WorkbenchState } from "./store";

export async function fetchCatalog(): Promise<Catalog> {
  const res = await fetch("/catalog");
  if (!res.ok) {
    throw new Error(`catalog ${res.status}`);
  }
  return parseCatalog(await res.json());
}

export async function loadCatalogInto(api: StoreApi<WorkbenchState>): Promise<void> {
  try {
    const catalog = await fetchCatalog();
    api.getState().setCatalog(catalog);
  } catch {
    api.getState().setCatalogError("API unreachable");
  }
}

function caseUrl(program: string, stem: string): string {
  return `/cases/${encodeURIComponent(program)}/${encodeURIComponent(stem)}`;
}

export async function putCase(program: string, stem: string, scenario: Scenario): Promise<Response> {
  return fetch(caseUrl(program, stem), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario: scenarioToJson(scenario) }),
  });
}

export type ValidateResult = { ok: true } | { ok: false; error: string };

export type ImportResult = { ok: true; scenario: unknown } | { ok: false; error: string };

export async function validateScenario(scenario: Scenario): Promise<ValidateResult> {
  try {
    const res = await fetch("/cases/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: scenarioToJson(scenario) }),
    });
    const body = (await res.json()) as { ok?: boolean; error?: unknown };
    if (body.ok === true) return { ok: true };
    return { ok: false, error: String(body.error ?? "invalid scenario") };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

export async function importFile(file: File): Promise<ImportResult> {
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch("/cases/import", { method: "POST", body });
    const data = (await res.json()) as { ok?: boolean; scenario?: unknown; error?: unknown };
    if (data.ok === true) return { ok: true, scenario: data.scenario };
    return { ok: false, error: String(data.error ?? "import failed") };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}
