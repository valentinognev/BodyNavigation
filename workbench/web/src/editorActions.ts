import type { StoreApi } from "zustand/vanilla";
import { importFile, putCase, validateScenario } from "./api";
import type { WorkbenchState } from "./store";

export async function saveCurrentCase(api: StoreApi<WorkbenchState>): Promise<boolean> {
  const state = api.getState();
  if (state.drawerDirty) state.applyDrawerJson();
  const after = api.getState();
  if (after.drawerDirty) return false;
  if (after.program == null || after.stem == null || after.scenario == null) return false;
  try {
    const res = await putCase(after.program, after.stem, after.scenario);
    let body: { ok?: boolean; error?: unknown } = {};
    try {
      body = (await res.json()) as { ok?: boolean; error?: unknown };
    } catch {
      body = {};
    }
    if (!res.ok || body.ok === false) {
      api.getState().applyParseFail(String(body.error ?? `save ${res.status}`));
      return false;
    }
    return true;
  } catch (err) {
    api.getState().applyParseFail(err instanceof Error ? err.message : String(err));
    return false;
  }
}

export async function runValidate(api: StoreApi<WorkbenchState>): Promise<void> {
  const { drawerDirty, revision } = api.getState();
  if (drawerDirty) api.getState().applyDrawerJson();
  const after = api.getState();
  if (after.drawerDirty) return;
  if (after.scenario == null) return;
  try {
    const result = await validateScenario(after.scenario);
    if (api.getState().revision !== revision) return;
    if (!result.ok) api.getState().applyParseFail(result.error);
  } catch (err) {
    if (api.getState().revision !== revision) return;
    api.getState().applyParseFail(err instanceof Error ? err.message : String(err));
  }
}

function importedStem(name: string): string {
  const stem = name.replace(/\.(jsonc|asc)$/i, "");
  return stem || "imported";
}

export async function openImportedFile(api: StoreApi<WorkbenchState>, file: File): Promise<void> {
  const result = await importFile(file);
  if (!result.ok) {
    api.getState().applyParseFail(result.error);
    return;
  }
  api.getState().openImported(result.scenario, importedStem(file.name), result.description);
}
