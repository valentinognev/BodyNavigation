import type { StoreApi } from "zustand/vanilla";
import { scenarioToJson } from "./scenario";
import type { PlotData, WorkbenchState } from "./store";

export const POLL_MS = 200;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

type PostRunBody = { ok?: boolean; runId?: string; error?: unknown };
type GetRunBody = {
  status?: string;
  ok?: boolean;
  progress?: unknown;
  columns?: string[];
  rows?: Record<string, number>[];
  vehicles?: PlotData["vehicles"];
  modules?: Record<string, string>;
  error?: unknown;
};

async function postCancel(runId: string): Promise<void> {
  try {
    await fetch(`/run/${runId}/cancel`, { method: "POST" });
  } catch {
    // lastPlot stays
  }
}

export async function startRun(api: StoreApi<WorkbenchState>): Promise<void> {
  const { program, stem, runInFlight } = api.getState();
  if (runInFlight) return;
  if (program == null || stem == null) return;
  if (api.getState().drawerDirty) api.getState().applyDrawerJson();
  const flushed = api.getState();
  if (flushed.drawerDirty) return;
  if (flushed.parseError != null || flushed.scenario == null) return;
  api.setState({ runInFlight: true, runProgress: 0, runAbandoned: false, runError: null });
  try {
    const payload = {
      program,
      stem,
      end_time: flushed.scenario.end_time,
      scenario: scenarioToJson(flushed.scenario),
    };
    const res = await fetch("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = (await res.json()) as PostRunBody;
    if (!body.ok || body.runId == null) {
      api.setState({ runError: String(body.error ?? "run failed") });
      return;
    }
    const runId = body.runId;
    if (api.getState().runAbandoned) {
      await postCancel(runId);
      return;
    }
    api.getState().setActiveRunId(runId);
    while (api.getState().activeRunId === runId) {
      if (api.getState().runAbandoned) {
        await postCancel(runId);
        break;
      }
      const getRes = await fetch(`/run/${runId}`);
      const st = (await getRes.json()) as GetRunBody;
      if (st.status === "running") {
        if (typeof st.progress === "number" && Number.isFinite(st.progress)) {
          api.setState({ runProgress: Math.min(1, Math.max(0, st.progress)) });
        }
        await sleep(POLL_MS);
        continue;
      }
      if (
        !api.getState().runAbandoned &&
        st.status === "done" &&
        st.ok === true &&
        Array.isArray(st.columns) &&
        Array.isArray(st.rows)
      ) {
        const plot: PlotData = {
          columns: st.columns,
          rows: st.rows,
          ...(Array.isArray(st.vehicles) ? { vehicles: st.vehicles } : {}),
          ...(st.modules != null && typeof st.modules === "object" ? { modules: st.modules } : {}),
        };
        api.getState().setLastPlot(plot);
      } else if (!api.getState().runAbandoned && st.status !== "cancelled") {
        api.setState({ runError: String(st.error ?? "run failed") });
      }
      break;
    }
  } catch (err) {
    api.setState({ runError: err instanceof Error ? err.message : String(err) });
  } finally {
    api.setState({ runInFlight: false, activeRunId: null, runProgress: null });
  }
}

export async function cancelCurrentRun(api: StoreApi<WorkbenchState>): Promise<void> {
  api.getState().setRunAbandoned(true);
  const { activeRunId } = api.getState();
  if (activeRunId == null) return;
  await postCancel(activeRunId);
}
