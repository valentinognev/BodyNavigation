import { afterEach, expect, it, vi } from "vitest";
import { cancelCurrentRun, startRun } from "./run";
import { scenarioFromJson } from "./scenario";
import { createStore } from "./store";

const hyper3Like = {
  title: "t",
  options: { scrn: true },
  modules: [],
  timing: { int_step: 0.01 },
  end_time: 90,
  vehicles: [{ type: "CRUISE3", name: "HYPER3", params: { alt: 3000 }, events: [] }],
};

const priorPlot = {
  columns: ["time", "alt"],
  rows: [{ time: 0, alt: 1 }],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

function jsonResponse(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

it("done ok sets lastPlot", async () => {
  const plot = {
    columns: ["time", "alt"],
    rows: [
      { time: 0, alt: 10 },
      { time: 1, alt: 20 },
    ],
  };
  const fetchMock = vi.fn(async (url: string) => {
    if (url === "/run") return jsonResponse({ ok: true, runId: "r1" });
    if (url === "/run/r1") return jsonResponse({ status: "done", ok: true, ...plot });
    throw new Error(`unexpected ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  store.setState({
    program: "hyper3",
    stem: "input_climb",
    scenario: scenarioFromJson(hyper3Like),
  });
  await startRun(store);
  expect(fetchMock).toHaveBeenCalledWith(
    "/run",
    expect.objectContaining({ method: "POST" }),
  );
  const init = fetchMock.mock.calls[0][1] as RequestInit;
  const body = JSON.parse(String(init.body)) as {
    program: string;
    stem: string;
    end_time?: number;
    scenario?: { title?: string; end_time?: number };
  };
  expect(body.program).toBe("hyper3");
  expect(body.stem).toBe("input_climb");
  expect(body.end_time).toBe(90);
  expect(body.scenario?.title).toBe("t");
  expect(body.scenario?.end_time).toBe(90);
  expect(store.getState().lastPlot).toEqual(plot);
  expect(store.getState().selectedColumns).toEqual(["alt"]);
  expect(store.getState().runInFlight).toBe(false);
});

it("error does not clear lastPlot", async () => {
  const fetchMock = vi.fn(async (url: string) => {
    if (url === "/run") return jsonResponse({ ok: true, runId: "r1" });
    if (url === "/run/r1") return jsonResponse({ status: "error", ok: false, error: "timeout" });
    throw new Error(`unexpected ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  store.setState({
    program: "hyper3",
    stem: "input_climb",
    scenario: scenarioFromJson(hyper3Like),
    lastPlot: priorPlot,
  });
  await startRun(store);
  expect(store.getState().lastPlot).toEqual(priorPlot);
  expect(store.getState().runError).toBe("timeout");
  expect(store.getState().runInFlight).toBe(false);
});

it("POST ok false sets runError and keeps lastPlot", async () => {
  const fetchMock = vi.fn(async (url: string) => {
    if (url === "/run") return jsonResponse({ ok: false, error: "unknown stem" });
    throw new Error(`unexpected ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  store.setState({
    program: "hyper3",
    stem: "input_climb",
    scenario: scenarioFromJson(hyper3Like),
    lastPlot: priorPlot,
    runError: "stale",
  });
  await startRun(store);
  expect(store.getState().lastPlot).toEqual(priorPlot);
  expect(store.getState().runError).toBe("unknown stem");
  expect(store.getState().runInFlight).toBe(false);
});

it("thrown fetch sets runError and keeps lastPlot", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));
  const store = createStore();
  store.setState({
    program: "hyper3",
    stem: "input_climb",
    scenario: scenarioFromJson(hyper3Like),
    lastPlot: priorPlot,
  });
  await startRun(store);
  expect(store.getState().lastPlot).toEqual(priorPlot);
  expect(store.getState().runError).toMatch(/network down/);
  expect(store.getState().runInFlight).toBe(false);
});

it("new Run clears previous runError", async () => {
  const fetchMock = vi.fn(async (url: string) => {
    if (url === "/run") return jsonResponse({ ok: true, runId: "r1" });
    if (url === "/run/r1") return jsonResponse({ status: "done", ok: true, ...priorPlot });
    throw new Error(`unexpected ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  store.setState({
    program: "hyper3",
    stem: "input_climb",
    scenario: scenarioFromJson(hyper3Like),
    runError: "timeout",
  });
  await startRun(store);
  expect(store.getState().runError).toBeNull();
  expect(store.getState().lastPlot).toEqual(priorPlot);
});

it("startRun flushes dirty drawer into POST scenario", async () => {
  const fetchMock = vi.fn(async (url: string) => {
    if (url === "/run") return jsonResponse({ ok: true, runId: "r1" });
    if (url === "/run/r1") return jsonResponse({ status: "done", ok: true, ...priorPlot });
    throw new Error(`unexpected ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  store.setState({
    program: "hyper3",
    stem: "input_climb",
    scenario: scenarioFromJson(hyper3Like),
  });
  store.getState().setDrawerText(JSON.stringify({ ...hyper3Like, title: "from-drawer" }));
  await startRun(store);
  const init = fetchMock.mock.calls[0][1] as RequestInit;
  const body = JSON.parse(String(init.body)) as { scenario?: { title?: string } };
  expect(body.scenario?.title).toBe("from-drawer");
  expect(store.getState().drawerDirty).toBe(false);
});

it("cancel does not clear lastPlot", async () => {
  let cancelled = false;
  const fetchMock = vi.fn(async (url: string) => {
    const path = String(url);
    if (path === "/run") return jsonResponse({ ok: true, runId: "r1" });
    if (path === "/run/r1/cancel") {
      cancelled = true;
      return jsonResponse({ ok: true });
    }
    if (path === "/run/r1") {
      if (cancelled) return jsonResponse({ status: "cancelled", ok: false });
      return jsonResponse({ status: "running", ok: false });
    }
    throw new Error(`unexpected ${path}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  store.setState({
    program: "hyper3",
    stem: "input_climb",
    scenario: scenarioFromJson(hyper3Like),
    lastPlot: priorPlot,
  });
  const running = startRun(store);
  await vi.waitFor(() => expect(store.getState().activeRunId).toBe("r1"));
  await cancelCurrentRun(store);
  await running;
  expect(store.getState().lastPlot).toEqual(priorPlot);
  expect(store.getState().runInFlight).toBe(false);
  expect(fetchMock).toHaveBeenCalledWith("/run/r1/cancel", expect.objectContaining({ method: "POST" }));
});

it("cancel during POST abandons and does not set lastPlot", async () => {
  let releasePost: () => void = () => {};
  const postHeld = new Promise<void>((resolve) => {
    releasePost = resolve;
  });
  const latePlot = {
    columns: ["time", "mach"],
    rows: [{ time: 0, mach: 9 }],
  };
  const fetchMock = vi.fn(async (url: string) => {
    const path = String(url);
    if (path === "/run") {
      await postHeld;
      return jsonResponse({ ok: true, runId: "r-late" });
    }
    if (path === "/run/r-late/cancel") return jsonResponse({ ok: true });
    if (path === "/run/r-late") return jsonResponse({ status: "done", ok: true, ...latePlot });
    throw new Error(`unexpected ${path}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  store.setState({
    program: "hyper3",
    stem: "input_climb",
    scenario: scenarioFromJson(hyper3Like),
    lastPlot: priorPlot,
  });
  const running = startRun(store);
  await vi.waitFor(() => expect(store.getState().runInFlight).toBe(true));
  expect(store.getState().activeRunId).toBeNull();
  await cancelCurrentRun(store);
  releasePost();
  await running;
  expect(store.getState().lastPlot).toEqual(priorPlot);
  expect(store.getState().runInFlight).toBe(false);
  expect(fetchMock).toHaveBeenCalledWith(
    "/run/r-late/cancel",
    expect.objectContaining({ method: "POST" }),
  );
});
