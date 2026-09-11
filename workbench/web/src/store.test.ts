import { it, expect, vi, afterEach } from "vitest";
import { runValidate, saveCurrentCase, openImportedFile } from "./editorActions";
import { runButtonDisabled, saveButtonDisabled } from "./nav";
import { startRun } from "./run";
import { createStore } from "./store";
import { scenarioFromJson } from "./scenario";

const hyper3Like = {
  title: "t",
  options: { scrn: true },
  modules: [],
  timing: { int_step: 0.01 },
  end_time: 90,
  vehicles: [{ type: "CRUISE3", name: "HYPER3", params: { alt: 3000 }, events: [] }],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

it("applyFormPatch applies dirty drawer first then keeps drawer title", () => {
  const store = createStore();
  store.setState({
    scenario: scenarioFromJson(hyper3Like),
    drawerDirty: false,
  });
  store.getState().setDrawerText(JSON.stringify({ ...hyper3Like, title: "drawer-title" }));
  expect(store.getState().drawerDirty).toBe(true);
  store.getState().applyFormPatch({ end_time: 12 });
  expect(store.getState().scenario?.title).toBe("drawer-title");
  expect(store.getState().scenario?.end_time).toBe(12);
  expect(store.getState().drawerDirty).toBe(false);
  expect(store.getState().parseError).toBeNull();
});

it("applyFormPatch aborts when dirty drawer still fails to parse", () => {
  const store = createStore();
  store.setState({
    scenario: scenarioFromJson(hyper3Like),
    drawerDirty: false,
  });
  store.getState().setDrawerText("{not json");
  store.getState().applyFormPatch({ title: "from-form" });
  expect(store.getState().scenario?.title).toBe("t");
  expect(store.getState().drawerDirty).toBe(true);
  expect(store.getState().parseError?.message).toBeTruthy();
});

it("applyFormPatch bumps revision and clears parseError", () => {
  const store = createStore();
  store.setState({
    scenario: scenarioFromJson(hyper3Like),
    parseError: { message: "bad" },
    revision: 2,
  });
  store.getState().applyFormPatch({ title: "u" });
  expect(store.getState().revision).toBe(3);
  expect(store.getState().parseError).toBeNull();
  expect(store.getState().scenario?.title).toBe("u");
  expect(store.getState().scenario?.end_time).toBe(90);
});

it("failed drawer parse keeps last-good scenario", () => {
  const store = createStore();
  const scenario = scenarioFromJson(hyper3Like);
  store.setState({ scenario, revision: 1, parseError: null, drawerDirty: false });
  store.getState().setDrawerText("{not json");
  store.getState().applyDrawerJson();
  expect(store.getState().scenario).toEqual(scenario);
  expect(store.getState().parseError?.message).toBeTruthy();
  expect(store.getState().drawerDirty).toBe(true);
});

it("openCase GETs case JSONC into scenario using HTTP status", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ ok: true, path: "p", scenario: hyper3Like }),
  });
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  await store.getState().openCase("hyper3", "input_climb");
  expect(fetchMock).toHaveBeenCalledWith("/cases/hyper3/input_climb");
  expect(store.getState().view).toBe("editor");
  expect(store.getState().program).toBe("hyper3");
  expect(store.getState().stem).toBe("input_climb");
  expect(store.getState().scenario?.vehicles[0].params.alt).toBe(3000);
});

it("openCase fetch rejection sets parseError", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));
  const store = createStore();
  await store.getState().openCase("hyper3", "input_climb");
  expect(store.getState().view).toBe("editor");
  expect(store.getState().scenario).toBeNull();
  expect(store.getState().parseError?.message).toMatch(/network down/);
});

it("drawer apply rejects unparseable vehicle and keeps last-good scenario", () => {
  const store = createStore();
  const scenario = scenarioFromJson(hyper3Like);
  store.setState({ scenario, revision: 1, parseError: null, drawerDirty: false });
  store.getState().setDrawerText(
    JSON.stringify({
      ...hyper3Like,
      vehicles: [{ name: "HYPER3", params: {}, events: [] }],
    }),
  );
  store.getState().applyDrawerJson();
  expect(store.getState().scenario?.vehicles[0].type).toBe("CRUISE3");
  expect(store.getState().parseError?.message).toBeTruthy();
  expect(store.getState().drawerDirty).toBe(true);
});

it("save aborts when drawer stays dirty after parse fail", async () => {
  const fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  store.setState({
    program: "hyper3",
    stem: "input_climb",
    scenario: scenarioFromJson(hyper3Like),
  });
  store.getState().setDrawerText("{not json");
  const saved = await saveCurrentCase(store);
  expect(saved).toBe(false);
  expect(fetchMock).not.toHaveBeenCalled();
  expect(store.getState().scenario?.title).toBe("t");
  expect(store.getState().drawerDirty).toBe(true);
});

it("save applies dirty drawer then PUTs", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ ok: true }),
  });
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  store.setState({
    program: "hyper3",
    stem: "input_climb",
    scenario: scenarioFromJson(hyper3Like),
  });
  store.getState().setDrawerText(JSON.stringify({ ...hyper3Like, title: "edited" }));
  const saved = await saveCurrentCase(store);
  expect(saved).toBe(true);
  expect(store.getState().drawerDirty).toBe(false);
  expect(store.getState().scenario?.title).toBe("edited");
  expect(fetchMock).toHaveBeenCalledWith(
    "/cases/hyper3/input_climb",
    expect.objectContaining({ method: "PUT" }),
  );
});

it("runValidate fetch rejection sets parseError", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("validate down")));
  const store = createStore();
  store.setState({
    scenario: scenarioFromJson(hyper3Like),
    drawerDirty: false,
    revision: 4,
    parseError: null,
  });
  await runValidate(store);
  expect(store.getState().scenario?.title).toBe("t");
  expect(store.getState().parseError?.message).toMatch(/validate down/);
});

it("openCase ignores GET body.ok when HTTP status is 404", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: false,
    status: 404,
    json: async () => ({ ok: true, scenario: hyper3Like }),
  });
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  await store.getState().openCase("hyper3", "missing");
  expect(store.getState().view).toBe("editor");
  expect(store.getState().scenario).toBeNull();
});

it("setLastPlot initializes selectedColumns from defaultColumns when empty", () => {
  const store = createStore();
  store.getState().setLastPlot({
    columns: ["time", "FSPV1", "alt", "mach", "lonx"],
    rows: [{ time: 0, FSPV1: 1, alt: 2, mach: 3, lonx: 4 }],
  });
  expect(store.getState().selectedColumns).toEqual(["alt", "mach"]);
});

it("setLastPlot keeps selectedColumns when columns set is unchanged", () => {
  const store = createStore();
  const plot = {
    columns: ["time", "alt", "mach"],
    rows: [{ time: 0, alt: 1, mach: 2 }],
  };
  store.getState().setLastPlot(plot);
  store.getState().toggleSelectedColumn("mach");
  store.getState().setLastPlot({
    columns: ["time", "alt", "mach"],
    rows: [{ time: 1, alt: 3, mach: 4 }],
  });
  expect(store.getState().selectedColumns).toEqual(["alt"]);
});

it("setLastPlot resets selectedColumns when columns set changed", () => {
  const store = createStore();
  store.getState().setLastPlot({
    columns: ["time", "alt", "mach"],
    rows: [{ time: 0, alt: 1, mach: 2 }],
  });
  store.getState().toggleSelectedColumn("mach");
  store.getState().setLastPlot({
    columns: ["time", "foo", "bar", "baz", "qux"],
    rows: [{ time: 0, foo: 1, bar: 2, baz: 3, qux: 4 }],
  });
  expect(store.getState().selectedColumns).toEqual(["foo", "bar", "baz", "qux"]);
});

it("openImported loads scenario without cataloguing", () => {
  const store = createStore();
  store.getState().openImported(hyper3Like, "imported");
  expect(store.getState().view).toBe("editor");
  expect(store.getState().program).toBeNull();
  expect(store.getState().stem).toBe("imported");
  expect(store.getState().scenario?.vehicles[0].type).toBe("CRUISE3");
  expect(store.getState().scenario?.vehicles[0].params.alt).toBe(3000);
});

it("openImportedFile POSTs import then loads editor", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ ok: true, scenario: hyper3Like }),
  });
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  await openImportedFile(store, new File(["asc"], "input_climb.asc"));
  expect(fetchMock).toHaveBeenCalledWith(
    "/cases/import",
    expect.objectContaining({ method: "POST" }),
  );
  expect(store.getState().view).toBe("editor");
  expect(store.getState().program).toBeNull();
  expect(store.getState().stem).toBe("input_climb");
  expect(store.getState().scenario?.vehicles[0].type).toBe("CRUISE3");
});

it("imported case disables Save/Run and does not POST", async () => {
  const fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  const store = createStore();
  store.getState().openImported(hyper3Like, "imported");
  expect(store.getState().program).toBeNull();
  expect(saveButtonDisabled(store.getState().program)).toBe(true);
  expect(runButtonDisabled(store.getState().parseError, store.getState().runInFlight, store.getState().program)).toBe(true);
  await saveCurrentCase(store);
  await startRun(store);
  expect(fetchMock).not.toHaveBeenCalled();
});

it("openImportedFile fail stays on start with parseError", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ ok: false, error: "translate failed" }),
    }),
  );
  const store = createStore();
  await openImportedFile(store, new File(["x"], "bad.asc"));
  expect(store.getState().view).toBe("start");
  expect(store.getState().scenario).toBeNull();
  expect(store.getState().parseError?.message).toMatch(/translate failed/);
});
