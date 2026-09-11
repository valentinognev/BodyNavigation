import { afterEach, expect, it, vi } from "vitest";
import { importFile, loadCatalogInto, putCase, validateScenario } from "./api";
import { scenarioFromJson } from "./scenario";
import { createStore } from "./store";

const hyper3Like = {
  title: "t",
  options: { scrn: true, nope: true },
  modules: [],
  timing: { int_step: 0.01 },
  end_time: 90,
  vehicles: [{ type: "CRUISE3", name: "HYPER3", params: { alt: 3000 }, events: [] }],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

it("loadCatalogInto fetch throw sets catalogError not empty library", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));
  const store = createStore();
  await loadCatalogInto(store);
  expect(store.getState().catalog).toBeNull();
  expect(store.getState().catalogError).toBe("API unreachable");
});

it("loadCatalogInto empty catalog is not an error", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ programs: [] }),
    }),
  );
  const store = createStore();
  await loadCatalogInto(store);
  expect(store.getState().catalog).toEqual({ programs: [] });
  expect(store.getState().catalogError).toBeNull();
});

it("putCase PUTs scenario to /cases/{program}/{stem}", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ ok: true }),
  });
  vi.stubGlobal("fetch", fetchMock);
  await putCase("hyper3", "input_climb", scenarioFromJson(hyper3Like));
  expect(fetchMock).toHaveBeenCalledWith(
    "/cases/hyper3/input_climb",
    expect.objectContaining({ method: "PUT" }),
  );
  const init = fetchMock.mock.calls[0][1] as RequestInit;
  const body = JSON.parse(String(init.body)) as { scenario: { end_time: number; options: { nope?: boolean } } };
  expect(body.scenario.end_time).toBe(90);
  expect(body.scenario.options.nope).toBe(true);
});

it("validateScenario POSTs current scenario", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ ok: false, error: "unknown option key 'nope'" }),
  });
  vi.stubGlobal("fetch", fetchMock);
  const result = await validateScenario(scenarioFromJson(hyper3Like));
  expect(fetchMock).toHaveBeenCalledWith(
    "/cases/validate",
    expect.objectContaining({ method: "POST" }),
  );
  expect(result).toEqual({ ok: false, error: "unknown option key 'nope'" });
});

it("validateScenario fetch rejection returns ok false", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("validate down")));
  const result = await validateScenario(scenarioFromJson(hyper3Like));
  expect(result.ok).toBe(false);
  if (!result.ok) expect(result.error).toMatch(/validate down/);
});

it("importFile POSTs multipart file to /cases/import", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ ok: true, scenario: hyper3Like }),
  });
  vi.stubGlobal("fetch", fetchMock);
  const file = new File(["asc"], "input_climb.asc");
  const result = await importFile(file);
  expect(fetchMock).toHaveBeenCalledWith(
    "/cases/import",
    expect.objectContaining({ method: "POST" }),
  );
  const init = fetchMock.mock.calls[0][1] as RequestInit;
  expect(init.body).toBeInstanceOf(FormData);
  expect(result).toEqual({ ok: true, scenario: hyper3Like });
});

it("importFile ok false returns error", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ ok: false, error: "bad asc" }),
    }),
  );
  const result = await importFile(new File(["x"], "bad.asc"));
  expect(result).toEqual({ ok: false, error: "bad asc" });
});
