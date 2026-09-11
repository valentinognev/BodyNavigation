import { it, expect } from "vitest";
import {
  commitNumericRecord,
  commitParamValue,
  commitPhases,
  parseDrawerJson,
  scenarioFromJson,
  scenarioToJson,
} from "./scenario";
import { NAV_ITEMS } from "./nav";

it("roundtrips hyper3-like", () => {
  const raw = {
    title: "t",
    options: { scrn: true, events: true, plot: true, doc: true, csv: true },
    modules: [{ name: "newton", phases: ["def", "init", "exec"] }],
    timing: { int_step: 0.01 },
    end_time: 90,
    vehicles: [{ type: "CRUISE3", name: "HYPER3", params: { alt: 3000 }, events: [] }],
  };
  const s = scenarioFromJson(raw);
  expect(s.vehicles[0].params.alt).toBe(3000);
  expect(scenarioToJson(s).end_time).toBe(90);
});

it("keeps extra option key nope", () => {
  const raw = {
    title: "t",
    options: { scrn: true, nope: true },
    modules: [],
    timing: { int_step: 0.01 },
    end_time: 1,
    vehicles: [{ type: "CRUISE3", name: "HYPER3", params: {}, events: [] }],
  };
  const s = scenarioFromJson(raw);
  expect(scenarioToJson(s).options.nope).toBe(true);
});

it("blank number on blur deletes param key", () => {
  const next = commitParamValue({ alt: 3000, mach: 1 }, "alt", "  ");
  expect(next).toEqual({ mach: 1 });
});

it("blank number on blur deletes timing key", () => {
  expect(commitNumericRecord({ int_step: 0.01, plot_step: 0.2 }, "plot_step", "")).toEqual({
    int_step: 0.01,
  });
});

it("failed drawer parse does not yield a scenario", () => {
  const result = parseDrawerJson("{not json");
  expect(result.ok).toBe(false);
  if (!result.ok) expect(result.error.length).toBeGreaterThan(0);
});

it("drawer parse rejects vehicle missing type or name", () => {
  const result = parseDrawerJson(
    JSON.stringify({
      title: "t",
      options: {},
      modules: [],
      timing: {},
      end_time: 1,
      vehicles: [{ name: "HYPER3", params: {}, events: [] }],
    }),
  );
  expect(result.ok).toBe(false);
});

it("scenarioFromJson stays lenient and drops unparseable vehicles", () => {
  const s = scenarioFromJson({
    title: "t",
    vehicles: [{ name: "HYPER3" }],
  });
  expect(s.vehicles).toEqual([]);
});

it("commitPhases includes a phase typed after a trailing comma", () => {
  expect(commitPhases("def, init, exec, term")).toEqual(["def", "init", "exec", "term"]);
});

it("left nav labels are exact", () => {
  expect(NAV_ITEMS.map((item) => item.label)).toEqual([
    "Overview",
    "Modules",
    "Vehicles",
    "Timing",
    "Events",
    "Decks",
    "Results",
  ]);
});
