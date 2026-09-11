import { describe, it, expect, vi } from "vitest";
import { parseCatalog } from "./catalog";

it("maps hyper3 climb", () => {
  const c = parseCatalog({
    programs: [{ id: "hyper3", label: "HYPER3", cases: [{ stem: "input_climb", title: "climb" }] }],
  });
  expect(c.programs[0].cases[0].stem).toBe("input_climb");
});
