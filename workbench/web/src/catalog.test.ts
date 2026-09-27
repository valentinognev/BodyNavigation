import { describe, it, expect, vi } from "vitest";
import { groupCatalog, parseCatalog } from "./catalog";

it("maps hyper3 climb", () => {
  const c = parseCatalog({
    programs: [{ id: "hyper3", label: "HYPER3", cases: [{ stem: "input_climb", title: "climb" }] }],
  });
  expect(c.programs[0].cases[0].stem).toBe("input_climb");
});

it("keeps dimension and subgroup", () => {
  const c = parseCatalog({
    programs: [
      {
        id: "magsix",
        label: "MAGSIX",
        dimension: "xz",
        subgroup: "spinner",
        cases: [{ stem: "input", title: "attitude" }],
      },
    ],
  });
  expect(c.programs[0].dimension).toBe("xz");
  expect(c.programs[0].subgroup).toBe("spinner");
});

function program(id: string, dimension: string, subgroup: string) {
  return { id, label: id.toUpperCase(), dimension, subgroup, cases: [] };
}

it("groups programs by dimension then subgroup", () => {
  const sections = groupCatalog({
    programs: [
      program("hyper3", "3", "round"),
      program("aim5", "5", "flat"),
      program("falcon5", "5", "flat"),
      program("cruise5", "5", "round"),
      program("hyper5", "5", "round"),
      program("falcon6", "6", "flat"),
      program("sam6", "6", "flat"),
      program("sraam6", "6", "flat"),
      program("agm6", "6", "flat"),
      program("hyper6", "6", "round"),
      program("rocket6", "6", "round"),
      program("magsix", "xz", "spinner"),
      program("future6", "6", "spinner"),
      program("mystery", "9", "flat"),
    ],
  });

  expect(sections.map((section) => section.label)).toEqual([
    "2D (X-Z)",
    "3 DOF",
    "5 DOF",
    "6 DOF",
    "Other",
  ]);

  const byLabel = Object.fromEntries(sections.map((section) => [section.label, section]));
  expect(byLabel["2D (X-Z)"].groups.map((group) => group.label)).toEqual(["spinner"]);
  expect(byLabel["2D (X-Z)"].groups[0].programs.map((item) => item.id)).toEqual(["magsix"]);
  expect(byLabel["3 DOF"].groups.map((group) => group.label)).toEqual(["round"]);
  expect(byLabel["3 DOF"].groups[0].programs.map((item) => item.id)).toEqual(["hyper3"]);
  expect(byLabel["5 DOF"].groups.map((group) => [group.label, group.programs.map((item) => item.id)])).toEqual([
    ["flat", ["aim5", "falcon5"]],
    ["round", ["cruise5", "hyper5"]],
  ]);
  expect(byLabel["6 DOF"].groups.map((group) => [group.label, group.programs.map((item) => item.id)])).toEqual([
    ["flat", ["falcon6", "sam6", "sraam6", "agm6"]],
    ["round", ["hyper6", "rocket6"]],
    ["spinner", ["future6"]],
  ]);
  expect(byLabel["Other"].groups[0].programs.map((item) => item.id)).toEqual(["mystery"]);
});

it("sends a round-earth planar program to Other", () => {
  const sections = groupCatalog({
    programs: [program("planar", "xz", "round")],
  });
  expect(sections.map((section) => section.label)).toEqual(["Other"]);
  expect(sections[0].groups[0].programs.map((item) => item.id)).toEqual(["planar"]);
});
