import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { scenarioFromJson } from "../scenario";
import { OverviewForm } from "./Overview";

const scenario = scenarioFromJson({
  title: "t",
  options: {},
  modules: [],
  timing: { int_step: 0.01 },
  end_time: 1,
  vehicles: [{ type: "CRUISE3", name: "HYPER3", params: {}, events: [] }],
});

it("overview shows the case description", () => {
  const html = renderToStaticMarkup(
    createElement(OverviewForm, {
      scenario,
      description: "HYPER3, 3-DOF round earth. One CRUISE3.\nTwo-phase climb.",
      applyFormPatch: () => {},
    }),
  );
  expect(html).toContain("HYPER3, 3-DOF round earth. One CRUISE3.");
  expect(html).toContain("Two-phase climb.");
  expect(html).toContain("<p");
});

it("overview labels explain the field", () => {
  const html = renderToStaticMarkup(
    createElement(OverviewForm, {
      scenario,
      description: "",
      applyFormPatch: () => {},
    }),
  );
  expect(html).toContain('title="Run title from the TITLE line."');
  expect(html).toContain('title="Seed for stochastic draws."');
  expect(html).toContain('title="Display vehicle data on the screen at scrn_step intervals."');
  expect(html).toContain("decoration-dotted");
});

it("overview omits the description block when the file has none", () => {
  const html = renderToStaticMarkup(
    createElement(OverviewForm, {
      scenario,
      description: "",
      applyFormPatch: () => {},
    }),
  );
  expect(html).not.toContain("Two-phase climb.");
  expect(html).toContain("title");
});
