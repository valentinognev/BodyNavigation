/**
 * @vitest-environment happy-dom
 */
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, expect, it } from "vitest";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
import { scenarioFromJson } from "../scenario";
import store from "../store";
import { Vehicles } from "./Vehicles";

let root: Root | null = null;

function mountVehicles(): HTMLElement {
  const host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => {
    root?.render(createElement(Vehicles));
  });
  return host;
}

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  document.body.replaceChildren();
});

it("vehicle card starts collapsed and shows type and name", () => {
  store.setState({
    scenario: scenarioFromJson({
      title: "t",
      options: {},
      modules: [],
      timing: { int_step: 0.01 },
      end_time: 1,
      vehicles: [{ type: "CRUISE3", name: "HYPER3", params: { alt: 3000 }, events: [] }],
    }),
    drawerDirty: false,
  });
  const host = mountVehicles();
  const toggle = host.querySelector("button[aria-expanded]");
  expect(toggle?.getAttribute("aria-expanded")).toBe("false");
  expect(toggle?.textContent).toContain("HYPER3");
  expect(host.textContent).not.toContain("params");
  expect(host.textContent).not.toContain("Remove");
});

it("clicking a vehicle header expands only that card", () => {
  store.setState({
    scenario: scenarioFromJson({
      title: "t",
      options: {},
      modules: [],
      timing: { int_step: 0.01 },
      end_time: 1,
      vehicles: [
        { type: "CRUISE3", name: "HYPER3", params: { alt: 3000 }, events: [] },
        { type: "PLANE6", name: "FALCON", params: { mach: 0.8 }, events: [] },
      ],
    }),
    drawerDirty: false,
  });
  const host = mountVehicles();
  const toggles = host.querySelectorAll("button[aria-expanded]");
  act(() => {
    toggles[0]?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(toggles[0]?.getAttribute("aria-expanded")).toBe("true");
  expect(toggles[1]?.getAttribute("aria-expanded")).toBe("false");
  expect(host.textContent).toContain("alt");
  expect(host.textContent).not.toContain("mach");
  const card = toggles[0]?.closest("div.rounded");
  const cardButtons = [...(card?.querySelectorAll("button") ?? [])];
  expect(cardButtons.at(-1)?.textContent).toBe("Remove");
  for (const button of cardButtons) {
    if (button.hasAttribute("aria-expanded")) continue;
    expect(button.className).toContain("bg-slate-800");
    expect(button.className).not.toContain("bg-white");
  }
  act(() => {
    toggles[0]?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(toggles[0]?.getAttribute("aria-expanded")).toBe("false");
  expect(host.textContent).not.toContain("params");
  expect(host.textContent).not.toContain("Remove");
});

it("blank type and name use Vehicle N", () => {
  store.setState({
    scenario: scenarioFromJson({
      title: "t",
      options: {},
      modules: [],
      timing: { int_step: 0.01 },
      end_time: 1,
      vehicles: [
        { type: "", name: "  ", params: {}, events: [] },
        { type: "", name: "", params: {}, events: [] },
      ],
    }),
    drawerDirty: false,
  });
  const host = mountVehicles();
  const toggles = [...host.querySelectorAll("button[aria-expanded]")];
  expect(toggles[0]?.textContent).toContain("Vehicle 1");
  expect(toggles[1]?.textContent).toContain("Vehicle 2");
});
