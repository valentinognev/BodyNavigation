/**
 * @vitest-environment happy-dom
 */
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, expect, it } from "vitest";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
import { scenarioFromJson } from "../scenario";
import store from "../store";
import { Decks } from "./Decks";
import { Events } from "./Events";
import { Modules } from "./Modules";
import { Timing } from "./Timing";
import { Vehicles } from "./Vehicles";

let root: Root | null = null;

function mount(node: ReturnType<typeof createElement>): HTMLElement {
  const host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => {
    root?.render(node);
  });
  return host;
}

function titleOf(host: HTMLElement, label: string): string | null {
  const span = [...host.querySelectorAll("span")].find((el) => el.textContent === label);
  return span?.getAttribute("title") ?? null;
}

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  document.body.replaceChildren();
});

const scenario = scenarioFromJson({
  title: "t",
  options: {},
  modules: [{ name: "newton", phases: ["def", "exec"] }],
  timing: { int_step: 0.01, custom_step: 1 },
  end_time: 1,
  vehicles: [
    {
      type: "CRUISE3",
      name: "HYPER3",
      params: { alt: 3000 },
      events: [{ when: { time: 1 }, set: { alphax: 2 } }],
    },
  ],
});

it("vehicle, timing, module, event, and deck labels explain the field", () => {
  store.setState({ scenario, program: "hyper3", drawerDirty: false });

  const vehicles = mount(createElement(Vehicles));
  const toggle = vehicles.querySelector("button[aria-expanded]");
  act(() => {
    toggle?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(titleOf(vehicles, "type")).toBe("CADAC vehicle type token, such as CRUISE3 or PLANE6.");
  expect(titleOf(vehicles, "alt")).toBe("Vehicle altitude - m");
  expect(titleOf(vehicles, "params")).toBe("Initial values of this vehicle's module-variables.");

  const timing = mount(createElement(Timing));
  expect(titleOf(timing, "end_time")).toBe("Time when the run stops, in seconds.");
  expect(titleOf(timing, "int_step")).toBe("Integration step, in seconds.");

  const modules = mount(createElement(Modules));
  expect(titleOf(modules, "name")).toBe("Module called in this order.");
  expect(titleOf(modules, "phases")).toBe("Which parts of the module run: def, init, exec, term.");

  const events = mount(createElement(Events));
  expect(titleOf(events, "when")).toBe("Watch criteria. Keys are module-variable names.");
  expect(titleOf(events, "set")).toBe("Module-variables assigned when the event fires.");

  const decks = mount(createElement(Decks));
  expect(titleOf(decks, "aero_deck")).toBe("Aerodynamic table file.");
  expect(titleOf(decks, "srmb_deck")).toBe("SRBM table file.");
});
