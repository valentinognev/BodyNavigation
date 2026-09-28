/**
 * @vitest-environment happy-dom
 */
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
import { scenarioFromJson, type Scenario } from "../scenario";
import store from "../store";
import { Decks } from "./Decks";
import { Modules } from "./Modules";
import { OverviewForm } from "./Overview";
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

function fieldControl(host: HTMLElement, label: string): HTMLSelectElement | HTMLInputElement | null {
  const span = [...host.querySelectorAll("span")].find((el) => el.textContent === label);
  const row = span?.closest("label");
  return row?.querySelector("select, input") ?? null;
}

const base = {
  title: "t",
  options: {},
  modules: [{ name: "newton", phases: ["def", "exec"] }],
  timing: { int_step: 0.01 },
  end_time: 1,
  vehicles: [
    {
      type: "CRUISE3",
      name: "HYPER3",
      aero_deck: "ghame3_aero_deck.jsonc",
      params: { alt: 3000, mprop: 1 },
      events: [],
    },
  ],
};

function load(program: string | null, extra: Record<string, unknown> = {}) {
  store.setState({
    scenario: scenarioFromJson({ ...base, ...extra }),
    program,
    stem: "input_climb",
    drawerDirty: false,
  });
}

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

it("scenario family is a dropdown of the registered families", () => {
  const scenario = scenarioFromJson(base);
  const html = mount(
    createElement(OverviewForm, {
      scenario,
      description: "",
      applyFormPatch: () => {},
    }),
  );
  const family = fieldControl(html, "family");
  expect(family?.tagName).toBe("SELECT");
  const values = [...(family as HTMLSelectElement).options].map((option) => option.value);
  expect(values).toContain("");
  expect(values).toContain("aim5");
  expect(values).toContain("agm6");
  expect(fieldControl(html, "title")?.tagName).toBe("INPUT");
});

it("vehicle type and a mode param are dropdowns; altitude stays a text field", () => {
  load("hyper3");
  const host = mount(createElement(Vehicles));
  const toggle = host.querySelector("button[aria-expanded]");
  act(() => {
    toggle?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  const type = fieldControl(host, "type");
  expect(type?.tagName).toBe("SELECT");
  expect([...(type as HTMLSelectElement).options].map((option) => option.value)).toContain("CRUISE3");
  const mprop = fieldControl(host, "mprop");
  expect(mprop?.tagName).toBe("SELECT");
  expect([...(mprop as HTMLSelectElement).options].map((option) => option.value)).toEqual([
    "0",
    "1",
    "2",
  ]);
  expect((mprop as HTMLSelectElement).value).toBe("1");
  expect(fieldControl(host, "alt")?.tagName).toBe("INPUT");
  expect(fieldControl(host, "name")?.tagName).toBe("INPUT");
});

it("a mode switch shows the mode description, not only the code", () => {
  load("hyper6", {
    vehicles: [{ type: "HYPER6", name: "H", params: { mguide: 30 }, events: [] }],
  });
  const host = mount(createElement(Vehicles));
  act(() => {
    host.querySelector("button[aria-expanded]")?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  const mguide = fieldControl(host, "mguide") as HTMLSelectElement;
  const texts = [...mguide.options].map((option) => option.text);
  expect(texts).toContain("30: line-guidance lateral, with maut 35");
  expect(texts).toContain("3: line-guidance in pitch");
  expect(texts.some((text) => /^\d+$/.test(text))).toBe(false);
});

it("a type outside the program list stays selectable", () => {
  load("hyper3", {
    vehicles: [{ type: "NOPE", name: "X", params: {}, events: [] }],
  });
  const host = mount(createElement(Vehicles));
  act(() => {
    host.querySelector("button[aria-expanded]")?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  const type = fieldControl(host, "type") as HTMLSelectElement;
  expect([...type.options].map((option) => option.value)).toContain("NOPE");
  expect(type.value).toBe("NOPE");
});

it("module name is a dropdown and phases are checkboxes", () => {
  load("hyper3");
  const host = mount(createElement(Modules));
  const name = fieldControl(host, "name");
  expect(name?.tagName).toBe("SELECT");
  expect([...(name as HTMLSelectElement).options].map((option) => option.value)).toContain("newton");
  const boxes = [...host.querySelectorAll("input[type=checkbox]")].map((box) => ({
    label: box.parentElement?.textContent?.trim(),
    checked: (box as HTMLInputElement).checked,
  }));
  expect(boxes).toEqual([
    { label: "def", checked: true },
    { label: "init", checked: false },
    { label: "exec", checked: true },
    { label: "term", checked: false },
  ]);
  const exec = [...host.querySelectorAll("input[type=checkbox]")].find(
    (box) => box.parentElement?.textContent?.trim() === "exec",
  );
  act(() => {
    exec?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  const phases = (store.getState().scenario as Scenario).modules[0]?.phases;
  expect(phases).toEqual(["def"]);
});

it("a deck path fills the row and Browse stays beside it", () => {
  load("hyper3");
  const host = mount(createElement(Decks));
  const input = fieldControl(host, "aero_deck") as HTMLInputElement;
  const line = input.parentElement;
  const button = line?.querySelector("button");
  const classes = (el: Element | null | undefined) => (el?.className ?? "").split(/\s+/);
  expect(classes(line)).toContain("min-w-0");
  expect(classes(input)).toEqual(expect.arrayContaining(["min-w-0", "flex-1"]));
  expect(classes(input)).not.toContain("w-full");
  expect(classes(button)).not.toContain("w-full");
  expect(classes(button)).toContain("shrink-0");
});

it("browse writes the dialog path into the deck field", async () => {
  load("hyper3");
  const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true, path: "/tmp/aero.jsonc" })));
  vi.stubGlobal("fetch", fetchMock);
  const host = mount(createElement(Decks));
  const input = fieldControl(host, "aero_deck") as HTMLInputElement;
  expect(input.tagName).toBe("INPUT");
  expect(input.value).toBe("ghame3_aero_deck.jsonc");
  const browse = [...host.querySelectorAll("button")].find((button) => button.textContent === "Browse");
  await act(async () => {
    browse?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(fetchMock).toHaveBeenCalledWith(
    "/browse",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        program: "hyper3",
        stem: "input_climb",
        current: "ghame3_aero_deck.jsonc",
      }),
    }),
  );
  expect(fieldControl(host, "aero_deck")?.value).toBe("/tmp/aero.jsonc");
  const vehicle = store.getState().scenario?.vehicles[0];
  expect(vehicle?.aero_deck).toBe("/tmp/aero.jsonc");
});
