/**
 * @vitest-environment happy-dom
 */
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, expect, it } from "vitest";
import { Editor } from "./Editor";
import store from "./store";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
const initialState = { ...store.getState() };

function mount(): HTMLElement {
  const host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => {
    root?.render(createElement(Editor));
  });
  return host;
}

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  document.body.replaceChildren();
  store.setState(initialState, true);
});

it("shows a progress bar under the header while a run is in flight", () => {
  store.setState({ runInFlight: true, runProgress: 0.4 });
  const host = mount();
  const bar = host.querySelector('[role="progressbar"]');
  expect(bar).not.toBeNull();
  expect(bar?.getAttribute("aria-valuemin")).toBe("0");
  expect(bar?.getAttribute("aria-valuemax")).toBe("100");
  expect(bar?.getAttribute("aria-valuenow")).toBe("40");

  const shell = host.firstElementChild;
  const header = shell?.querySelector(":scope > header");
  expect(header).not.toBeNull();
  expect(header?.nextElementSibling).toBe(bar);
  expect(header?.contains(bar)).toBe(false);
});

it("hides the progress bar when no run is in flight", () => {
  store.setState({ runInFlight: false, runProgress: null });
  const host = mount();
  expect(host.querySelector('[role="progressbar"]')).toBeNull();
});
