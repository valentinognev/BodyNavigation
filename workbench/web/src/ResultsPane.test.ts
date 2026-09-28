/**
 * @vitest-environment happy-dom
 */
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, expect, it } from "vitest";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
import { ResultsPane } from "./ResultsPane";
import store from "./store";

let root: Root | null = null;
const initialState = { ...store.getState() };

function mount(): HTMLElement {
  const host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => {
    root?.render(createElement(ResultsPane));
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
  store.setState(initialState, true);
});

it("shows CADAC headings, checkbox hints, and symbol axis labels", () => {
  store.setState({
    program: "aim5",
    lastPlot: {
      columns: ["time", "alt", "latx", "lonx", "foo", "SBEL1"],
      rows: [
        { time: 0, alt: 1000, latx: 1, lonx: 2, foo: 3, SBEL1: 0 },
        { time: 1, alt: 1100, latx: 1.1, lonx: 2.1, foo: 4, SBEL1: 10 },
      ],
    },
    selectedColumns: ["alt", "foo"],
  });

  const host = mount();
  const captions = [...host.querySelectorAll("p")].map((el) => el.textContent);
  expect(captions).toContain("Vehicle altitude vs time");
  expect(captions).toContain("foo vs time");
  expect(captions).toContain("Vehicle latitude vs vehicle longitude");

  expect(titleOf(host, "alt")).toBe("Vehicle altitude");
  expect(titleOf(host, "foo")).toBeNull();
  expect(titleOf(host, "SBEL1")).toBe(
    "Missile pos. wrt point E in local level axes, component 1 - m",
  );

  const altChart = [...host.querySelectorAll("svg")].find(
    (svg) => svg.getAttribute("aria-label") === "Vehicle altitude vs time",
  );
  expect(altChart?.textContent).toContain("alt (m)");
  expect(altChart?.textContent).toContain("time (s)");

  const ground = [...host.querySelectorAll("svg")].find(
    (svg) => svg.getAttribute("aria-label") === "Vehicle latitude vs vehicle longitude",
  );
  expect(ground?.textContent).toContain("latx (deg)");
  expect(ground?.textContent).toContain("lonx (deg)");
});

it("shows a geographic trajectory still and an orbit view", () => {
  store.setState({
    program: "hyper3",
    lastPlot: {
      columns: ["time", "latx", "lonx", "alt"],
      rows: [
        { time: 0, latx: 28.43, lonx: -80.55, alt: 3000 },
        { time: 1, latx: 28.44, lonx: -80.54, alt: 3200 },
      ],
    },
    selectedColumns: [],
  });
  const host = mount();
  const still = host.querySelector(
    '[aria-label="Vehicle altitude vs vehicle latitude vs vehicle longitude"]',
  );
  expect(still?.tagName).toBe("svg");
  expect(still?.textContent).toContain("lonx (deg)");
  expect(still?.textContent).toContain("latx (deg)");
  expect(still?.textContent).toContain("alt (m)");
  expect(still?.querySelector("polyline")?.getAttribute("points")?.length).toBeGreaterThan(0);
  expect(still?.querySelector("circle")?.getAttribute("fill")).toBe("#15803d");
  expect(still?.querySelector("circle")?.getAttribute("r")).toBe("2.5");
  expect(
    [...(still?.querySelectorAll("rect") ?? [])].some(
      (rect) => rect.getAttribute("fill") === "#b91c1c" && rect.getAttribute("width") === "5",
    ),
  ).toBe(true);
  const labels = [...(still?.querySelectorAll("text") ?? [])].map((el) => el.textContent);
  expect(labels).toContain("vehicle");
  expect(labels).toContain("start");
  expect(labels).toContain("end");
  expect([...host.querySelectorAll("button")].some((button) => button.textContent === "Turntable GIF")).toBe(
    true,
  );
  expect(host.querySelector("canvas")?.getAttribute("aria-label")).toBe(
    "Vehicle altitude vs vehicle latitude vs vehicle longitude, drag to orbit",
  );
  expect(host.textContent).toContain("Drag to orbit. Scroll to zoom.");
  expect([...host.querySelectorAll("button")].some((button) => button.textContent === "Record")).toBe(true);
});

it("uses the local-level sentence when the plot has SBEL only", () => {
  store.setState({
    program: "aim5",
    lastPlot: {
      columns: ["time", "SBEL1", "SBEL2", "SBEL3"],
      rows: [
        { time: 0, SBEL1: 0, SBEL2: 0, SBEL3: -1000 },
        { time: 1, SBEL1: 10, SBEL2: 20, SBEL3: -1100 },
      ],
    },
    selectedColumns: [],
  });
  const host = mount();
  const still = [...host.querySelectorAll("svg")].find((svg) => svg.textContent?.includes("-SBEL3 (m)"));
  expect(still?.getAttribute("aria-label")).toBe("Missile pos. wrt point E in local level axes");
  expect(still?.textContent).toContain("SBEL1 (m)");
  expect(still?.textContent).toContain("SBEL2 (m)");
});

it("draws every vehicle on a time chart and on the ground track", () => {
  store.setState({
    program: "aim5",
    lastPlot: {
      columns: ["time", "alt", "latx", "lonx"],
      rows: [
        { time: 0, alt: 1000, latx: 10, lonx: 20 },
        { time: 1, alt: 1100, latx: 11, lonx: 21 },
      ],
      vehicles: [
        {
          name: "Aim",
          columns: ["time", "alt", "latx", "lonx"],
          rows: [
            { time: 0, alt: 1000, latx: 10, lonx: 20 },
            { time: 1, alt: 1100, latx: 11, lonx: 21 },
          ],
        },
        {
          name: "Aircraft",
          columns: ["time", "alt", "latx", "lonx", "mach"],
          rows: [
            { time: 0, alt: 8000, latx: 12, lonx: 30, mach: 0.8 },
            { time: 1, alt: 8100, latx: 12.2, lonx: 30.4, mach: 0.9 },
          ],
        },
      ],
    },
    selectedColumns: ["alt"],
  });
  const host = mount();
  const alt = [...host.querySelectorAll("svg")].find(
    (svg) => svg.getAttribute("aria-label") === "Vehicle altitude vs time",
  );
  expect(alt?.querySelectorAll("polyline")).toHaveLength(2);
  expect(alt?.textContent).toContain("Aim");
  expect(alt?.textContent).toContain("Aircraft");
  const ground = [...host.querySelectorAll("svg")].find(
    (svg) => svg.getAttribute("aria-label") === "Vehicle latitude vs vehicle longitude",
  );
  expect(ground?.querySelectorAll("polyline")).toHaveLength(2);
  expect(ground?.textContent).toContain("Aim");
  expect(ground?.textContent).toContain("Aircraft");
  expect([...host.querySelectorAll("label")].some((label) => label.textContent === "mach")).toBe(true);
});

it("draws one curve for each missile and target", () => {
  store.setState({
    program: "sraam6",
    lastPlot: {
      columns: ["time", "SBEL1", "SBEL2", "SBEL3"],
      rows: [
        { time: 0, SBEL1: 0, SBEL2: 0, SBEL3: 0 },
        { time: 1, SBEL1: 0, SBEL2: 100, SBEL3: 0 },
      ],
      vehicles: [
        {
          name: "Missile 1",
          columns: ["SBEL1", "SBEL2", "SBEL3"],
          rows: [
            { SBEL1: 0, SBEL2: 0, SBEL3: 0 },
            { SBEL1: 0, SBEL2: 100, SBEL3: 0 },
          ],
        },
        {
          name: "Missile 2",
          columns: ["SBEL1", "SBEL2", "SBEL3"],
          rows: [
            { SBEL1: 10, SBEL2: 0, SBEL3: 0 },
            { SBEL1: 10, SBEL2: 80, SBEL3: 0 },
          ],
        },
        {
          name: "Target 1",
          columns: ["SAEL1", "SAEL2", "SAEL3"],
          rows: [
            { SAEL1: 0, SAEL2: 40, SAEL3: -1000 },
            { SAEL1: 20, SAEL2: 40, SAEL3: -1000 },
          ],
        },
        {
          name: "Target 2",
          columns: ["SAEL1", "SAEL2", "SAEL3"],
          rows: [
            { SAEL1: 0, SAEL2: 60, SAEL3: -1000 },
            { SAEL1: 30, SAEL2: 60, SAEL3: -1000 },
          ],
        },
      ],
    },
    selectedColumns: [],
  });
  const host = mount();
  const still = [...host.querySelectorAll("svg")].find((svg) => svg.textContent?.includes("Missile 1"));
  expect(still?.querySelectorAll("polyline")).toHaveLength(4);
  expect(still?.textContent).toContain("Missile 2");
  expect(still?.textContent).toContain("Target 1");
  expect(still?.textContent).toContain("Target 2");
  const strokes = [...(still?.querySelectorAll("polyline") ?? [])].map((line) => line.getAttribute("stroke"));
  expect(new Set(strokes).size).toBe(4);
});

it("omits the trajectory when the plot has no position triple", () => {
  store.setState({
    program: "hyper3",
    lastPlot: {
      columns: ["time", "alt"],
      rows: [{ time: 0, alt: 1 }],
    },
    selectedColumns: ["alt"],
  });
  const host = mount();
  expect([...host.querySelectorAll("button")].some((button) => button.textContent === "Turntable GIF")).toBe(
    false,
  );
  expect(host.querySelector("canvas")).toBeNull();
});

it("reports that recording is unavailable when the canvas cannot capture", () => {
  store.setState({
    program: "hyper3",
    lastPlot: {
      columns: ["time", "latx", "lonx", "alt"],
      rows: [
        { time: 0, latx: 1, lonx: 2, alt: 3 },
        { time: 1, latx: 1.1, lonx: 2.1, alt: 4 },
      ],
    },
    selectedColumns: [],
  });
  const host = mount();
  const record = [...host.querySelectorAll("button")].find((button) => button.textContent === "Record");
  act(() => {
    record?.click();
  });
  expect(host.textContent).toContain("Recording is unavailable");
});

it("groups curve checkboxes by physics then module", () => {
  store.setState({
    program: "aim5",
    lastPlot: {
      columns: ["time", "alt", "mach", "foo", "SBEL1"],
      rows: [{ time: 0, alt: 1, mach: 0.5, foo: 1, SBEL1: 0 }],
      modules: { alt: "newton", mach: "environment", SBEL1: "newton" },
    },
    selectedColumns: ["alt"],
  });
  const host = mount();
  expect([...host.querySelectorAll("h3")].map((el) => el.textContent)).toEqual([
    "Position",
    "Velocity",
    "Other",
  ]);
  expect([...host.querySelectorAll("h4")].map((el) => el.textContent)).toEqual([
    "newton",
    "environment",
  ]);
  expect([...host.querySelectorAll("label")].map((el) => el.textContent)).toEqual([
    "alt",
    "SBEL1",
    "mach",
    "foo",
  ]);
  const alt = [...host.querySelectorAll("label")].find((el) => el.textContent === "alt");
  expect(alt?.querySelector("input")?.checked).toBe(true);
});

it("uses the first vehicle module when vehicles disagree", () => {
  store.setState({
    program: "aim5",
    lastPlot: {
      columns: ["time", "alt"],
      rows: [{ time: 0, alt: 1 }],
      vehicles: [
        {
          name: "Aim",
          columns: ["time", "alt"],
          rows: [{ time: 0, alt: 1 }],
          modules: { alt: "newton" },
        },
        {
          name: "Aircraft",
          columns: ["time", "alt", "mach"],
          rows: [{ time: 0, alt: 2, mach: 0.8 }],
          modules: { alt: "kinematics", mach: "environment" },
        },
      ],
    },
    selectedColumns: [],
  });
  const host = mount();
  expect([...host.querySelectorAll("h4")].map((el) => el.textContent)).toEqual([
    "newton",
    "environment",
  ]);
});

it("shows physics headings when the plot has no module map", () => {
  store.setState({
    program: "aim5",
    lastPlot: {
      columns: ["time", "alt", "foo"],
      rows: [{ time: 0, alt: 1, foo: 2 }],
    },
    selectedColumns: [],
  });
  const host = mount();
  expect([...host.querySelectorAll("h3")].map((el) => el.textContent)).toEqual(["Position", "Other"]);
  expect(host.querySelectorAll("h4")).toHaveLength(0);
  expect([...host.querySelectorAll("label")].map((el) => el.textContent)).toEqual(["alt", "foo"]);
});
