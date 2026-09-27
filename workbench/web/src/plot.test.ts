import { expect, it } from "vitest";
import { axisLabel, chartGeometry, defaultColumns, hasGroundTrack } from "./plot";

it("prefers alt mach", () => {
  expect(defaultColumns(["time", "FSPV1", "alt", "mach", "lonx"])).toEqual(["alt", "mach"]);
});
it("falls back", () => {
  expect(defaultColumns(["time", "foo", "bar", "baz", "qux"])).toEqual(["foo", "bar", "baz", "qux"]);
});
it("ground", () => {
  expect(hasGroundTrack(["latx", "lonx", "alt"])).toBe(true);
  expect(hasGroundTrack(["alt"])).toBe(false);
});

it("labels axes with CADAC units", () => {
  expect(axisLabel("time")).toBe("time (s)");
  expect(axisLabel("alt")).toBe("alt (m)");
  expect(axisLabel("dvbe")).toBe("dvbe (m/s)");
  expect(axisLabel("latx")).toBe("latx (deg)");
  expect(axisLabel("lonx")).toBe("lonx (deg)");
  expect(axisLabel("mach")).toBe("mach (ND)");
  expect(axisLabel("pdynmc")).toBe("pdynmc (Pa)");
  expect(axisLabel("FSPV1")).toBe("FSPV1 (m/s²)");
  expect(axisLabel("SBEG2")).toBe("SBEG2 (m)");
  expect(axisLabel("VBEG3")).toBe("VBEG3 (m/s)");
  expect(axisLabel("alphax")).toBe("alphax (deg)");
  expect(axisLabel("foo1")).toBe("foo1");
});

it("chart geometry includes tick values on both axes", () => {
  const geo = chartGeometry([0, 5, 10], [0, 50, 100], "time", "alt", 360, 220);
  expect(geo).not.toBeNull();
  expect(geo!.xLabel).toBe("time (s)");
  expect(geo!.yLabel).toBe("alt (m)");
  expect(geo!.xTicks.length).toBeGreaterThanOrEqual(2);
  expect(geo!.yTicks.length).toBeGreaterThanOrEqual(2);
  expect(geo!.xTicks.some((tick) => tick.label === "0")).toBe(true);
  expect(geo!.yTicks.some((tick) => tick.label === "0")).toBe(true);
  expect(geo!.yTicks.some((tick) => tick.label === "100")).toBe(true);
  expect(geo!.polyline.length).toBeGreaterThan(0);
  const parts = geo!.polyline.split(" ");
  const firstY = Number(parts[0].split(",")[1]);
  const lastY = Number(parts[2].split(",")[1]);
  expect(lastY).toBeLessThan(firstY);
});

it("constant series still gets numeric ticks", () => {
  const geo = chartGeometry([0, 1], [5, 5], "time", "mach", 360, 220);
  expect(geo!.yTicks.length).toBeGreaterThanOrEqual(2);
  expect(new Set(geo!.yTicks.map((tick) => tick.label)).size).toBeGreaterThan(1);
});

it("draws a grid through interior ticks", () => {
  const geo = chartGeometry([0, 5, 10], [0, 50, 100], "time", "alt", 360, 220);
  expect(geo).not.toBeNull();
  const { frame } = geo!;
  const vertical = geo!.grid.filter((line) => line.x1 === line.x2);
  const horizontal = geo!.grid.filter((line) => line.y1 === line.y2);
  expect(vertical.length).toBeGreaterThan(0);
  expect(horizontal.length).toBeGreaterThan(0);
  for (const line of vertical) {
    expect(line.y1).toBe(frame.y);
    expect(line.y2).toBeCloseTo(frame.y + frame.h);
    expect(line.x1).toBeGreaterThan(frame.x);
    expect(line.x1).toBeLessThan(frame.x + frame.w);
  }
  for (const line of horizontal) {
    expect(line.x1).toBe(frame.x);
    expect(line.x2).toBeCloseTo(frame.x + frame.w);
    expect(line.y1).toBeGreaterThan(frame.y);
    expect(line.y1).toBeLessThan(frame.y + frame.h);
  }
});

it("keeps tick labels distinct on a short axis", () => {
  const geo = chartGeometry([-80.55, -80.544], [28.43, 28.43002], "lonx", "latx", 360, 220);
  expect(geo).not.toBeNull();
  const yLabels = geo!.yTicks.map((tick) => tick.label);
  const xLabels = geo!.xTicks.map((tick) => tick.label);
  expect(new Set(yLabels).size).toBe(yLabels.length);
  expect(new Set(xLabels).size).toBe(xLabels.length);
});
