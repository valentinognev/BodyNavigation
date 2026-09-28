import { expect, it } from "vitest";
import {
  axisTickMarks,
  centeredCloud,
  dataAxisLines,
  dragOrbit,
  legendHudRows,
  markerSize,
  labelClearance,
  labelInk,
  labelOffsetDistance,
  opaqueClearColor,
  orbitCamera,
  projectTurntable,
  projectSceneFrames,
  sceneFrame,
  sceneOf,
  sceneOfTracks,
  sceneOfTracksUpTo,
  formatSliderTime,
  shouldResetOrbit,
  shouldStopRecorder,
  sliderTimes,
  startFramePump,
  tickFraction,
  trackPositionAt,
  trajectoryOf,
  truncateTrajectory,
  wheelOrbit,
} from "./plot3d";

it("prefers longitude, latitude, and altitude over local level", () => {
  const path = trajectoryOf(
    ["time", "latx", "lonx", "alt", "SBEL1", "SBEL2", "SBEL3"],
    [{ latx: 1, lonx: 2, alt: 3, SBEL1: 9, SBEL2: 8, SBEL3: 7 }],
  );
  expect(path).toEqual({
    kind: "geographic",
    xName: "lonx",
    yName: "latx",
    zName: "alt",
    points: [{ x: 2, y: 1, z: 3 }],
    times: [0],
  });
});

it("maps local level to east, north, and up", () => {
  const path = trajectoryOf(["SBEL1", "SBEL2", "SBEL3"], [{ SBEL1: 10, SBEL2: 20, SBEL3: -5 }]);
  expect(path).toEqual({
    kind: "local",
    xName: "SBEL2",
    yName: "SBEL1",
    zName: "-SBEL3",
    points: [{ x: 20, y: 10, z: 5 }],
    times: [0],
  });
});

it("maps aircraft local level the same way as missile local level", () => {
  const path = trajectoryOf(["SAEL1", "SAEL2", "SAEL3"], [{ SAEL1: 10, SAEL2: 20, SAEL3: -5 }]);
  expect(path).toEqual({
    kind: "local",
    xName: "SAEL2",
    yName: "SAEL1",
    zName: "-SAEL3",
    points: [{ x: 20, y: 10, z: 5 }],
    times: [0],
  });
});

it("returns null without a full position triple", () => {
  expect(trajectoryOf(["alt", "latx"], [{ alt: 1, latx: 2 }])).toBeNull();
  expect(trajectoryOf(["SBEL1", "SBEL2"], [{ SBEL1: 1, SBEL2: 2 }])).toBeNull();
});

it("drops rows that are not finite", () => {
  const path = trajectoryOf(
    ["latx", "lonx", "alt"],
    [
      { latx: 1, lonx: 2, alt: Number.NaN },
      { latx: 1, lonx: 2, alt: 3 },
    ],
  );
  expect(path?.points).toEqual([{ x: 2, y: 1, z: 3 }]);
  expect(path?.times).toEqual([1]);
});

it("returns null when every position row is non-finite", () => {
  expect(trajectoryOf(["latx", "lonx", "alt"], [{ latx: 1, lonx: 2, alt: Number.NaN }])).toBeNull();
});

it("places a greater altitude higher in the still", () => {
  const [frame] = projectTurntable(
    [
      { x: 0, y: 0, z: 0 },
      { x: 0, y: 0, z: 100 },
    ],
    [0],
    200,
    200,
  );
  expect(frame[1].sy).toBeLessThan(frame[0].sy);
});

it("reverses the east direction after half a turn", () => {
  const points = [
    { x: 0, y: 0, z: 0 },
    { x: 10, y: 0, z: 0 },
  ];
  const [east, west] = projectTurntable(points, [0, 180], 200, 200);
  expect(east[1].sx).toBeGreaterThan(east[0].sx);
  expect(west[1].sx).toBeLessThan(west[0].sx);
});

it("uses one scale for the whole turn so an edge-on frame does not enlarge the path", () => {
  const points = [
    { x: 0, y: 0, z: 0 },
    { x: 100, y: 0, z: 0 },
    { x: 0, y: 0, z: 10 },
  ];
  const [, edge] = projectTurntable(points, [0, 90], 200, 200);
  const zEdge = Math.abs(edge[2].sy - edge[0].sy);
  const [fitted] = projectTurntable(
    [
      { x: 0, y: 0, z: 0 },
      { x: 0, y: 0, z: 10 },
    ],
    [0],
    200,
    200,
  );
  const zFitted = Math.abs(fitted[1].sy - fitted[0].sy);
  expect(zEdge).toBeLessThan(zFitted * 0.5);
});

it("places the camera on the axes of a sphere", () => {
  expect(orbitCamera(0, 0, 10)).toEqual({ x: 10, y: 0, z: 0 });
  const side = orbitCamera(90, 0, 10);
  expect(side.x).toBeCloseTo(0);
  expect(side.y).toBeCloseTo(10);
  expect(side.z).toBeCloseTo(0);
  expect(orbitCamera(0, 90, 10).z).toBeCloseTo(10);
});

it("paints and requests a capture frame on each recording tick, then stops", () => {
  let queued: (() => void) | null = null;
  let scheduledId = 0;
  const events: string[] = [];
  const stop = startFramePump(
    () => events.push("draw"),
    () => events.push("request"),
    (tick) => {
      queued = tick;
      scheduledId += 1;
      return scheduledId;
    },
    (id) => events.push(`cancel:${id}`),
  );
  expect(queued).not.toBeNull();
  queued!();
  expect(events).toEqual(["draw", "request"]);
  stop();
  expect(events).toContain("cancel:2");
  const stoppedAt = events.length;
  queued!();
  expect(events.length).toBe(stoppedAt);
});

it("turns and zooms the camera from pointer motion", () => {
  const turned = dragOrbit({ azimuth: 0, elevation: 20, distance: 8 }, -40, 10);
  expect(turned).toEqual({ azimuth: 20, elevation: 15, distance: 8 });
  const zoomed = wheelOrbit({ azimuth: 0, elevation: 20, distance: 8 }, 1);
  expect(zoomed.distance).toBeCloseTo(8.8);
});

it("returns null for a trajectory with no points", () => {
  expect(
    sceneOf({
      kind: "geographic",
      xName: "lonx",
      yName: "latx",
      zName: "alt",
      points: [],
      times: [],
    }),
  ).toBeNull();
});

it("keeps a single-point scene to one series and only a start marker", () => {
  const scene = sceneOf({
    kind: "local",
    xName: "SBEL2",
    yName: "SBEL1",
    zName: "-SBEL3",
    points: [{ x: 1, y: 2, z: 3 }],
    times: [0],
  });
  expect(scene).not.toBeNull();
  expect(scene!.axes).toEqual([
    { name: "SBEL2", min: 0.9, max: 1.05, ticks: [0.9, 0.95, 1, 1.05] },
    { name: "SBEL1", min: 1.8, max: 2.1, ticks: [1.8, 1.9, 2, 2.1] },
    { name: "-SBEL3", min: 2.8, max: 3.2, ticks: [2.8, 2.9, 3, 3.1, 3.2] },
  ]);
  const at = scene!.series[0].points[0];
  expect(at.x).toBeCloseTo(2 / 3);
  expect(at.y).toBeCloseTo(2 / 3);
  expect(at.z).toBe(0.5);
  expect(scene!.markers).toEqual([{ kind: "start", label: "start", point: at }]);
  expect(scene!.series[0]).toMatchObject({ name: "vehicle", color: "#0f766e" });
});

it("covers a two-point geographic path with x/y/z ticks and a three-entry legend", () => {
  const scene = sceneOf({
    kind: "geographic",
    xName: "lonx",
    yName: "latx",
    zName: "alt",
    points: [
      { x: 0, y: 10, z: 100 },
      { x: 20, y: 30, z: 500 },
    ],
    times: [0, 1],
  });
  expect(scene).not.toBeNull();
  expect(scene!.series).toEqual([
    {
      name: "vehicle",
      color: "#0f766e",
      points: [
        { x: 0, y: 0, z: 0 },
        { x: 1, y: 1, z: 1 },
      ],
    },
  ]);
  expect(scene!.markers).toEqual([
    { kind: "start", label: "start", point: { x: 0, y: 0, z: 0 } },
    { kind: "end", label: "end", point: { x: 1, y: 1, z: 1 } },
  ]);
  expect(scene!.axes).toEqual([
    { name: "lonx", min: 0, max: 20, ticks: [0, 5, 10, 15, 20] },
    { name: "latx", min: 10, max: 30, ticks: [10, 15, 20, 25, 30] },
    { name: "alt", min: 100, max: 500, ticks: [100, 200, 300, 400, 500] },
  ]);
  expect(scene!.legend).toEqual([
    { label: "vehicle", color: "#0f766e", kind: "line" },
    { label: "start", color: "#15803d", kind: "start" },
    { label: "end", color: "#b91c1c", kind: "end" },
  ]);
});

it("draws every vehicle on one shared scale", () => {
  const missile = trajectoryOf(
    ["SBEL1", "SBEL2", "SBEL3"],
    [
      { SBEL1: 0, SBEL2: 0, SBEL3: 0 },
      { SBEL1: 0, SBEL2: 100, SBEL3: 0 },
    ],
  );
  const target = trajectoryOf(
    ["SAEL1", "SAEL2", "SAEL3"],
    [
      { SAEL1: 0, SAEL2: 0, SAEL3: 0 },
      { SAEL1: 0, SAEL2: 10, SAEL3: 0 },
    ],
  );
  const scene = sceneOfTracks([
    { name: "Missile 1", trajectory: missile! },
    { name: "Target 1", trajectory: target! },
  ]);
  expect(scene).not.toBeNull();
  expect(scene!.series.map((item) => item.name)).toEqual(["Missile 1", "Target 1"]);
  expect(scene!.series[0].color).not.toBe(scene!.series[1].color);
  expect(scene!.series[0].points[1].x).toBeCloseTo(1);
  expect(scene!.series[1].points[1].x).toBeCloseTo(0.1);
  expect(scene!.legend.map((item) => item.label)).toEqual(["Missile 1", "Target 1", "start", "end"]);
  expect(scene!.markers.filter((marker) => marker.kind === "start")).toHaveLength(2);
  expect(scene!.markers.filter((marker) => marker.kind === "end")).toHaveLength(2);
});

it("puts a custom series name on the series and the line legend", () => {
  const scene = sceneOf(
    {
      kind: "geographic",
      xName: "lonx",
      yName: "latx",
      zName: "alt",
      points: [
        { x: 0, y: 0, z: 0 },
        { x: 1, y: 1, z: 1 },
      ],
      times: [0, 1],
    },
    "slot 0",
  );
  expect(scene).not.toBeNull();
  expect(scene!.series[0].name).toBe("slot 0");
  expect(scene!.legend[0]).toEqual({ label: "slot 0", color: "#0f766e", kind: "line" });
});

const GEO_SCENE = sceneOf({
  kind: "geographic",
  xName: "lonx",
  yName: "latx",
  zName: "alt",
  points: [
    { x: 0, y: 10, z: 100 },
    { x: 20, y: 30, z: 500 },
  ],
  times: [0, 1],
})!;

it("keeps comparable x and z pixel extent in sceneFrame after mixed-unit input", () => {
  const scene = sceneOf({
    kind: "geographic",
    xName: "lonx",
    yName: "latx",
    zName: "alt",
    points: [
      { x: 0, y: 0, z: 0 },
      { x: 100, y: 0, z: 0 },
      { x: 0, y: 0, z: 10 },
    ],
    times: [0, 1, 2],
  })!;
  const path = sceneFrame(scene, 0, 200, 200).segments.find((segment) => segment.color === "line")!.points;
  const xSpan = Math.abs(path[1].sx - path[0].sx);
  const zSpan = Math.abs(path[2].sy - path[0].sy);
  expect(xSpan).toBeGreaterThan(0);
  expect(zSpan / xSpan).toBeGreaterThan(0.5);
});

it("projects axis endpoints and ticks inside the sceneFrame", () => {
  const frame = sceneFrame(GEO_SCENE, 45, 200, 200);
  expect(frame.axes.map((axis) => axis.name)).toEqual(["lonx", "latx", "alt"]);
  expect(frame.axes.map((axis) => axis.label)).toEqual(["lonx (deg)", "latx (deg)", "alt (m)"]);
  expect(frame.axes[0].ticks.map((tick) => tick.label)).toEqual(["0", "5", "10", "15", "20"]);
  expect(frame.axes[1].ticks.map((tick) => tick.label)).toEqual(["10", "15", "20", "25", "30"]);
  expect(frame.axes[2].ticks.map((tick) => tick.label)).toEqual(["100", "200", "300", "400", "500"]);
  for (const axis of frame.axes) {
    for (const point of [axis.from, axis.to, ...axis.ticks.map((tick) => tick.at)]) {
      expect(point.sx).toBeGreaterThanOrEqual(0);
      expect(point.sx).toBeLessThanOrEqual(200);
      expect(point.sy).toBeGreaterThanOrEqual(0);
      expect(point.sy).toBeLessThanOrEqual(200);
    }
  }
});

it("passes the scene legend through sceneFrame", () => {
  expect(sceneFrame(GEO_SCENE, 45, 200, 200).legend).toEqual([
    { label: "vehicle", color: "#0f766e", kind: "line" },
    { label: "start", color: "#15803d", kind: "start" },
    { label: "end", color: "#b91c1c", kind: "end" },
  ]);
});

it("places sceneFrame markers on the path endpoints", () => {
  const frame = sceneFrame(GEO_SCENE, 45, 200, 200);
  const path = frame.segments.find((segment) => segment.color === "line")!.points;
  const start = frame.segments.find((segment) => segment.color === "start")!.points[0];
  const end = frame.segments.find((segment) => segment.color === "end")!.points[0];
  expect(start).toEqual(path[0]);
  expect(end).toEqual(path[path.length - 1]);
});

it("shares one turntable scale so a pure z-step has the same pixel length at every azimuth", () => {
  const scene = sceneOf({
    kind: "local",
    xName: "SBEL2",
    yName: "SBEL1",
    zName: "-SBEL3",
    points: [
      { x: 0, y: 0, z: 0 },
      { x: 100, y: 0, z: 0 },
      { x: 100, y: 100, z: 1 },
    ],
    times: [0, 1, 2],
  })!;
  const [face, diagonal] = projectSceneFrames(scene, [0, 45], 200, 200);
  const zSpan = (frame: (typeof face)!) => Math.abs(frame.axes[2].to.sy - frame.axes[2].from.sy);
  const faceZ = zSpan(face);
  const diagonalZ = zSpan(diagonal);
  expect(faceZ).toBeGreaterThan(0);
  expect(diagonalZ).toBe(faceZ);
});

it("gives data-space axis lines from the scene box origin to each max", () => {
  expect(dataAxisLines(GEO_SCENE.axes)).toEqual([
    { name: "lonx", from: { x: 0, y: 0, z: 0 }, to: { x: 1, y: 0, z: 0 } },
    { name: "latx", from: { x: 0, y: 0, z: 0 }, to: { x: 0, y: 1, z: 0 } },
    { name: "alt", from: { x: 0, y: 0, z: 0 }, to: { x: 0, y: 0, z: 1 } },
  ]);
});

it("places a tick by linear fraction along the span, or zero when min equals max", () => {
  expect(tickFraction(5, 0, 20)).toBe(0.25);
  expect(tickFraction(0, 0, 20)).toBe(0);
  expect(tickFraction(20, 0, 20)).toBe(1);
  expect(tickFraction(7, 7, 7)).toBe(0);
});

it("puts tick marks at lerp points with 2% radius crosses and formatTick labels", () => {
  const xAxis = GEO_SCENE.axes[0];
  const from = { x: 0, y: 10, z: 100 };
  const to = { x: 20, y: 10, z: 100 };
  const marks = axisTickMarks(xAxis, from, to, 0, 100);
  expect(marks).toEqual([
    {
      at: { x: 0, y: 10, z: 100 },
      label: "0",
      crosses: [
        { from: { x: 0, y: 9, z: 100 }, to: { x: 0, y: 11, z: 100 } },
        { from: { x: 0, y: 10, z: 99 }, to: { x: 0, y: 10, z: 101 } },
      ],
    },
    {
      at: { x: 5, y: 10, z: 100 },
      label: "5",
      crosses: [
        { from: { x: 5, y: 9, z: 100 }, to: { x: 5, y: 11, z: 100 } },
        { from: { x: 5, y: 10, z: 99 }, to: { x: 5, y: 10, z: 101 } },
      ],
    },
    {
      at: { x: 10, y: 10, z: 100 },
      label: "10",
      crosses: [
        { from: { x: 10, y: 9, z: 100 }, to: { x: 10, y: 11, z: 100 } },
        { from: { x: 10, y: 10, z: 99 }, to: { x: 10, y: 10, z: 101 } },
      ],
    },
    {
      at: { x: 15, y: 10, z: 100 },
      label: "15",
      crosses: [
        { from: { x: 15, y: 9, z: 100 }, to: { x: 15, y: 11, z: 100 } },
        { from: { x: 15, y: 10, z: 99 }, to: { x: 15, y: 10, z: 101 } },
      ],
    },
    {
      at: { x: 20, y: 10, z: 100 },
      label: "20",
      crosses: [
        { from: { x: 20, y: 9, z: 100 }, to: { x: 20, y: 11, z: 100 } },
        { from: { x: 20, y: 10, z: 99 }, to: { x: 20, y: 10, z: 101 } },
      ],
    },
  ]);
  const yMarks = axisTickMarks(GEO_SCENE.axes[1], { x: 0, y: 10, z: 100 }, { x: 0, y: 30, z: 100 }, 1, 100);
  expect(yMarks[1]).toEqual({
    at: { x: 0, y: 15, z: 100 },
    label: "15",
    crosses: [
      { from: { x: -1, y: 15, z: 100 }, to: { x: 1, y: 15, z: 100 } },
      { from: { x: 0, y: 15, z: 99 }, to: { x: 0, y: 15, z: 101 } },
    ],
  });
  const zMarks = axisTickMarks(GEO_SCENE.axes[2], { x: 0, y: 10, z: 100 }, { x: 0, y: 10, z: 500 }, 2, 100);
  expect(zMarks[0]).toEqual({
    at: { x: 0, y: 10, z: 100 },
    label: "100",
    crosses: [
      { from: { x: -1, y: 10, z: 100 }, to: { x: 1, y: 10, z: 100 } },
      { from: { x: 0, y: 9, z: 100 }, to: { x: 0, y: 11, z: 100 } },
    ],
  });
});

it("centers the cloud on the mean and uses hypot radius, or 1 when empty", () => {
  expect(centeredCloud([])).toEqual({ center: { x: 0, y: 0, z: 0 }, radius: 1, cloud: [] });
  expect(
    centeredCloud([
      { x: 0, y: 10, z: 100 },
      { x: 20, y: 30, z: 500 },
    ]),
  ).toEqual({
    center: { x: 10, y: 20, z: 300 },
    radius: Math.sqrt(40200),
    cloud: [
      { x: -10, y: -10, z: -200 },
      { x: 10, y: 10, z: 200 },
    ],
  });
});

it("sizes markers at 5 percent of the scene radius", () => {
  expect(markerSize(100)).toBe(5);
  expect(markerSize(1)).toBe(0.05);
});

it("lays out legend rows in the top-left of a 360x280 HUD", () => {
  const rows = legendHudRows(GEO_SCENE.legend, 360, 280);
  expect(rows).toEqual([
    {
      label: "vehicle",
      color: "#0f766e",
      kind: "line",
      chip: { x: 8, y: 8, w: 12, h: 10 },
      text: { x: 24, y: 13 },
    },
    {
      label: "start",
      color: "#15803d",
      kind: "start",
      chip: { x: 8, y: 30, w: 12, h: 10 },
      text: { x: 24, y: 35 },
    },
    {
      label: "end",
      color: "#b91c1c",
      kind: "end",
      chip: { x: 8, y: 52, w: 12, h: 10 },
      text: { x: 24, y: 57 },
    },
  ]);
  for (const row of rows) {
    expect(row.chip.x).toBeGreaterThanOrEqual(0);
    expect(row.chip.y).toBeGreaterThanOrEqual(0);
    expect(row.chip.x + row.chip.w).toBeLessThan(360);
    expect(row.chip.y + row.chip.h).toBeLessThan(280);
    expect(row.text.x).toBeGreaterThan(row.chip.x);
    expect(row.text.y).toBeLessThan(280);
  }
});

it("does not stop an inactive recorder", () => {
  expect(shouldStopRecorder("inactive")).toBe(false);
  expect(shouldStopRecorder("recording")).toBe(true);
  expect(shouldStopRecorder("paused")).toBe(true);
});

it("offsets a label so its box clears the anchor", () => {
  expect(labelOffsetDistance(1, 0, 40, 16, 6)).toBe(26);
  expect(labelOffsetDistance(0, -2, 40, 16, 6)).toBe(14);
  expect(labelClearance({ x: 10, y: 20 }, { x: 10, y: 0 }, 14)).toEqual({ x: 10, y: 6 });
});

it("paints plot labels light on a dark canvas and dark on a light canvas", () => {
  expect(labelInk("rgb(15, 23, 42)")).toEqual({ fill: "#f8fafc", stroke: "#020617" });
  expect(labelInk("rgb(255, 255, 255)")).toEqual({ fill: "#1e293b", stroke: "#ffffff" });
  expect(labelInk("#0f172a")).toEqual({ fill: "#f8fafc", stroke: "#020617" });
  expect(labelInk("transparent")).toEqual({ fill: "#1e293b", stroke: "#ffffff" });
  expect(labelInk(undefined)).toEqual({ fill: "#1e293b", stroke: "#ffffff" });
});

it("uses the canvas CSS background as an opaque clear color, or white when transparent", () => {
  expect(opaqueClearColor("rgb(15, 23, 42)")).toBe("rgb(15, 23, 42)");
  expect(opaqueClearColor("rgb(255, 255, 255)")).toBe("rgb(255, 255, 255)");
  expect(opaqueClearColor("transparent")).toBe("#ffffff");
  expect(opaqueClearColor("rgba(0, 0, 0, 0)")).toBe("#ffffff");
  expect(opaqueClearColor(undefined)).toBe("#ffffff");
  expect(opaqueClearColor("")).toBe("#ffffff");
});

it("normalizes a thin geographic scene so x/y extent matches z", () => {
  const scene = sceneOf({
    kind: "geographic",
    xName: "lonx",
    yName: "latx",
    zName: "alt",
    points: [
      { x: -80.55, y: 28.43, z: 3000 },
      { x: -80.546, y: 28.434, z: 20000 },
    ],
    times: [0, 1],
  });
  expect(scene).not.toBeNull();
  const xs = scene!.series[0].points.map((point) => point.x);
  const ys = scene!.series[0].points.map((point) => point.y);
  const zs = scene!.series[0].points.map((point) => point.z);
  const extent = (values: number[]) => Math.max(...values) - Math.min(...values);
  const zExtent = extent(zs);
  expect(extent(xs) / zExtent).toBeGreaterThan(0.5);
  expect(extent(ys) / zExtent).toBeGreaterThan(0.5);
});

it("normalizes a local scene onto the padded tick range", () => {
  const scene = sceneOf({
    kind: "local",
    xName: "SBEL2",
    yName: "SBEL1",
    zName: "-SBEL3",
    points: [
      { x: 10, y: 20, z: 30 },
      { x: 50, y: 80, z: 90 },
    ],
    times: [0, 1],
  });
  expect(scene!.axes[0]).toEqual({ name: "SBEL2", min: 10, max: 50, ticks: [10, 20, 30, 40, 50] });
  expect(scene!.axes[2]).toEqual({ name: "-SBEL3", min: 20, max: 100, ticks: [20, 40, 60, 80, 100] });
  expect(scene!.series[0].points).toEqual([
    { x: 0, y: 0, z: 0.125 },
    { x: 1, y: 1, z: 0.875 },
  ]);
});

it("places samples on the padded nice axis, not the raw data min", () => {
  const scene = sceneOf({
    kind: "geographic",
    xName: "lonx",
    yName: "latx",
    zName: "alt",
    points: [
      { x: 0, y: 0, z: 3000 },
      { x: 1, y: 1, z: 20000 },
    ],
    times: [0, 1],
  });
  expect(scene).not.toBeNull();
  expect(scene!.axes[2]).toEqual({
    name: "alt",
    min: 0,
    max: 20000,
    ticks: [0, 5000, 10000, 15000, 20000],
  });
  expect(scene!.series[0].points).toEqual([
    { x: 0, y: 0, z: 0.15 },
    { x: 1, y: 1, z: 1 },
  ]);
  expect(scene!.markers[0].point.z).toBe(0.15);
});

const TIMED = {
  kind: "geographic" as const,
  xName: "lonx",
  yName: "latx",
  zName: "alt",
  points: [
    { x: 0, y: 0, z: 0 },
    { x: 10, y: 10, z: 100 },
    { x: 20, y: 20, z: 200 },
  ],
  times: [0, 1, 2],
};

it("collects one sorted unique time domain across vehicles", () => {
  const other = { ...TIMED, times: [1, 3] };
  expect(
    sliderTimes([
      { name: "a", trajectory: TIMED },
      { name: "b", trajectory: other },
    ]),
  ).toEqual([0, 1, 2, 3]);
});

it("interpolates the vehicle position at an exact moment", () => {
  expect(trackPositionAt(TIMED, 0.5)).toEqual({ x: 5, y: 5, z: 50 });
  expect(trackPositionAt(TIMED, -1)).toEqual({ x: 0, y: 0, z: 0 });
  expect(trackPositionAt(TIMED, 9)).toEqual({ x: 20, y: 20, z: 200 });
  expect(trackPositionAt({ ...TIMED, points: [], times: [] }, 1)).toBeNull();
});

it("truncates to the prefix plus the interpolated moment", () => {
  const cut = truncateTrajectory(TIMED, 0.5);
  expect(cut.points).toEqual([
    { x: 0, y: 0, z: 0 },
    { x: 5, y: 5, z: 50 },
  ]);
  expect(cut.position).toEqual({ x: 5, y: 5, z: 50 });
});

it("builds a scrubbed scene on full-run axes with one dot per vehicle", () => {
  const scene = sceneOfTracksUpTo([{ name: "vehicle", trajectory: TIMED }], 1);
  expect(scene).not.toBeNull();
  expect(scene!.series[0].points).toHaveLength(2);
  expect(scene!.axes).toEqual(sceneOfTracks([{ name: "vehicle", trajectory: TIMED }])!.axes);
  expect(scene!.markers.filter((m) => m.kind === "position")).toHaveLength(1);
  expect(scene!.markers.some((m) => m.kind === "end")).toBe(false);
  expect(scene!.markers.find((m) => m.kind === "position")).toMatchObject({
    label: "vehicle",
    color: "#0f766e",
  });
});

it("scrubs two vehicles that share a time domain with colored position dots", () => {
  const other = {
    ...TIMED,
    points: [
      { x: 0, y: 0, z: 0 },
      { x: 5, y: 5, z: 50 },
      { x: 10, y: 10, z: 100 },
    ],
  };
  const tracks = [
    { name: "Missile 1", trajectory: TIMED },
    { name: "Target 1", trajectory: other },
  ];
  const scene = sceneOfTracksUpTo(tracks, 1);
  expect(scene).not.toBeNull();
  const positions = scene!.markers.filter((m) => m.kind === "position");
  expect(positions).toHaveLength(2);
  expect(positions.map((m) => m.color)).toEqual(["#0f766e", "#2563eb"]);
  expect(scene!.axes).toEqual(sceneOfTracks(tracks)!.axes);
  expect(scene!.markers.some((m) => m.kind === "end")).toBe(false);
});

it("formats slider times without float residue", () => {
  expect(formatSliderTime(0)).toBe("0");
  expect(formatSliderTime(2)).toBe("2");
  expect(formatSliderTime(9)).toBe("9");
  expect(formatSliderTime(44.999999999999616)).toBe("45");
  expect(formatSliderTime(90.00000000000914)).toBe("90");
  expect(formatSliderTime(0.5)).toBe("0.5");
  expect(formatSliderTime(1.23456789)).toBe("1.234568");
});

it("resets the orbit only when the run key changes", () => {
  expect(shouldResetOrbit("a", "a")).toBe(false);
  expect(shouldResetOrbit("a", "b")).toBe(true);
});
