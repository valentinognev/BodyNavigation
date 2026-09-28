import { GIFEncoder, applyPalette, quantize } from "gifenc";
import { axisLabel, axisTicks, formatTick } from "./plot";

export type Vec3 = { x: number; y: number; z: number };

export type Trajectory = {
  kind: "geographic" | "local";
  xName: string;
  yName: string;
  zName: string;
  points: Vec3[];
};

export type ScreenPoint = { sx: number; sy: number };

export type Orbit = { azimuth: number; elevation: number; distance: number };

export type TrajectoryScene = {
  series: { name: string; color: string; points: Vec3[] }[];
  markers: { kind: "start" | "end"; label: string; point: Vec3 }[];
  axes: { name: string; min: number; max: number; ticks: number[] }[];
  legend: { label: string; color: string; kind: "line" | "start" | "end" }[];
};

const SERIES_COLOR = "#0f766e";
export const TRACK_COLORS = ["#0f766e", "#2563eb", "#d97706", "#7c3aed", "#db2777", "#0891b2"];
const START_COLOR = "#15803d";
const END_COLOR = "#b91c1c";

const STROKE = [15, 118, 110, 255] as const;
const DRAG_DEG_PER_PX = 0.5;
const FRAME_DELAY_MS = 80;

const ELEVATION = Math.PI / 6;

function hasColumn(columns: string[], name: string): boolean {
  return columns.includes(name);
}

function finitePoint(point: Vec3): boolean {
  return Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z);
}

export function trajectoryOf(
  columns: string[],
  rows: Record<string, number>[],
): Trajectory | null {
  const geographic = hasColumn(columns, "latx") && hasColumn(columns, "lonx") && hasColumn(columns, "alt");
  const missile =
    hasColumn(columns, "SBEL1") && hasColumn(columns, "SBEL2") && hasColumn(columns, "SBEL3");
  const aircraft =
    hasColumn(columns, "SAEL1") && hasColumn(columns, "SAEL2") && hasColumn(columns, "SAEL3");
  if (!geographic && !missile && !aircraft) return null;

  const points: Vec3[] = [];
  for (const row of rows) {
    const point = geographic
      ? { x: Number(row.lonx), y: Number(row.latx), z: Number(row.alt) }
      : missile
        ? { x: Number(row.SBEL2), y: Number(row.SBEL1), z: -Number(row.SBEL3) }
        : { x: Number(row.SAEL2), y: Number(row.SAEL1), z: -Number(row.SAEL3) };
    if (finitePoint(point)) points.push(point);
  }
  if (points.length === 0) return null;

  if (geographic) {
    return { kind: "geographic", xName: "lonx", yName: "latx", zName: "alt", points };
  }
  if (missile) {
    return { kind: "local", xName: "SBEL2", yName: "SBEL1", zName: "-SBEL3", points };
  }
  return { kind: "local", xName: "SAEL2", yName: "SAEL1", zName: "-SAEL3", points };
}

export function trajectoriesFromPlot(plot: {
  columns: string[];
  rows: Record<string, number>[];
  vehicles?: { name: string; columns: string[]; rows: Record<string, number>[] }[];
}): { name: string; trajectory: Trajectory }[] {
  const vehicles = plot.vehicles ?? [];
  if (vehicles.length > 0) {
    const tracks: { name: string; trajectory: Trajectory }[] = [];
    for (const vehicle of vehicles) {
      const trajectory = trajectoryOf(vehicle.columns, vehicle.rows);
      if (trajectory != null) tracks.push({ name: vehicle.name, trajectory });
    }
    return tracks;
  }
  const trajectory = trajectoryOf(plot.columns, plot.rows);
  return trajectory == null ? [] : [{ name: "vehicle", trajectory }];
}

export function sceneOf(trajectory: Trajectory, seriesName = "vehicle"): TrajectoryScene | null {
  return sceneOfTracks([{ name: seriesName, trajectory }]);
}

export function sceneOfTracks(
  tracks: { name: string; trajectory: Trajectory }[],
): TrajectoryScene | null {
  const present = tracks.filter((track) => track.trajectory.points.length > 0);
  if (present.length === 0) return null;
  const kind = present[0].trajectory.kind;
  const same = present.filter((track) => track.trajectory.kind === kind);
  const names = axisNames(same.map((track) => track.trajectory), kind);
  const raw = same.flatMap((track) => track.trajectory.points);
  const axes = [
    { name: names.xName, ...axisOf(raw.map((point) => point.x)) },
    { name: names.yName, ...axisOf(raw.map((point) => point.y)) },
    { name: names.zName, ...axisOf(raw.map((point) => point.z)) },
  ];
  const series = same.map((track, index) => ({
    name: track.name,
    color: TRACK_COLORS[index % TRACK_COLORS.length],
    points: track.trajectory.points.map((point) => ({
      x: tickFraction(point.x, axes[0].min, axes[0].max),
      y: tickFraction(point.y, axes[1].min, axes[1].max),
      z: tickFraction(point.z, axes[2].min, axes[2].max),
    })),
  }));
  const markers: TrajectoryScene["markers"] = [];
  for (const item of series) {
    markers.push({ kind: "start", label: "start", point: item.points[0] });
    if (item.points.length > 1) {
      markers.push({ kind: "end", label: "end", point: item.points[item.points.length - 1] });
    }
  }
  const legend: TrajectoryScene["legend"] = series.map((item) => ({
    label: item.name,
    color: item.color,
    kind: "line",
  }));
  legend.push({ label: "start", color: START_COLOR, kind: "start" });
  if (markers.some((marker) => marker.kind === "end")) {
    legend.push({ label: "end", color: END_COLOR, kind: "end" });
  }
  return { series, markers, axes, legend };
}

function axisNames(
  trajectories: Trajectory[],
  kind: Trajectory["kind"],
): { xName: string; yName: string; zName: string } {
  const first = trajectories[0];
  const shared =
    trajectories.every((trajectory) => trajectory.xName === first.xName) &&
    trajectories.every((trajectory) => trajectory.yName === first.yName) &&
    trajectories.every((trajectory) => trajectory.zName === first.zName);
  if (shared) return { xName: first.xName, yName: first.yName, zName: first.zName };
  if (kind === "local") return { xName: "east", yName: "north", zName: "up" };
  return { xName: first.xName, yName: first.yName, zName: first.zName };
}

function axisOf(values: number[]): { min: number; max: number; ticks: number[] } {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const ticks = axisTicks(min, max);
  return { min: ticks.min, max: ticks.max, ticks: ticks.ticks };
}

export function dataAxisLines(axes: TrajectoryScene["axes"]): { name: string; from: Vec3; to: Vec3 }[] {
  const origin = { x: 0, y: 0, z: 0 };
  return [
    { name: axes[0].name, from: origin, to: { x: 1, y: 0, z: 0 } },
    { name: axes[1].name, from: origin, to: { x: 0, y: 1, z: 0 } },
    { name: axes[2].name, from: origin, to: { x: 0, y: 0, z: 1 } },
  ];
}

export function tickFraction(value: number, min: number, max: number): number {
  return (value - min) / (max - min || 1);
}

function lerpVec3(from: Vec3, to: Vec3, t: number): Vec3 {
  return {
    x: from.x + (to.x - from.x) * t,
    y: from.y + (to.y - from.y) * t,
    z: from.z + (to.z - from.z) * t,
  };
}

const TICK_CROSS: [Vec3, Vec3][] = [
  [
    { x: 0, y: 1, z: 0 },
    { x: 0, y: 0, z: 1 },
  ],
  [
    { x: 1, y: 0, z: 0 },
    { x: 0, y: 0, z: 1 },
  ],
  [
    { x: 1, y: 0, z: 0 },
    { x: 0, y: 1, z: 0 },
  ],
];

export function axisTickMarks(
  axis: TrajectoryScene["axes"][number],
  from: Vec3,
  to: Vec3,
  axisIndex: 0 | 1 | 2,
  radius: number,
): { at: Vec3; label: string; crosses: { from: Vec3; to: Vec3 }[] }[] {
  const step = axis.ticks.length >= 2 ? axis.ticks[1] - axis.ticks[0] || 1 : 1;
  const half = (radius * 0.02) / 2;
  const [u, v] = TICK_CROSS[axisIndex];
  return axis.ticks.map((tick) => {
    const at = lerpVec3(from, to, tickFraction(tick, axis.min, axis.max));
    return {
      at,
      label: formatTick(tick, step),
      crosses: [
        {
          from: { x: at.x - u.x * half, y: at.y - u.y * half, z: at.z - u.z * half },
          to: { x: at.x + u.x * half, y: at.y + u.y * half, z: at.z + u.z * half },
        },
        {
          from: { x: at.x - v.x * half, y: at.y - v.y * half, z: at.z - v.z * half },
          to: { x: at.x + v.x * half, y: at.y + v.y * half, z: at.z + v.z * half },
        },
      ],
    };
  });
}

export function centeredCloud(points: Vec3[]): { center: Vec3; radius: number; cloud: Vec3[] } {
  if (points.length === 0) return { center: { x: 0, y: 0, z: 0 }, radius: 1, cloud: [] };
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (const point of points) {
    cx += point.x;
    cy += point.y;
    cz += point.z;
  }
  const count = points.length;
  const center = { x: cx / count, y: cy / count, z: cz / count };
  const cloud = points.map((point) => ({
    x: point.x - center.x,
    y: point.y - center.y,
    z: point.z - center.z,
  }));
  let radius = 0;
  for (const point of cloud) {
    const distance = Math.hypot(point.x, point.y, point.z);
    if (distance > radius) radius = distance;
  }
  return { center, radius: radius || 1, cloud };
}

export function markerSize(radius: number): number {
  return radius * 0.05;
}

export type LegendHudRow = {
  label: string;
  color: string;
  kind: "line" | "start" | "end";
  chip: { x: number; y: number; w: number; h: number };
  text: { x: number; y: number };
};

export function legendHudRows(
  legend: TrajectoryScene["legend"],
  _width = 360,
  _height = 280,
): LegendHudRow[] {
  return legend.map((item, index) => {
    const y = 8 + index * 22;
    return {
      label: item.label,
      color: item.color,
      kind: item.kind,
      chip: { x: 8, y, w: 12, h: 10 },
      text: { x: 24, y: y + 5 },
    };
  });
}

function boxCorners(): Vec3[] {
  return [
    { x: 0, y: 0, z: 0 },
    { x: 1, y: 0, z: 0 },
    { x: 0, y: 1, z: 0 },
    { x: 1, y: 1, z: 0 },
    { x: 0, y: 0, z: 1 },
    { x: 1, y: 0, z: 1 },
    { x: 0, y: 1, z: 1 },
    { x: 1, y: 1, z: 1 },
  ];
}

function lerpScreen(from: ScreenPoint, to: ScreenPoint, t: number): ScreenPoint {
  return { sx: from.sx + (to.sx - from.sx) * t, sy: from.sy + (to.sy - from.sy) * t };
}

function tickStep(axis: TrajectoryScene["axes"][number]): number {
  if (axis.ticks.length >= 2) return axis.ticks[1] - axis.ticks[0];
  const span = axis.max - axis.min;
  return span === 0 ? 1 : span;
}

function axisScreen(
  axis: TrajectoryScene["axes"][number],
  from: ScreenPoint,
  to: ScreenPoint,
): {
  name: string;
  label: string;
  from: ScreenPoint;
  to: ScreenPoint;
  ticks: { label: string; at: ScreenPoint }[];
} {
  const span = axis.max - axis.min;
  const step = tickStep(axis);
  return {
    name: axis.name,
    label: axisLabel(axis.name),
    from,
    to,
    ticks: axis.ticks.map((value) => ({
      label: formatTick(value, step),
      at: lerpScreen(from, to, span === 0 ? 0 : (value - axis.min) / span),
    })),
  };
}

function frameFromProjected(
  scene: TrajectoryScene,
  points: ScreenPoint[],
): {
  segments: { points: ScreenPoint[]; color: "line" | "start" | "end" }[];
  axes: {
    name: string;
    label: string;
    from: ScreenPoint;
    to: ScreenPoint;
    ticks: { label: string; at: ScreenPoint }[];
  }[];
  legend: TrajectoryScene["legend"];
} {
  const segments: { points: ScreenPoint[]; color: "line" | "start" | "end" }[] = [];
  let offset = 0;
  for (const series of scene.series) {
    segments.push({ points: points.slice(offset, offset + series.points.length), color: "line" });
    offset += series.points.length;
  }
  for (const marker of scene.markers) {
    segments.push({ points: [points[offset]], color: marker.kind });
    offset += 1;
  }
  const origin = points[offset];
  const xEnd = points[offset + 1];
  const yEnd = points[offset + 2];
  const zEnd = points[offset + 4];
  return {
    segments,
    axes: [
      axisScreen(scene.axes[0], origin, xEnd),
      axisScreen(scene.axes[1], origin, yEnd),
      axisScreen(scene.axes[2], origin, zEnd),
    ],
    legend: scene.legend,
  };
}

function combinedScenePoints(scene: TrajectoryScene): Vec3[] {
  const seriesPoints: Vec3[] = [];
  for (const series of scene.series) seriesPoints.push(...series.points);
  return [...seriesPoints, ...scene.markers.map((marker) => marker.point), ...boxCorners()];
}

export function sceneFrame(
  scene: TrajectoryScene,
  azimuthDeg: number,
  width: number,
  height: number,
  pad = 16,
): ReturnType<typeof frameFromProjected> {
  const [frame] = projectSceneFrames(scene, [azimuthDeg], width, height, pad);
  return frame ?? { segments: [], axes: [], legend: scene.legend };
}

export function projectSceneFrames(
  scene: TrajectoryScene,
  azimuths: number[],
  width: number,
  height: number,
  pad = 16,
): ReturnType<typeof frameFromProjected>[] {
  const projected = projectTurntable(combinedScenePoints(scene), azimuths, width, height, pad);
  return projected.map((points) => frameFromProjected(scene, points));
}

function projectRaw(point: Vec3, azimuthDeg: number): ScreenPoint {
  const azimuth = (azimuthDeg * Math.PI) / 180;
  const cos = Math.cos(azimuth);
  const sin = Math.sin(azimuth);
  const xr = point.x * cos - point.y * sin;
  const yr = point.x * sin + point.y * cos;
  return {
    sx: xr,
    sy: -(yr * Math.sin(ELEVATION) + point.z * Math.cos(ELEVATION)),
  };
}

export function projectTurntable(
  points: Vec3[],
  azimuths: number[],
  width: number,
  height: number,
  pad = 16,
): ScreenPoint[][] {
  if (points.length === 0 || azimuths.length === 0) return [];
  const frames = azimuths.map((azimuth) => points.map((point) => projectRaw(point, azimuth)));
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const frame of frames) {
    for (const point of frame) {
      if (point.sx < minX) minX = point.sx;
      if (point.sx > maxX) maxX = point.sx;
      if (point.sy < minY) minY = point.sy;
      if (point.sy > maxY) maxY = point.sy;
    }
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const innerW = Math.max(1, width - 2 * pad);
  const innerH = Math.max(1, height - 2 * pad);
  const scale = Math.min(innerW / spanX, innerH / spanY);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const midX = width / 2;
  const midY = height / 2;
  return frames.map((frame) =>
    frame.map((point) => ({
      sx: midX + (point.sx - cx) * scale,
      sy: midY + (point.sy - cy) * scale,
    })),
  );
}

function plotPixel(
  data: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  color: readonly [number, number, number, number] = STROKE,
): void {
  if (x < 0 || y < 0 || x >= width || y >= height) return;
  const index = (y * width + x) * 4;
  data[index] = color[0];
  data[index + 1] = color[1];
  data[index + 2] = color[2];
  data[index + 3] = color[3];
}

function strokeSegment(
  data: Uint8Array,
  width: number,
  height: number,
  from: ScreenPoint,
  to: ScreenPoint,
  color: readonly [number, number, number, number] = STROKE,
): void {
  let x0 = Math.round(from.sx);
  let y0 = Math.round(from.sy);
  const x1 = Math.round(to.sx);
  const y1 = Math.round(to.sy);
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const stepX = x0 < x1 ? 1 : -1;
  const stepY = y0 < y1 ? 1 : -1;
  let err = dx - dy;
  for (;;) {
    plotPixel(data, width, height, x0, y0, color);
    if (x0 === x1 && y0 === y1) break;
    const doubled = 2 * err;
    if (doubled > -dy) {
      err -= dy;
      x0 += stepX;
    }
    if (doubled < dx) {
      err += dx;
      y0 += stepY;
    }
  }
}

export function rasterizeSegments(
  segments: ScreenPoint[][],
  width: number,
  height: number,
): Uint8Array {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 255;
    data[i + 1] = 255;
    data[i + 2] = 255;
    data[i + 3] = 255;
  }
  for (const segment of segments) {
    if (segment.length === 1) {
      plotPixel(data, width, height, Math.round(segment[0].sx), Math.round(segment[0].sy));
    }
    for (let i = 1; i < segment.length; i++) {
      strokeSegment(data, width, height, segment[i - 1], segment[i]);
    }
  }
  return data;
}

export function encodeTurntableGif(
  points: Vec3[],
  azimuths: number[],
  width: number,
  height: number,
): Uint8Array {
  const frames = projectTurntable(points, azimuths, width, height);
  const gif = GIFEncoder();
  for (const frame of frames) {
    const rgba = rasterizeSegments([frame], width, height);
    const palette = quantize(rgba, 16);
    const index = applyPalette(rgba, palette);
    gif.writeFrame(index, width, height, { palette, delay: FRAME_DELAY_MS });
  }
  gif.finish();
  return gif.bytes();
}

const AXIS_RGBA = [100, 116, 139, 255] as const;
const START_RGBA = [21, 128, 61, 255] as const;
const END_RGBA = [185, 28, 28, 255] as const;

function rgbaOf(hex: string): readonly [number, number, number, number] {
  const n = Number.parseInt(hex.slice(1), 16);
  if (!Number.isFinite(n)) return STROKE;
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255, 255];
}

function fillDisc(
  data: Uint8Array,
  width: number,
  height: number,
  cx: number,
  cy: number,
  radius: number,
  color: readonly [number, number, number, number],
): void {
  const x0 = Math.round(cx);
  const y0 = Math.round(cy);
  const r2 = radius * radius;
  for (let y = y0 - radius; y <= y0 + radius; y++) {
    for (let x = x0 - radius; x <= x0 + radius; x++) {
      if ((x - x0) * (x - x0) + (y - y0) * (y - y0) <= r2) {
        plotPixel(data, width, height, x, y, color);
      }
    }
  }
}

function fillRect(
  data: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  w: number,
  h: number,
  color: readonly [number, number, number, number],
): void {
  const x0 = Math.round(x);
  const y0 = Math.round(y);
  for (let yy = 0; yy < h; yy++) {
    for (let xx = 0; xx < w; xx++) {
      plotPixel(data, width, height, x0 + xx, y0 + yy, color);
    }
  }
}

function strokePolyline(
  data: Uint8Array,
  width: number,
  height: number,
  points: ScreenPoint[],
  color: readonly [number, number, number, number],
): void {
  if (points.length === 1) {
    plotPixel(data, width, height, Math.round(points[0].sx), Math.round(points[0].sy), color);
  }
  for (let i = 1; i < points.length; i++) {
    strokeSegment(data, width, height, points[i - 1], points[i], color);
  }
}

export function rasterizeScene(
  frame: ReturnType<typeof sceneFrame>,
  scene: TrajectoryScene,
  width: number,
  height: number,
): Uint8Array {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 255;
    data[i + 1] = 255;
    data[i + 2] = 255;
    data[i + 3] = 255;
  }
  let lineIndex = 0;
  for (const segment of frame.segments) {
    if (segment.color === "line") {
      const color = rgbaOf(scene.series[lineIndex]?.color ?? SERIES_COLOR);
      strokePolyline(data, width, height, segment.points, color);
      lineIndex += 1;
    }
  }
  for (const axis of frame.axes) {
    strokeSegment(data, width, height, axis.from, axis.to, AXIS_RGBA);
    for (const tick of axis.ticks) {
      strokeSegment(
        data,
        width,
        height,
        { sx: tick.at.sx - 1.5, sy: tick.at.sy },
        { sx: tick.at.sx + 1.5, sy: tick.at.sy },
        AXIS_RGBA,
      );
      strokeSegment(
        data,
        width,
        height,
        { sx: tick.at.sx, sy: tick.at.sy - 1.5 },
        { sx: tick.at.sx, sy: tick.at.sy + 1.5 },
        AXIS_RGBA,
      );
    }
  }
  for (const segment of frame.segments) {
    if (segment.color === "start" && segment.points[0]) {
      fillDisc(data, width, height, segment.points[0].sx, segment.points[0].sy, 3, START_RGBA);
    }
    if (segment.color === "end" && segment.points[0]) {
      const sx = Math.round(segment.points[0].sx);
      const sy = Math.round(segment.points[0].sy);
      fillRect(data, width, height, sx - 2, sy - 2, 5, 5, END_RGBA);
    }
  }
  let legendY = 20;
  for (const item of frame.legend) {
    fillRect(data, width, height, 4, legendY, 10, 10, rgbaOf(item.color));
    legendY += 12;
  }
  return data;
}

export function renderSceneGif(
  scene: TrajectoryScene,
  azimuths: number[],
  width: number,
  height: number,
): Uint8Array {
  const gif = GIFEncoder();
  for (const frame of projectSceneFrames(scene, azimuths, width, height)) {
    const rgba = rasterizeScene(frame, scene, width, height);
    const palette = quantize(rgba, 16);
    const index = applyPalette(rgba, palette);
    gif.writeFrame(index, width, height, { palette, delay: FRAME_DELAY_MS });
  }
  gif.finish();
  return gif.bytes();
}

export function turntableAzimuths(steps: number): number[] {
  const count = Math.max(1, Math.floor(steps));
  const azimuths: number[] = [];
  for (let i = 0; i < count; i++) azimuths.push((i * 360) / count);
  return azimuths;
}

export function orbitCamera(azimuthDeg: number, elevationDeg: number, distance: number): Vec3 {
  const azimuth = (azimuthDeg * Math.PI) / 180;
  const elevation = (elevationDeg * Math.PI) / 180;
  const horizontal = distance * Math.cos(elevation);
  return {
    x: horizontal * Math.cos(azimuth),
    y: horizontal * Math.sin(azimuth),
    z: distance * Math.sin(elevation),
  };
}

export function dragOrbit(orbit: Orbit, dx: number, dy: number): Orbit {
  let elevation = orbit.elevation - dy * DRAG_DEG_PER_PX;
  if (elevation > 89) elevation = 89;
  if (elevation < -89) elevation = -89;
  return {
    azimuth: orbit.azimuth - dx * DRAG_DEG_PER_PX,
    elevation,
    distance: orbit.distance,
  };
}

export function wheelOrbit(orbit: Orbit, deltaY: number): Orbit {
  const factor = deltaY > 0 ? 1.1 : 0.9;
  let distance = orbit.distance * factor;
  if (distance < 0.5) distance = 0.5;
  if (distance > 1e7) distance = 1e7;
  return { ...orbit, distance };
}

export function startFramePump(
  draw: () => void,
  requestFrame: (() => void) | undefined,
  schedule: (tick: () => void) => number = (tick) => requestAnimationFrame(tick),
  cancel: (id: number) => void = (id) => cancelAnimationFrame(id),
): () => void {
  let stopped = false;
  let frameId = 0;
  const tick = () => {
    if (stopped) return;
    draw();
    requestFrame?.();
    frameId = schedule(tick);
  };
  frameId = schedule(tick);
  return () => {
    stopped = true;
    cancel(frameId);
  };
}

export function shouldStopRecorder(state: string): boolean {
  return state !== "inactive";
}

export function opaqueClearColor(cssBackground: string | null | undefined): string {
  if (cssBackground == null) return "#ffffff";
  const value = cssBackground.trim();
  if (value === "" || value === "transparent") return "#ffffff";
  const rgba = /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)$/i.exec(value);
  if (rgba != null && rgba[4] != null && Number(rgba[4]) === 0) return "#ffffff";
  return value;
}

function cssRgb(color: string): [number, number, number] | null {
  const hex = /^#([\da-f]{6})$/i.exec(color.trim());
  if (hex != null) {
    const packed = Number.parseInt(hex[1], 16);
    return [(packed >> 16) & 255, (packed >> 8) & 255, packed & 255];
  }
  const rgb = /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/i.exec(color.trim());
  if (rgb == null) return null;
  return [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])];
}

function relativeLuminance(rgb: [number, number, number]): number {
  const linear = rgb.map((channel) => {
    const s = channel / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}

/** Distance from an anchor to a label center so an axis-aligned box clears the anchor. */
export function labelOffsetDistance(
  dirX: number,
  dirY: number,
  boxWidth: number,
  boxHeight: number,
  gap: number,
): number {
  const len = Math.hypot(dirX, dirY) || 1;
  return (Math.abs(dirX) / len) * (boxWidth / 2) + (Math.abs(dirY) / len) * (boxHeight / 2) + gap;
}

/** Move `anchor` toward `tip` by `distance`. */
export function labelClearance(
  anchor: { x: number; y: number },
  tip: { x: number; y: number },
  distance: number,
): { x: number; y: number } {
  const dx = tip.x - anchor.x;
  const dy = tip.y - anchor.y;
  const len = Math.hypot(dx, dy) || 1;
  return { x: anchor.x + (dx / len) * distance, y: anchor.y + (dy / len) * distance };
}

/** Fill and halo for axis, tick, and legend sprites. Dark canvases get light glyphs. */
export function labelInk(cssBackground: string | null | undefined): { fill: string; stroke: string } {
  const rgb = cssRgb(opaqueClearColor(cssBackground));
  if (rgb != null && relativeLuminance(rgb) < 0.4) return { fill: "#f8fafc", stroke: "#020617" };
  return { fill: "#1e293b", stroke: "#ffffff" };
}

export function downloadBytes(bytes: Uint8Array, filename: string, mime: string): void {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const url = URL.createObjectURL(new Blob([copy], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
