import { axisLabel, axisTicks, formatTick } from "./plot";

export type Vec3 = { x: number; y: number; z: number };

export type Trajectory = {
  kind: "geographic" | "local";
  xName: string;
  yName: string;
  zName: string;
  points: Vec3[];
  times: number[];
};

export type ScreenPoint = { sx: number; sy: number };

export type Orbit = { azimuth: number; elevation: number; distance: number };

export type TrajectoryMarker = {
  kind: "start" | "end" | "position";
  label: string;
  point: Vec3;
  color?: string;
};

export type TrajectoryScene = {
  series: { name: string; color: string; points: Vec3[] }[];
  markers: TrajectoryMarker[];
  axes: { name: string; min: number; max: number; ticks: number[] }[];
  legend: { label: string; color: string; kind: "line" | "start" | "end" }[];
};

export const TRACK_COLORS = ["#0f766e", "#2563eb", "#d97706", "#7c3aed", "#db2777", "#0891b2"];
const START_COLOR = "#15803d";
const END_COLOR = "#b91c1c";

const DRAG_DEG_PER_PX = 0.5;

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
  const times: number[] = [];
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const point = geographic
      ? { x: Number(row.lonx), y: Number(row.latx), z: Number(row.alt) }
      : missile
        ? { x: Number(row.SBEL2), y: Number(row.SBEL1), z: -Number(row.SBEL3) }
        : { x: Number(row.SAEL2), y: Number(row.SAEL1), z: -Number(row.SAEL3) };
    if (finitePoint(point)) {
      points.push(point);
      const t = Number(row.time);
      times.push(Number.isFinite(t) ? t : index);
    }
  }
  if (points.length === 0) return null;

  if (geographic) {
    return { kind: "geographic", xName: "lonx", yName: "latx", zName: "alt", points, times };
  }
  if (missile) {
    return { kind: "local", xName: "SBEL2", yName: "SBEL1", zName: "-SBEL3", points, times };
  }
  return { kind: "local", xName: "SAEL2", yName: "SAEL1", zName: "-SAEL3", points, times };
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

function axesOf(
  rawPoints: Vec3[],
  kind: Trajectory["kind"],
  trajectories: Trajectory[],
): TrajectoryScene["axes"] {
  const names = axisNames(trajectories, kind);
  return [
    { name: names.xName, ...axisOf(rawPoints.map((point) => point.x)) },
    { name: names.yName, ...axisOf(rawPoints.map((point) => point.y)) },
    { name: names.zName, ...axisOf(rawPoints.map((point) => point.z)) },
  ];
}

export function sceneOfTracks(
  tracks: { name: string; trajectory: Trajectory }[],
): TrajectoryScene | null {
  const present = tracks.filter((track) => track.trajectory.points.length > 0);
  if (present.length === 0) return null;
  const kind = present[0].trajectory.kind;
  const same = present.filter((track) => track.trajectory.kind === kind);
  const trajectories = same.map((track) => track.trajectory);
  const raw = same.flatMap((track) => track.trajectory.points);
  const axes = axesOf(raw, kind, trajectories);
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

export function sliderTimes(tracks: { name: string; trajectory: Trajectory }[]): number[] {
  const seen = new Set<number>();
  for (const track of tracks) {
    for (const t of track.trajectory.times) {
      if (Number.isFinite(t)) seen.add(t);
    }
  }
  return [...seen].sort((a, b) => a - b);
}

export function trackPositionAt(trajectory: Trajectory, t: number): Vec3 | null {
  const { points, times } = trajectory;
  if (points.length === 0) return null;
  if (t <= times[0]) return points[0];
  for (let i = 1; i < points.length; i += 1) {
    if (t <= times[i]) {
      const span = times[i] - times[i - 1] || 1;
      const f = (t - times[i - 1]) / span;
      const a = points[i - 1];
      const b = points[i];
      return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, z: a.z + (b.z - a.z) * f };
    }
  }
  return points[points.length - 1];
}

export function truncateTrajectory(
  trajectory: Trajectory,
  t: number,
): { points: Vec3[]; position: Vec3 | null } {
  const position = trackPositionAt(trajectory, t);
  if (position == null) return { points: [], position: null };
  const prefix = trajectory.points.filter((_, i) => trajectory.times[i] <= t);
  const last = prefix[prefix.length - 1];
  if (last == null || last.x !== position.x || last.y !== position.y || last.z !== position.z) {
    prefix.push(position);
  }
  return { points: prefix, position };
}

export function sceneOfTracksUpTo(
  tracks: { name: string; trajectory: Trajectory }[],
  t: number,
): TrajectoryScene | null {
  const full = sceneOfTracks(tracks);
  if (full == null) return null;
  const byName = new Map(tracks.map((track) => [track.name, track.trajectory]));
  const series: TrajectoryScene["series"] = [];
  const markers: TrajectoryScene["markers"] = [];
  for (const item of full.series) {
    const trajectory = byName.get(item.name);
    if (trajectory == null) continue;
    const truncated = truncateTrajectory(trajectory, t);
    if (truncated.points.length === 0) continue;
    const points = truncated.points.map((point) => ({
      x: tickFraction(point.x, full.axes[0].min, full.axes[0].max),
      y: tickFraction(point.y, full.axes[1].min, full.axes[1].max),
      z: tickFraction(point.z, full.axes[2].min, full.axes[2].max),
    }));
    series.push({ name: item.name, color: item.color, points });
    markers.push({ kind: "start", label: "start", point: points[0] });
    if (truncated.position != null) {
      markers.push({
        kind: "position",
        label: item.name,
        color: item.color,
        point: {
          x: tickFraction(truncated.position.x, full.axes[0].min, full.axes[0].max),
          y: tickFraction(truncated.position.y, full.axes[1].min, full.axes[1].max),
          z: tickFraction(truncated.position.z, full.axes[2].min, full.axes[2].max),
        },
      });
    }
  }
  return { series, markers, axes: full.axes, legend: full.legend };
}

export function shouldResetOrbit(prevKey: string, nextKey: string): boolean {
  return prevKey !== nextKey;
}

/** Slider caption: integers within float noise as ints; else up to 6 decimals, trailing zeros stripped. */
export function formatSliderTime(t: number): string {
  if (!Number.isFinite(t)) return String(t);
  const nearest = Math.round(t);
  if (Math.abs(t - nearest) < 1e-9) return String(nearest);
  return String(Number(t.toFixed(6)));
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

/** Camera frame for the orbit view. The axis box is the full run, so a short scrub does not pull the far plane inside the camera. */
export function viewFrame(scene: TrajectoryScene): { center: Vec3; radius: number } {
  if (scene.axes.length < 3) return centeredCloud([]);
  const { center, radius } = centeredCloud(boxCorners());
  return { center, radius };
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
  segments: { points: ScreenPoint[]; color: "line" | "start" | "end" | "position" }[];
  axes: {
    name: string;
    label: string;
    from: ScreenPoint;
    to: ScreenPoint;
    ticks: { label: string; at: ScreenPoint }[];
  }[];
  legend: TrajectoryScene["legend"];
} {
  const segments: { points: ScreenPoint[]; color: "line" | "start" | "end" | "position" }[] = [];
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
