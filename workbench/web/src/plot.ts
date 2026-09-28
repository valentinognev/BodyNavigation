const PREFERRED = ["alt", "mach", "dvbe", "alphax"] as const;

/** CADAC definition units (SI). Vector components inherit the stem. */
const UNITS: Record<string, string> = {
  time: "s",
  FSPV: "m/s²",
  pdynmc: "Pa",
  mach: "ND",
  lonx: "deg",
  latx: "deg",
  alt: "m",
  dvbe: "m/s",
  psivgx: "deg",
  thtvgx: "deg",
  SBEG: "m",
  SBEL: "m",
  SAEL: "m",
  east: "m",
  north: "m",
  up: "m",
  VBEG: "m/s",
  throttle: "ND",
  mass: "kg",
  thrust: "N",
  fmassr: "kg",
  cl_ov_cd: "ND",
  alphax: "deg",
};

const VECTOR_STEMS = ["FSPV", "SBEG", "SBEL", "SAEL", "VBEG"] as const;

const TICK_TARGET = 4;

export type TickMark = {
  value: number;
  label: string;
  x: number;
  y: number;
};

export type GridLine = { x1: number; y1: number; x2: number; y2: number };

export type ChartGeometry = {
  polyline: string;
  xLabel: string;
  yLabel: string;
  xTicks: TickMark[];
  yTicks: TickMark[];
  grid: GridLine[];
  frame: { x: number; y: number; w: number; h: number };
  xTitle: { x: number; y: number };
  yTitle: { x: number; y: number };
};

export function axisLabel(name: string): string {
  const unit = unitOf(name);
  return unit == null ? name : `${name} (${unit})`;
}

function unitOf(name: string): string | undefined {
  const direct = UNITS[name];
  if (direct != null) return direct;
  const stemName = name.startsWith("-") ? name.slice(1) : name;
  for (const stem of VECTOR_STEMS) {
    if (stemName.startsWith(stem) && /^[123]$/.test(stemName.slice(stem.length))) return UNITS[stem];
  }
  return undefined;
}

function niceStep(raw: number): number {
  if (!(raw > 0) || !Number.isFinite(raw)) return 1;
  const exp = Math.floor(Math.log10(raw));
  const frac = raw / 10 ** exp;
  const nice = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10;
  return nice * 10 ** exp;
}

function snap(value: number, step: number): number {
  if (!Number.isFinite(value)) return value;
  if (!(step > 0) || !Number.isFinite(step)) return value;
  const decimals = Math.min(8, Math.max(0, -Math.floor(Math.log10(step)) + 1));
  return Number(value.toFixed(decimals));
}

export function formatTick(value: number, step = 1): string {
  if (!Number.isFinite(value)) return "";
  const absStep = Math.abs(step);
  if (!(absStep > 0) || !Number.isFinite(absStep)) return String(value);
  if (absStep >= 1e6 || (Math.abs(value) >= 1e6 && absStep >= 1)) {
    return value.toExponential(2).replace(/\.0+e/, "e").replace("e+", "e");
  }
  const decimals = Math.min(8, Math.max(0, Math.ceil(-Math.log10(absStep) - 1e-12)));
  return value.toFixed(decimals);
}

export function axisTicks(min: number, max: number): { min: number; max: number; step: number; ticks: number[] } {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1, step: 1, ticks: [0, 1] };
  }
  if (min > max) {
    const swap = min;
    min = max;
    max = swap;
  }
  if (min === max) {
    const delta = min === 0 ? 1 : Math.abs(min) * 0.05;
    min -= delta;
    max += delta;
  }
  const step = niceStep((max - min) / TICK_TARGET);
  let axisMin = Math.floor(min / step) * step;
  let axisMax = Math.ceil(max / step) * step;
  if (!(axisMax > axisMin)) axisMax = axisMin + step;
  const count = Math.round((axisMax - axisMin) / step);
  const ticks: number[] = [];
  for (let i = 0; i <= count && i < 12; i++) ticks.push(snap(axisMin + i * step, step));
  return { min: snap(axisMin, step), max: snap(axisMax, step), step, ticks };
}

function finitePairs(xs: number[], ys: number[]): { x: number; y: number }[] {
  const n = Math.min(xs.length, ys.length);
  const pairs: { x: number; y: number }[] = [];
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    const y = ys[i];
    if (Number.isFinite(x) && Number.isFinite(y)) pairs.push({ x, y });
  }
  return pairs;
}

export function chartGeometry(
  xs: number[],
  ys: number[],
  xName: string,
  yName: string,
  width: number,
  height: number,
): ChartGeometry | null {
  const pairs = finitePairs(xs, ys);
  if (pairs.length === 0) return null;
  let xmin = Infinity;
  let xmax = -Infinity;
  let ymin = Infinity;
  let ymax = -Infinity;
  for (const pair of pairs) {
    if (pair.x < xmin) xmin = pair.x;
    if (pair.x > xmax) xmax = pair.x;
    if (pair.y < ymin) ymin = pair.y;
    if (pair.y > ymax) ymax = pair.y;
  }
  const xAxis = axisTicks(xmin, xmax);
  const yAxis = axisTicks(ymin, ymax);
  const yLabelWidth = Math.max(
    ...yAxis.ticks.map((value) => 8 + formatTick(value, yAxis.step).length * 6),
  );
  const left = Math.min(130, Math.max(52, yLabelWidth + 6));
  const right = 14;
  const top = 20;
  const bottom = 38;
  const frame = {
    x: left,
    y: top,
    w: Math.max(1, width - left - right),
    h: Math.max(1, height - top - bottom),
  };
  const xSpan = xAxis.max - xAxis.min || 1;
  const ySpan = yAxis.max - yAxis.min || 1;
  const xPix = (value: number) => frame.x + ((value - xAxis.min) / xSpan) * frame.w;
  const yPix = (value: number) => frame.y + frame.h - ((value - yAxis.min) / ySpan) * frame.h;
  const bottomY = frame.y + frame.h;
  const rightX = frame.x + frame.w;
  const onEdge = (value: number, edge: number) => Math.abs(value - edge) < 0.5;
  const xTicks = xAxis.ticks.map((value) => ({
    value,
    label: formatTick(value, xAxis.step),
    x: xPix(value),
    y: bottomY,
  }));
  const yTicks = yAxis.ticks.map((value) => ({
    value,
    label: formatTick(value, yAxis.step),
    x: frame.x,
    y: yPix(value),
  }));
  const grid: GridLine[] = [
    ...xTicks
      .filter((tick) => !onEdge(tick.x, frame.x) && !onEdge(tick.x, rightX))
      .map((tick) => ({ x1: tick.x, y1: frame.y, x2: tick.x, y2: bottomY })),
    ...yTicks
      .filter((tick) => !onEdge(tick.y, frame.y) && !onEdge(tick.y, bottomY))
      .map((tick) => ({ x1: frame.x, y1: tick.y, x2: rightX, y2: tick.y })),
  ];
  return {
    polyline: pairs.map((pair) => `${xPix(pair.x)},${yPix(pair.y)}`).join(" "),
    xLabel: axisLabel(xName),
    yLabel: axisLabel(yName),
    xTicks,
    yTicks,
    grid,
    frame,
    xTitle: { x: frame.x + frame.w / 2, y: height - 6 },
    yTitle: { x: frame.x, y: 12 },
  };
}

export function overlayChart(
  series: { name: string; xs: number[]; ys: number[] }[],
  xName: string,
  yName: string,
  width: number,
  height: number,
): { geometry: ChartGeometry; series: { name: string; polyline: string }[] } | null {
  const groups = series
    .map((item) => ({ name: item.name, pairs: finitePairs(item.xs, item.ys) }))
    .filter((item) => item.pairs.length > 0);
  if (groups.length === 0) return null;
  const xs = groups.flatMap((item) => item.pairs.map((pair) => pair.x));
  const ys = groups.flatMap((item) => item.pairs.map((pair) => pair.y));
  const geometry = chartGeometry(xs, ys, xName, yName, width, height);
  if (geometry == null) return null;
  const xmin = geometry.xTicks[0].value;
  const xmax = geometry.xTicks[geometry.xTicks.length - 1].value;
  const ymin = geometry.yTicks[0].value;
  const ymax = geometry.yTicks[geometry.yTicks.length - 1].value;
  const xSpan = xmax - xmin || 1;
  const ySpan = ymax - ymin || 1;
  const xPix = (value: number) => geometry.frame.x + ((value - xmin) / xSpan) * geometry.frame.w;
  const yPix = (value: number) =>
    geometry.frame.y + geometry.frame.h - ((value - ymin) / ySpan) * geometry.frame.h;
  return {
    geometry,
    series: groups.map((item) => ({
      name: item.name,
      polyline: item.pairs.map((pair) => `${xPix(pair.x)},${yPix(pair.y)}`).join(" "),
    })),
  };
}

export function defaultColumns(columns: string[]): string[] {
  const set = new Set(columns);
  const preferred = PREFERRED.filter((name) => set.has(name));
  if (preferred.length > 0) return preferred;
  return columns.filter((name) => name !== "time").slice(0, 4);
}

export function hasGroundTrack(columns: string[]): boolean {
  return columns.includes("latx") && columns.includes("lonx");
}

export function polylinePoints(
  xs: number[],
  ys: number[],
  width: number,
  height: number,
  pad = 8,
): string {
  const n = Math.min(xs.length, ys.length);
  if (n === 0) return "";
  let xmin = Infinity;
  let xmax = -Infinity;
  let ymin = Infinity;
  let ymax = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    const y = ys[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  if (!Number.isFinite(xmin)) return "";
  const dx = xmax - xmin || 1;
  const dy = ymax - ymin || 1;
  const innerW = width - 2 * pad;
  const innerH = height - 2 * pad;
  const pts: string[] = [];
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    const y = ys[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const px = pad + ((x - xmin) / dx) * innerW;
    const py = pad + (1 - (y - ymin) / dy) * innerH;
    pts.push(`${px},${py}`);
  }
  return pts.join(" ");
}
