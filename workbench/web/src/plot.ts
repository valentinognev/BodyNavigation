const PREFERRED = ["alt", "mach", "dvbe", "alphax"] as const;

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
