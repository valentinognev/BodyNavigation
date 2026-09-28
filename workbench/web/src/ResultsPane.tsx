import { Fragment, useMemo } from "react";
import { useStore } from "zustand";
import { FieldLabel } from "./forms/fields";
import { hasGroundTrack, overlayChart, type ChartGeometry } from "./plot";
import {
  downloadBytes,
  renderSceneGif,
  sceneFrame,
  sceneOfTracks,
  TRACK_COLORS,
  trajectoriesFromPlot,
  turntableAzimuths,
  type TrajectoryScene,
} from "./plot3d";
import type { VehiclePlot } from "./store";
import { groupCurves, mergedColumnModules } from "./plotGroups";
import { groundTrackTitle, plotDescription, seriesTitle, trajectoryTitle } from "./plotTitles";
import store from "./store";
import { TrajectoryView } from "./TrajectoryView";

const SERIES_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"];
const PLOT_W = 360;
const PLOT_H = 200;

const CAPTION_CLASS = "mb-1 text-xs font-medium text-slate-500 dark:text-slate-400";
const BUTTON_CLASS =
  "mb-3 w-auto rounded bg-slate-800 px-3 py-1 text-sm text-white dark:bg-slate-100 dark:text-slate-900";
const STILL_AZIMUTH = 45;

function numbers(rows: Record<string, number>[], column: string): number[] {
  return rows.map((row) => Number(row[column]));
}

function columnHint(program: string | null | undefined, name: string): string | undefined {
  return plotDescription(program, name);
}

function TrajectoryStill({
  program,
  kind,
  scene,
}: {
  program: string | null;
  kind: "geographic" | "local";
  scene: TrajectoryScene;
}) {
  const title = trajectoryTitle(program, kind);
  const frame = sceneFrame(scene, STILL_AZIMUTH, PLOT_W, PLOT_H, 28);
  return (
    <>
      <p className={CAPTION_CLASS}>{title}</p>
      <svg
        className="mb-2 w-full rounded border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900"
        viewBox={`0 0 ${PLOT_W} ${PLOT_H}`}
        role="img"
        aria-label={title}
      >
        {frame.axes.map((axis) => (
          <g key={axis.name}>
            <line
              x1={axis.from.sx}
              y1={axis.from.sy}
              x2={axis.to.sx}
              y2={axis.to.sy}
              className="stroke-slate-500 dark:stroke-slate-400"
              stroke="#64748b"
              strokeWidth="1"
            />
            {axis.ticks.map((tick, index) => (
              <g key={`${axis.name}-${index}`}>
                <line
                  x1={tick.at.sx - 1.5}
                  y1={tick.at.sy}
                  x2={tick.at.sx + 1.5}
                  y2={tick.at.sy}
                  stroke="#64748b"
                  strokeWidth="1"
                />
                <line
                  x1={tick.at.sx}
                  y1={tick.at.sy - 1.5}
                  x2={tick.at.sx}
                  y2={tick.at.sy + 1.5}
                  stroke="#64748b"
                  strokeWidth="1"
                />
                <text
                  className="fill-slate-600 font-sans dark:fill-slate-300"
                  fontSize="9"
                  x={tick.at.sx + 4}
                  y={tick.at.sy - 4}
                >
                  {tick.label}
                </text>
              </g>
            ))}
            <text
              className="fill-slate-700 font-sans dark:fill-slate-200"
              fontSize="11"
              x={axis.to.sx + 4}
              y={axis.to.sy + 4}
            >
              {axis.label}
            </text>
          </g>
        ))}
        {frame.segments
          .filter((segment) => segment.color === "line")
          .map((segment, index) => (
            <polyline
              key={scene.series[index]?.name ?? index}
              fill="none"
              stroke={scene.series[index]?.color ?? "#0f766e"}
              strokeWidth="1.5"
              points={segment.points.map((point) => `${point.sx},${point.sy}`).join(" ")}
            />
          ))}
        {frame.segments
          .filter((segment) => segment.color === "start" && segment.points[0] != null)
          .map((segment, index) => (
            <circle
              key={`start-${index}`}
              cx={segment.points[0].sx}
              cy={segment.points[0].sy}
              r={2.5}
              fill="#15803d"
            />
          ))}
        {frame.segments
          .filter((segment) => segment.color === "end" && segment.points[0] != null)
          .map((segment, index) => (
            <rect
              key={`end-${index}`}
              x={segment.points[0].sx - 2.5}
              y={segment.points[0].sy - 2.5}
              width={5}
              height={5}
              fill="#b91c1c"
            />
          ))}
        <g>
          {frame.legend.map((item, index) => {
            const y = 12 + index * 14;
            return (
              <g key={item.label}>
                {item.kind === "line" ? (
                  <line x1={8} y1={y} x2={20} y2={y} stroke={item.color} strokeWidth={2} />
                ) : (
                  <rect x={8} y={y - 5} width={10} height={10} fill={item.color} />
                )}
                <text
                  className="fill-slate-700 font-sans dark:fill-slate-200"
                  fontSize="10"
                  x={24}
                  y={y + 3}
                >
                  {item.label}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
      <button
        type="button"
        className={BUTTON_CLASS}
        onClick={() => {
          const bytes = renderSceneGif(scene, turntableAzimuths(24), PLOT_W, PLOT_H);
          downloadBytes(bytes, "trajectory.gif", "image/gif");
        }}
      >
        Turntable GIF
      </button>
    </>
  );
}

type ChartLine = { name: string; color: string; points: string };

function chartLines(
  sources: VehiclePlot[],
  xColumn: string,
  yColumn: string,
  multi: boolean,
  fallbackColor: string,
): { geometry: ChartGeometry; lines: ChartLine[] } | null {
  const present = sources.flatMap((vehicle, index) => {
    if (!vehicle.columns.includes(xColumn) || !vehicle.columns.includes(yColumn)) return [];
    return [
      {
        name: vehicle.name,
        color: multi ? TRACK_COLORS[index % TRACK_COLORS.length] : fallbackColor,
        xs: numbers(vehicle.rows, xColumn),
        ys: numbers(vehicle.rows, yColumn),
      },
    ];
  });
  const chart = overlayChart(
    present.map((item) => ({ name: item.name, xs: item.xs, ys: item.ys })),
    xColumn,
    yColumn,
    PLOT_W,
    PLOT_H,
  );
  if (chart == null) return null;
  const colorOf = new Map(present.map((item) => [item.name, item.color]));
  return {
    geometry: chart.geometry,
    lines: chart.series.map((item) => ({
      name: item.name,
      color: colorOf.get(item.name) ?? fallbackColor,
      points: item.polyline,
    })),
  };
}

function GroundTrack({
  program,
  geometry,
  lines,
}: {
  program: string | null;
  geometry: ChartGeometry;
  lines: ChartLine[];
}) {
  const title = groundTrackTitle(program);
  return (
    <>
      <p className={CAPTION_CLASS}>{title}</p>
      <AxisChart geometry={geometry} lines={lines} label={title} />
    </>
  );
}

function AxisChart({
  geometry,
  lines,
  label,
}: {
  geometry: ChartGeometry;
  lines: ChartLine[];
  label: string;
}) {
  return (
    <svg
      className="mb-3 w-full rounded border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900"
      viewBox={`0 0 ${PLOT_W} ${PLOT_H}`}
      role="img"
      aria-label={label}
    >
      <text
        className="fill-slate-700 font-sans dark:fill-slate-200"
        fontSize="11"
        x={geometry.yTitle.x}
        y={geometry.yTitle.y}
      >
        {geometry.yLabel}
      </text>
      <rect
        x={geometry.frame.x}
        y={geometry.frame.y}
        width={geometry.frame.w}
        height={geometry.frame.h}
        className="fill-none stroke-slate-300 dark:stroke-slate-600"
        strokeWidth="1"
      />
      {geometry.grid.map((line) => (
        <line
          key={`${line.x1},${line.y1},${line.x2},${line.y2}`}
          x1={line.x1}
          y1={line.y1}
          x2={line.x2}
          y2={line.y2}
          className="stroke-slate-200 dark:stroke-slate-700"
          strokeWidth="1"
        />
      ))}
      {geometry.yTicks.map((tick) => (
        <g key={`y-${tick.label}`}>
          <line
            x1={tick.x}
            x2={tick.x + 4}
            y1={tick.y}
            y2={tick.y}
            className="stroke-slate-400 dark:stroke-slate-500"
            strokeWidth="1"
          />
          <text
            className="fill-slate-600 font-sans dark:fill-slate-300"
            fontSize="10"
            textAnchor="end"
            x={tick.x - 4}
            y={tick.y + 3}
          >
            {tick.label}
          </text>
        </g>
      ))}
      {geometry.xTicks.map((tick, index) => (
        <g key={`x-${tick.label}`}>
          <line
            x1={tick.x}
            x2={tick.x}
            y1={tick.y}
            y2={tick.y - 4}
            className="stroke-slate-400 dark:stroke-slate-500"
            strokeWidth="1"
          />
          <text
            className="fill-slate-600 font-sans dark:fill-slate-300"
            fontSize="10"
            textAnchor={index === 0 ? "start" : index === geometry.xTicks.length - 1 ? "end" : "middle"}
            x={tick.x}
            y={tick.y + 12}
          >
            {tick.label}
          </text>
        </g>
      ))}
      <text
        className="fill-slate-700 font-sans dark:fill-slate-200"
        fontSize="11"
        textAnchor="middle"
        x={geometry.xTitle.x}
        y={geometry.xTitle.y}
      >
        {geometry.xLabel}
      </text>
      {lines.map((line) => (
        <polyline key={line.name} fill="none" stroke={line.color} strokeWidth="1.5" points={line.points} />
      ))}
      {lines.length > 1 ? (
        <g>
          {lines.map((line, index) => {
            const y = 14 + index * 12;
            return (
              <g key={`legend-${line.name}`}>
                <line
                  x1={PLOT_W - 92}
                  y1={y}
                  x2={PLOT_W - 80}
                  y2={y}
                  stroke={line.color}
                  strokeWidth={2}
                />
                <text
                  className="fill-slate-700 font-sans dark:fill-slate-200"
                  fontSize="10"
                  x={PLOT_W - 76}
                  y={y + 3}
                >
                  {line.name}
                </text>
              </g>
            );
          })}
        </g>
      ) : null}
    </svg>
  );
}

export function ResultsPane() {
  const program = useStore(store, (s) => s.program);
  const lastPlot = useStore(store, (s) => s.lastPlot);
  const selected = useStore(store, (s) => s.selectedColumns);
  const toggleSelectedColumn = useStore(store, (s) => s.toggleSelectedColumn);

  const sources = useMemo(() => {
    if (lastPlot == null) return [];
    if (lastPlot.vehicles != null && lastPlot.vehicles.length > 0) return lastPlot.vehicles;
    return [{ name: "vehicle", columns: lastPlot.columns, rows: lastPlot.rows }];
  }, [lastPlot]);
  const pickable = useMemo(() => {
    const names: string[] = [];
    const seen = new Set<string>();
    for (const source of sources) {
      for (const name of source.columns) {
        if (name === "time" || seen.has(name)) continue;
        seen.add(name);
        names.push(name);
      }
    }
    return names;
  }, [sources]);
  const columnModules = useMemo(() => {
    if (lastPlot == null) return {};
    if (lastPlot.vehicles != null && lastPlot.vehicles.length > 0) {
      return mergedColumnModules(lastPlot.vehicles);
    }
    return mergedColumnModules([{ columns: lastPlot.columns, modules: lastPlot.modules }]);
  }, [lastPlot]);
  const groups = useMemo(
    () => groupCurves(pickable, columnModules),
    [columnModules, pickable],
  );
  const multi = sources.length > 1;
  const series = useMemo(() => {
    return selected.map((name, index) => ({
      name,
      chart: chartLines(sources, "time", name, multi, SERIES_COLORS[index % SERIES_COLORS.length]),
    }));
  }, [multi, selected, sources]);
  const groundChart = useMemo(() => {
    if (!sources.some((source) => hasGroundTrack(source.columns))) return null;
    return chartLines(sources, "lonx", "latx", multi, "#0f766e");
  }, [multi, sources]);
  const tracks = useMemo(
    () => (lastPlot == null ? [] : trajectoriesFromPlot(lastPlot)),
    [lastPlot],
  );
  const scene = useMemo(() => sceneOfTracks(tracks), [tracks]);
  const trajectoryKind = tracks[0]?.trajectory.kind ?? null;
  const trajectoryLabel = trajectoryKind == null ? "" : trajectoryTitle(program, trajectoryKind);

  if (lastPlot == null) {
    return (
      <div className="text-sm text-slate-500 dark:text-slate-400">
        <h2 className="mb-3 text-base font-semibold text-slate-800 dark:text-slate-100">Results</h2>
        <p>run a case</p>
      </div>
    );
  }

  return (
    <div className="text-sm text-slate-700 dark:text-slate-200">
      <h2 className="mb-3 text-base font-semibold text-slate-800 dark:text-slate-100">Results</h2>
      {series.map((item) => {
        if (item.chart == null) return null;
        const title = seriesTitle(program, item.name);
        return (
          <Fragment key={item.name}>
            <p className={CAPTION_CLASS}>{title}</p>
            <AxisChart geometry={item.chart.geometry} lines={item.chart.lines} label={title} />
          </Fragment>
        );
      })}
      {groundChart != null ? (
        <GroundTrack program={program} geometry={groundChart.geometry} lines={groundChart.lines} />
      ) : null}
      {scene != null && trajectoryKind != null ? (
        <>
          <TrajectoryStill program={program} kind={trajectoryKind} scene={scene} />
          <TrajectoryView scene={scene} label={trajectoryLabel} />
        </>
      ) : null}
      <fieldset className="space-y-1">
        <legend className="mb-1 text-xs font-medium text-slate-500 dark:text-slate-400">columns vs time</legend>
        {groups.map((group) => (
          <div key={group.physics}>
            <h3 className="mb-1 mt-2 text-xs font-semibold text-slate-600 dark:text-slate-300">
              {group.physics}
            </h3>
            {group.loose.map((name) => (
              <label key={name} className="flex items-center gap-2 pl-2">
                <input
                  type="checkbox"
                  checked={selected.includes(name)}
                  onChange={() => toggleSelectedColumn(name)}
                />
                <FieldLabel label={name} hint={columnHint(program, name)} />
              </label>
            ))}
            {group.modules.map((mod) => (
              <div key={mod.module}>
                <h4 className="mb-1 mt-1 pl-2 text-xs font-medium text-slate-500 dark:text-slate-400">
                  {mod.module}
                </h4>
                {mod.names.map((name) => (
                  <label key={name} className="flex items-center gap-2 pl-4">
                    <input
                      type="checkbox"
                      checked={selected.includes(name)}
                      onChange={() => toggleSelectedColumn(name)}
                    />
                    <FieldLabel label={name} hint={columnHint(program, name)} />
                  </label>
                ))}
              </div>
            ))}
          </div>
        ))}
      </fieldset>
    </div>
  );
}
