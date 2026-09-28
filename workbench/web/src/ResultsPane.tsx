import { Fragment, useMemo } from "react";
import { useStore } from "zustand";
import { FieldLabel } from "./forms/fields";
import { chartGeometry, hasGroundTrack, type ChartGeometry } from "./plot";
import {
  downloadBytes,
  renderSceneGif,
  sceneFrame,
  sceneOf,
  trajectoryOf,
  turntableAzimuths,
} from "./plot3d";
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
  xName,
  yName,
  zName,
  points,
}: {
  program: string | null;
  kind: "geographic" | "local";
  xName: string;
  yName: string;
  zName: string;
  points: { x: number; y: number; z: number }[];
}) {
  const title = trajectoryTitle(program, kind);
  const scene = sceneOf({ kind, xName, yName, zName, points });
  if (scene == null) return null;
  const frame = sceneFrame(scene, STILL_AZIMUTH, PLOT_W, PLOT_H, 28);
  const start = frame.segments.find((segment) => segment.color === "start")?.points[0];
  const end = frame.segments.find((segment) => segment.color === "end")?.points[0];
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
        {start != null ? <circle cx={start.sx} cy={start.sy} r={2.5} fill="#15803d" /> : null}
        {end != null ? (
          <rect x={end.sx - 2.5} y={end.sy - 2.5} width={5} height={5} fill="#b91c1c" />
        ) : null}
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

function GroundTrack({
  program,
  geometry,
}: {
  program: string | null;
  geometry: ChartGeometry;
}) {
  const title = groundTrackTitle(program);
  return (
    <>
      <p className={CAPTION_CLASS}>{title}</p>
      <AxisChart geometry={geometry} color="#0f766e" label={title} />
    </>
  );
}

function AxisChart({
  geometry,
  color,
  label,
}: {
  geometry: ChartGeometry;
  color: string;
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
      <polyline fill="none" stroke={color} strokeWidth="1.5" points={geometry.polyline} />
    </svg>
  );
}

export function ResultsPane() {
  const program = useStore(store, (s) => s.program);
  const lastPlot = useStore(store, (s) => s.lastPlot);
  const selected = useStore(store, (s) => s.selectedColumns);
  const toggleSelectedColumn = useStore(store, (s) => s.toggleSelectedColumn);

  const pickable = lastPlot?.columns.filter((name) => name !== "time") ?? [];
  const times = useMemo(() => (lastPlot == null ? [] : numbers(lastPlot.rows, "time")), [lastPlot]);
  const ground = lastPlot != null && hasGroundTrack(lastPlot.columns);
  const series = useMemo(() => {
    if (lastPlot == null) return [];
    return selected.map((name) => ({
      name,
      geometry: chartGeometry(times, numbers(lastPlot.rows, name), "time", name, PLOT_W, PLOT_H),
    }));
  }, [lastPlot, selected, times]);
  const groundGeometry = useMemo(() => {
    if (lastPlot == null || !ground) return null;
    return chartGeometry(
      numbers(lastPlot.rows, "lonx"),
      numbers(lastPlot.rows, "latx"),
      "lonx",
      "latx",
      PLOT_W,
      PLOT_H,
    );
  }, [ground, lastPlot]);
  const trajectory = useMemo(
    () => (lastPlot == null ? null : trajectoryOf(lastPlot.columns, lastPlot.rows)),
    [lastPlot],
  );
  const scene = useMemo(() => (trajectory == null ? null : sceneOf(trajectory)), [trajectory]);
  const trajectoryLabel = trajectory == null ? "" : trajectoryTitle(program, trajectory.kind);

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
      {series.map((item, i) => {
        if (item.geometry == null) return null;
        const title = seriesTitle(program, item.name);
        return (
          <Fragment key={item.name}>
            <p className={CAPTION_CLASS}>{title}</p>
            <AxisChart
              geometry={item.geometry}
              color={SERIES_COLORS[i % SERIES_COLORS.length]}
              label={title}
            />
          </Fragment>
        );
      })}
      {groundGeometry != null ? (
        <GroundTrack program={program} geometry={groundGeometry} />
      ) : null}
      {trajectory != null ? (
        <>
          <TrajectoryStill
            program={program}
            kind={trajectory.kind}
            xName={trajectory.xName}
            yName={trajectory.yName}
            zName={trajectory.zName}
            points={trajectory.points}
          />
          {scene != null ? <TrajectoryView scene={scene} label={trajectoryLabel} /> : null}
        </>
      ) : null}
      <fieldset className="space-y-1">
        <legend className="mb-1 text-xs font-medium text-slate-500 dark:text-slate-400">columns vs time</legend>
        {pickable.map((name) => (
          <label key={name} className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={selected.includes(name)}
              onChange={() => toggleSelectedColumn(name)}
            />
            <FieldLabel label={name} hint={columnHint(program, name)} />
          </label>
        ))}
      </fieldset>
    </div>
  );
}
