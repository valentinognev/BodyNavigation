import { useMemo } from "react";
import { useStore } from "zustand";
import { chartGeometry, hasGroundTrack, type ChartGeometry } from "./plot";
import store from "./store";

const SERIES_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"];
const PLOT_W = 360;
const PLOT_H = 200;

function numbers(rows: Record<string, number>[], column: string): number[] {
  return rows.map((row) => Number(row[column]));
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
      {series.map((item, i) =>
        item.geometry == null ? null : (
          <AxisChart
            key={item.name}
            geometry={item.geometry}
            color={SERIES_COLORS[i % SERIES_COLORS.length]}
            label={`${item.geometry.yLabel} vs ${item.geometry.xLabel}`}
          />
        ),
      )}
      {groundGeometry != null ? (
        <>
          <p className="mb-1 text-xs font-medium text-slate-500 dark:text-slate-400">latx vs lonx</p>
          <AxisChart
            geometry={groundGeometry}
            color="#0f766e"
            label={`${groundGeometry.yLabel} vs ${groundGeometry.xLabel}`}
          />
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
            <span>{name}</span>
          </label>
        ))}
      </fieldset>
    </div>
  );
}
