import { useMemo } from "react";
import { useStore } from "zustand";
import { hasGroundTrack, polylinePoints } from "./plot";
import store from "./store";

const SERIES_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"];
const PLOT_W = 320;
const PLOT_H = 160;

function numbers(rows: Record<string, number>[], column: string): number[] {
  return rows.map((row) => Number(row[column]));
}

export function ResultsPane() {
  const lastPlot = useStore(store, (s) => s.lastPlot);
  const selected = useStore(store, (s) => s.selectedColumns);
  const toggleSelectedColumn = useStore(store, (s) => s.toggleSelectedColumn);

  const pickable = lastPlot?.columns.filter((name) => name !== "time") ?? [];
  const times = useMemo(() => (lastPlot == null ? [] : numbers(lastPlot.rows, "time")), [lastPlot]);
  const ground = lastPlot != null && hasGroundTrack(lastPlot.columns);

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
      <svg
        className="mb-2 w-full rounded border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900"
        viewBox={`0 0 ${PLOT_W} ${PLOT_H}`}
        role="img"
        aria-label="columns vs time"
      >
        {selected.map((name, i) => (
          <polyline
            key={name}
            fill="none"
            stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
            strokeWidth="1.5"
            points={polylinePoints(times, numbers(lastPlot.rows, name), PLOT_W, PLOT_H)}
          />
        ))}
      </svg>
      <ul className="mb-3 flex flex-wrap gap-2 text-xs">
        {selected.map((name, i) => (
          <li key={name} className="flex items-center gap-1">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: SERIES_COLORS[i % SERIES_COLORS.length] }}
            />
            {name}
          </li>
        ))}
      </ul>
      {ground ? (
        <>
          <p className="mb-1 text-xs font-medium text-slate-500 dark:text-slate-400">latx vs lonx</p>
          <svg
            className="mb-3 w-full rounded border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900"
            viewBox={`0 0 ${PLOT_W} ${PLOT_H}`}
            role="img"
            aria-label="latx vs lonx"
          >
            <polyline
              fill="none"
              stroke="#0f766e"
              strokeWidth="1.5"
              points={polylinePoints(
                numbers(lastPlot.rows, "lonx"),
                numbers(lastPlot.rows, "latx"),
                PLOT_W,
                PLOT_H,
              )}
            />
          </svg>
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
