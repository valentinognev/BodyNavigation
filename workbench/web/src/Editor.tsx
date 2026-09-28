import { useEffect } from "react";
import { useStore } from "zustand";
import { VALIDATE_MS, debounce } from "./debounce";
import { runValidate, saveCurrentCase } from "./editorActions";
import { Decks } from "./forms/Decks";
import { Events } from "./forms/Events";
import { Modules } from "./forms/Modules";
import { Overview } from "./forms/Overview";
import { Timing } from "./forms/Timing";
import { Vehicles } from "./forms/Vehicles";
import { NamelistDrawer } from "./NamelistDrawer";
import { NAV_ITEMS, runButtonDisabled, saveButtonDisabled, type NavId } from "./nav";
import { ResultsPane } from "./ResultsPane";
import { cancelCurrentRun, startRun } from "./run";
import store from "./store";
import { ThemeToggle } from "./ThemeToggle";

function FormOutlet({ nav }: { nav: NavId }) {
  switch (nav) {
    case "overview":
      return <Overview />;
    case "modules":
      return <Modules />;
    case "vehicles":
      return <Vehicles />;
    case "timing":
      return <Timing />;
    case "events":
      return <Events />;
    case "decks":
      return <Decks />;
    case "results":
      return <ResultsPane />;
  }
}

export function Editor() {
  const program = useStore(store, (s) => s.program);
  const stem = useStore(store, (s) => s.stem);
  const scenario = useStore(store, (s) => s.scenario);
  const parseError = useStore(store, (s) => s.parseError);
  const runError = useStore(store, (s) => s.runError);
  const runInFlight = useStore(store, (s) => s.runInFlight);
  const runProgress = useStore(store, (s) => s.runProgress);
  const nav = useStore(store, (s) => s.nav);
  const caseName = stem ?? "";

  useEffect(() => {
    const queued = debounce(() => {
      void runValidate(store);
    }, VALIDATE_MS);
    return store.subscribe((state, prev) => {
      if (state.revision === prev.revision) return;
      queued();
    });
  }, []);

  return (
    <div className="flex h-full flex-col bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="flex items-center gap-2 border-b border-slate-200 bg-white px-3 py-2 dark:border-slate-800 dark:bg-slate-900">
        <div className="mr-auto min-w-0 flex-1 pr-3">
          <h1 className="truncate text-lg font-semibold">CADAC</h1>
          <p className="truncate text-xs text-slate-500 dark:text-slate-400" title={caseName}>
            <span className="font-medium text-slate-700 dark:text-slate-200">{caseName}</span>
            {program != null ? (
              <span>
                {" "}
                ({program}/{stem})
              </span>
            ) : null}
          </p>
          {runError ? (
            <p className="truncate text-xs text-red-600 dark:text-red-400" title={runError}>
              {runError}
            </p>
          ) : null}
        </div>
        <button
          type="button"
          className="rounded border border-slate-300 bg-white px-3 py-1 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
          onClick={() => store.getState().showStart()}
        >
          Home
        </button>
        <ThemeToggle />
        <button
          type="button"
          className="rounded border border-slate-300 bg-white px-3 py-1 text-sm disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
          disabled={saveButtonDisabled(program)}
          title={saveButtonDisabled(program) ? "not in catalog" : undefined}
          onClick={() => void saveCurrentCase(store)}
        >
          Save
        </button>
        <button
          type="button"
          className="rounded bg-slate-800 px-3 py-1 text-sm text-white disabled:cursor-not-allowed disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900"
          disabled={!runInFlight && runButtonDisabled(parseError, runInFlight, program)}
          title={
            !runInFlight && program == null ? "not in catalog" : undefined
          }
          onClick={() => {
            if (store.getState().runInFlight) {
              void cancelCurrentRun(store);
              return;
            }
            if (runButtonDisabled(store.getState().parseError, store.getState().runInFlight, store.getState().program)) return;
            void startRun(store);
          }}
        >
          {runInFlight ? "Cancel" : "Run"}
        </button>
      </header>
      {runInFlight ? (
        <div
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round((runProgress ?? 0) * 100)}
          className="h-1 w-full shrink-0 bg-slate-200 dark:bg-slate-800"
        >
          <div
            className="h-full bg-slate-800 dark:bg-slate-100"
            style={{ width: `${(runProgress ?? 0) * 100}%` }}
          />
        </div>
      ) : null}
      <div className="flex min-h-0 flex-1">
        <nav className="w-48 shrink-0 overflow-y-auto border-r border-slate-200 bg-white p-2 dark:border-slate-800 dark:bg-slate-900">
          <ul className="space-y-1 text-sm">
            {NAV_ITEMS.map((item) => {
              const active = nav === item.id;
              return (
                <li key={item.id}>
                  <button
                    type="button"
                    className={`w-full rounded px-2 py-1 text-left ${
                      active
                        ? "bg-slate-800 text-white dark:bg-slate-100 dark:text-slate-900"
                        : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
                    }`}
                    onClick={() => store.getState().setNav(item.id)}
                  >
                    {item.label}
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>
        <main className="min-w-0 flex-1 overflow-y-auto p-4">
          {scenario == null && parseError != null ? (
            <p className="text-sm text-red-600 dark:text-red-400">{parseError.message}</p>
          ) : scenario == null ? (
            <p className="text-sm text-slate-500 dark:text-slate-400">Loading…</p>
          ) : (
            <FormOutlet nav={nav} />
          )}
        </main>
        <aside className="w-80 shrink-0 overflow-y-auto border-l border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900">
          <ResultsPane />
        </aside>
      </div>
      <NamelistDrawer />
    </div>
  );
}
