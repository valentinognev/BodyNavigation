import { useStore } from "zustand";
import store from "./store";

export function NamelistDrawer() {
  const scenario = useStore(store, (s) => s.scenario);
  const drawerText = useStore(store, (s) => s.drawerText);
  const drawerDirty = useStore(store, (s) => s.drawerDirty);
  const parseError = useStore(store, (s) => s.parseError);
  const setDrawerText = useStore(store, (s) => s.setDrawerText);
  const value = drawerDirty
    ? drawerText
    : scenario == null
      ? ""
      : JSON.stringify(scenario, null, 2);

  return (
    <div className="flex h-40 shrink-0 flex-col border-t border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-center gap-2 px-2 py-1 text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
        JSONC
        {parseError ? (
          <span className="normal-case text-red-600 dark:text-red-400">{parseError.message}</span>
        ) : null}
      </div>
      <textarea
        className="min-h-0 flex-1 resize-none border-0 bg-transparent p-2 font-mono text-sm text-slate-900 outline-none dark:text-slate-100"
        value={value}
        onChange={(e) => setDrawerText(e.target.value)}
        spellCheck={false}
        aria-label="JSONC"
      />
    </div>
  );
}
