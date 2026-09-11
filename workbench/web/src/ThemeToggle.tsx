import { useStore } from "zustand";
import store from "./store";
import { nextTheme, persistTheme } from "./theme";

export function ThemeToggle() {
  const theme = useStore(store, (s) => s.theme);
  const next = nextTheme(theme);
  return (
    <button
      type="button"
      className="rounded border border-slate-300 bg-white px-3 py-1 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
      aria-label={`Switch to ${next} theme`}
      onClick={() => {
        persistTheme(next);
        store.getState().setTheme(next);
      }}
    >
      {next === "dark" ? "Dark" : "Light"}
    </button>
  );
}
