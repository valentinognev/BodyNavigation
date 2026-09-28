import { createStore as createVanillaStore } from "zustand/vanilla";
import type { Catalog } from "./catalog";
import type { NavId, ParseErrorInfo } from "./nav";
import { defaultColumns } from "./plot";
import { parseDrawerJson, scenarioFromJson, type Scenario } from "./scenario";
import { currentTheme, type Theme } from "./theme";

export type View = "start" | "editor";

export type VehiclePlot = {
  name: string;
  columns: string[];
  rows: Record<string, number>[];
  modules?: Record<string, string>;
};

export type PlotData = {
  columns: string[];
  rows: Record<string, number>[];
  vehicles?: VehiclePlot[];
  modules?: Record<string, string>;
};

export type WorkbenchState = {
  view: View;
  catalog: Catalog | null;
  catalogError: string | null;
  theme: Theme;
  program: string | null;
  stem: string | null;
  scenario: Scenario | null;
  caseDescription: string;
  drawerText: string;
  drawerDirty: boolean;
  revision: number;
  parseError: ParseErrorInfo | null;
  runInFlight: boolean;
  runAbandoned: boolean;
  runError: string | null;
  activeRunId: string | null;
  lastPlot: PlotData | null;
  selectedColumns: string[];
  nav: NavId;
  setCatalog: (catalog: Catalog) => void;
  setCatalogError: (catalogError: string | null) => void;
  setTheme: (theme: Theme) => void;
  setNav: (nav: NavId) => void;
  showStart: () => void;
  setRunInFlight: (runInFlight: boolean) => void;
  setRunAbandoned: (runAbandoned: boolean) => void;
  setRunError: (runError: string | null) => void;
  setActiveRunId: (activeRunId: string | null) => void;
  setLastPlot: (lastPlot: PlotData | null) => void;
  toggleSelectedColumn: (name: string) => void;
  openCase: (program: string, stem: string) => Promise<void>;
  openImported: (raw: unknown, stem: string, description?: string) => void;
  applyFormPatch: (partial: Partial<Scenario>) => void;
  setDrawerText: (text: string) => void;
  applyDrawerJson: () => void;
  applyParseFail: (error: string) => void;
};

function initialTheme(): Theme {
  if (typeof window === "undefined") return "light";
  return currentTheme();
}

function caseUrl(program: string, stem: string): string {
  return `/cases/${encodeURIComponent(program)}/${encodeURIComponent(stem)}`;
}

function sameColumnSet(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const set = new Set(a);
  return b.every((name) => set.has(name));
}

export function createStore() {
  return createVanillaStore<WorkbenchState>((set, get) => ({
    view: "start",
    catalog: null,
    catalogError: null,
    theme: initialTheme(),
    program: null,
    stem: null,
    scenario: null,
    caseDescription: "",
    drawerText: "",
    drawerDirty: false,
    revision: 0,
    parseError: null,
    runInFlight: false,
    runAbandoned: false,
    runError: null,
    activeRunId: null,
    lastPlot: null,
    selectedColumns: [],
    nav: "overview",
    setCatalog: (catalog) => set({ catalog, catalogError: null }),
    setCatalogError: (catalogError) => set({ catalogError }),
    setTheme: (theme) => set({ theme }),
    setNav: (nav) => set({ nav }),
    showStart: () => set({ view: "start" }),
    setRunInFlight: (runInFlight) => set({ runInFlight }),
    setRunAbandoned: (runAbandoned) => set({ runAbandoned }),
    setRunError: (runError) => set({ runError }),
    setActiveRunId: (activeRunId) => set({ activeRunId }),
    setLastPlot: (lastPlot) =>
      set((s) => {
        if (lastPlot == null) return { lastPlot: null, selectedColumns: [] };
        const prev = s.lastPlot?.columns ?? [];
        const changed = s.lastPlot == null || !sameColumnSet(prev, lastPlot.columns);
        const selected =
          s.selectedColumns.length === 0 || changed
            ? defaultColumns(lastPlot.columns)
            : s.selectedColumns;
        return { lastPlot, selectedColumns: selected };
      }),
    toggleSelectedColumn: (name) =>
      set((s) => ({
        selectedColumns: s.selectedColumns.includes(name)
          ? s.selectedColumns.filter((c) => c !== name)
          : [...s.selectedColumns, name],
      })),
    openCase: async (program, stem) => {
      set({
        view: "editor",
        program,
        stem,
        scenario: null,
        caseDescription: "",
        drawerText: "",
        drawerDirty: false,
        parseError: null,
        nav: "overview",
      });
      try {
        const res = await fetch(caseUrl(program, stem));
        if (!res.ok) {
          set({ parseError: { message: `case ${res.status}` } });
          return;
        }
        const body = (await res.json()) as { scenario?: unknown; description?: unknown };
        const scenario = scenarioFromJson(body.scenario);
        const caseDescription = typeof body.description === "string" ? body.description : "";
        set((s) => ({
          scenario,
          caseDescription,
          drawerText: JSON.stringify(scenario, null, 2),
          drawerDirty: false,
          parseError: null,
          revision: s.revision + 1,
        }));
      } catch (err) {
        set({
          parseError: { message: err instanceof Error ? err.message : String(err) },
        });
      }
    },
    openImported: (raw, stem, description = "") => {
      const scenario = scenarioFromJson(raw);
      set((s) => ({
        view: "editor",
        program: null,
        stem,
        scenario,
        caseDescription: description,
        drawerText: JSON.stringify(scenario, null, 2),
        drawerDirty: false,
        parseError: null,
        nav: "overview",
        revision: s.revision + 1,
      }));
    },
    applyFormPatch: (partial) => {
      if (get().drawerDirty) {
        get().applyDrawerJson();
        if (get().drawerDirty) return;
      }
      set((s) => ({
        scenario: s.scenario == null ? null : { ...s.scenario, ...partial },
        revision: s.revision + 1,
        parseError: null,
        drawerDirty: false,
      }));
    },
    setDrawerText: (text) =>
      set((s) => ({
        drawerText: text,
        drawerDirty: true,
        revision: s.revision + 1,
      })),
    applyDrawerJson: () => {
      const result = parseDrawerJson(get().drawerText);
      if (!result.ok) {
        set({ parseError: { message: result.error } });
        return;
      }
      set({ scenario: result.scenario, drawerDirty: false, parseError: null });
    },
    applyParseFail: (error) => set({ parseError: { message: error } }),
  }));
}

const store = createStore();
export default store;
