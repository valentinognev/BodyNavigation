import { useRef, useState } from "react";
import { useStore } from "zustand";
import { groupCatalog } from "./catalog";
import { openImportedFile } from "./editorActions";
import store from "./store";
import { ThemeToggle } from "./ThemeToggle";

export function StartScreen() {
  const catalog = useStore(store, (s) => s.catalog);
  const catalogError = useStore(store, (s) => s.catalogError);
  const parseError = useStore(store, (s) => s.parseError);
  const [openId, setOpenId] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const sections = catalog == null ? [] : groupCatalog(catalog);

  return (
    <div className="flex h-full flex-col bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <div className="flex flex-col items-center gap-4 px-4 pt-8">
        <h1 className="text-2xl font-semibold">CADAC</h1>
        <div className="flex gap-2">
          <button
            type="button"
            className="rounded border border-slate-300 bg-white px-4 py-2 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
            onClick={() => fileRef.current?.click()}
          >
            Open file
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".jsonc,.asc"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) void openImportedFile(store, file);
            }}
          />
          <ThemeToggle />
        </div>
        {parseError ? (
          <p className="text-sm text-red-600 dark:text-red-400">{parseError.message}</p>
        ) : null}
        {catalogError ? (
          <p className="text-sm text-red-600 dark:text-red-400">{catalogError}</p>
        ) : catalog == null ? (
          <p className="text-sm text-slate-500 dark:text-slate-400">Loading catalog…</p>
        ) : catalog.programs.length === 0 ? (
          <p className="text-sm text-slate-500 dark:text-slate-400">No programs in catalog.</p>
        ) : null}
      </div>
      {sections.length > 0 ? (
        <div className="mx-auto mt-4 w-full max-w-md flex-1 overflow-auto px-4 pb-8 text-sm">
          {sections.map((section) => (
            <section key={section.id} className="mb-4">
              <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                {section.label}
              </h2>
              {section.groups.map((group) => (
                <div key={group.id} className="mt-2">
                  {group.label ? (
                    <h3 className="pl-2 text-xs text-slate-500 dark:text-slate-400">{group.label}</h3>
                  ) : null}
                  <ul className="mt-1 space-y-1">
                    {group.programs.map((program) => {
                      const open = openId === program.id;
                      return (
                        <li key={program.id}>
                          <button
                            type="button"
                            className={`w-full rounded px-3 py-2 text-left ${
                              open
                                ? "bg-slate-800 text-white dark:bg-slate-100 dark:text-slate-900"
                                : "bg-white text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
                            }`}
                            onClick={() => setOpenId(open ? null : program.id)}
                          >
                            {program.label}
                          </button>
                          {open ? (
                            <ul className="mt-1 space-y-1 pl-4">
                              {program.cases.map((c) => (
                                <li key={c.stem}>
                                  <button
                                    type="button"
                                    className="w-full rounded px-3 py-1.5 text-left text-slate-600 hover:bg-slate-200 dark:text-slate-300 dark:hover:bg-slate-800"
                                    onClick={() => store.getState().openCase(program.id, c.stem)}
                                  >
                                    {c.title || c.stem}
                                  </button>
                                </li>
                              ))}
                            </ul>
                          ) : null}
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))}
            </section>
          ))}
        </div>
      ) : null}
    </div>
  );
}
