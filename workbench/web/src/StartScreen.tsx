import { useRef, useState } from "react";
import { useStore } from "zustand";
import { openImportedFile } from "./editorActions";
import store from "./store";
import { ThemeToggle } from "./ThemeToggle";

export function StartScreen() {
  const catalog = useStore(store, (s) => s.catalog);
  const catalogError = useStore(store, (s) => s.catalogError);
  const parseError = useStore(store, (s) => s.parseError);
  const [openId, setOpenId] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
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
      ) : (
        <ul className="w-full max-w-md space-y-1 text-sm">
          {catalog.programs.map((program) => {
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
      )}
    </div>
  );
}
