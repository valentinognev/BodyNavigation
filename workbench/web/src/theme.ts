export type Theme = "light" | "dark";

export const THEME_STORAGE_KEY = "cadac-theme";

export function resolveTheme(stored: string | null, prefersDark: boolean): Theme {
  if (stored === "light" || stored === "dark") return stored;
  return prefersDark ? "dark" : "light";
}

export function applyThemeClass(
  theme: Theme,
  classList: { add(token: string): void; remove(token: string): void },
): void {
  if (theme === "dark") classList.add("dark");
  else classList.remove("dark");
}

export function nextTheme(theme: Theme): Theme {
  return theme === "dark" ? "light" : "dark";
}

export function readStoredTheme(): string | null {
  try {
    return localStorage.getItem(THEME_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function prefersDarkScheme(): boolean {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function currentTheme(): Theme {
  return resolveTheme(readStoredTheme(), prefersDarkScheme());
}

export function applyTheme(theme: Theme): void {
  applyThemeClass(theme, document.documentElement.classList);
  if (document.body) applyThemeClass(theme, document.body.classList);
}

/** User chose a skin: apply it and remember it. */
export function persistTheme(theme: Theme): void {
  applyTheme(theme);
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    /* private mode */
  }
}
