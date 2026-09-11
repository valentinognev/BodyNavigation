export type CatalogCase = {
  stem: string;
  title: string;
};

export type CatalogProgram = {
  id: string;
  label: string;
  cases: CatalogCase[];
};

export type Catalog = {
  programs: CatalogProgram[];
};

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value == null || typeof value !== "object") return null;
  return value as Record<string, unknown>;
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function parseCase(raw: unknown): CatalogCase | null {
  const rec = asRecord(raw);
  if (rec == null) return null;
  const stem = asString(rec.stem);
  if (stem == null) return null;
  const title = asString(rec.title) ?? "";
  return { stem, title };
}

function parseProgram(raw: unknown): CatalogProgram | null {
  const rec = asRecord(raw);
  if (rec == null) return null;
  const id = asString(rec.id);
  const label = asString(rec.label);
  if (id == null || label == null) return null;
  const casesRaw = rec.cases;
  const cases: CatalogCase[] = [];
  if (Array.isArray(casesRaw)) {
    for (const item of casesRaw) {
      const parsed = parseCase(item);
      if (parsed != null) cases.push(parsed);
    }
  }
  return { id, label, cases };
}

export function parseCatalog(raw: unknown): Catalog {
  const rec = asRecord(raw);
  if (rec == null || !Array.isArray(rec.programs)) {
    return { programs: [] };
  }
  const programs: CatalogProgram[] = [];
  for (const item of rec.programs) {
    const parsed = parseProgram(item);
    if (parsed != null) programs.push(parsed);
  }
  return { programs };
}
