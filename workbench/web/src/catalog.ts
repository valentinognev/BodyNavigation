export type CatalogCase = {
  stem: string;
  title: string;
};

export type CatalogProgram = {
  id: string;
  label: string;
  cases: CatalogCase[];
  dimension?: string;
  subgroup?: string;
};

export type CatalogGroup = {
  id: string;
  label: string;
  programs: CatalogProgram[];
};

export type CatalogSection = {
  id: string;
  label: string;
  groups: CatalogGroup[];
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

const DIMENSIONS: { id: string; label: string; subgroups: readonly string[] }[] = [
  { id: "xz", label: "2D (X-Z)", subgroups: ["flat", "spinner"] },
  { id: "3", label: "3 DOF", subgroups: ["flat", "round", "spinner"] },
  { id: "5", label: "5 DOF", subgroups: ["flat", "round", "spinner"] },
  { id: "6", label: "6 DOF", subgroups: ["flat", "round", "spinner"] },
];

const SUBGROUP_LABEL: Record<string, string> = {
  flat: "flat",
  round: "round",
  spinner: "spinner",
};

export function groupCatalog(catalog: Catalog): CatalogSection[] {
  const allowed = new Map(DIMENSIONS.map((dimension) => [dimension.id, new Set(dimension.subgroups)]));
  const buckets = new Map<string, CatalogProgram[]>();
  const other: CatalogProgram[] = [];
  for (const program of catalog.programs) {
    const subgroups = program.dimension == null ? undefined : allowed.get(program.dimension);
    if (
      program.dimension == null ||
      program.subgroup == null ||
      subgroups == null ||
      !subgroups.has(program.subgroup)
    ) {
      other.push(program);
      continue;
    }
    const key = `${program.dimension}:${program.subgroup}`;
    const list = buckets.get(key);
    if (list == null) buckets.set(key, [program]);
    else list.push(program);
  }

  const sections: CatalogSection[] = [];
  for (const dimension of DIMENSIONS) {
    const groups: CatalogGroup[] = [];
    for (const subgroup of dimension.subgroups) {
      const programs = buckets.get(`${dimension.id}:${subgroup}`);
      if (programs == null || programs.length === 0) continue;
      groups.push({ id: subgroup, label: SUBGROUP_LABEL[subgroup], programs });
    }
    if (groups.length > 0) sections.push({ id: dimension.id, label: dimension.label, groups });
  }
  if (other.length > 0) {
    sections.push({ id: "other", label: "Other", groups: [{ id: "other", label: "", programs: other }] });
  }
  return sections;
}

function parseProgram(raw: unknown): CatalogProgram | null {
  const rec = asRecord(raw);
  if (rec == null) return null;
  const id = asString(rec.id);
  const label = asString(rec.label);
  if (id == null || label == null) return null;
  const dimension = asString(rec.dimension) ?? undefined;
  const subgroup = asString(rec.subgroup) ?? undefined;
  const casesRaw = rec.cases;
  const cases: CatalogCase[] = [];
  if (Array.isArray(casesRaw)) {
    for (const item of casesRaw) {
      const parsed = parseCase(item);
      if (parsed != null) cases.push(parsed);
    }
  }
  return { id, label, cases, dimension, subgroup };
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
