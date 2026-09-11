export const OPTION_KEYS = [
  "scrn",
  "events",
  "plot",
  "doc",
  "csv",
  "tabout",
  "merge",
  "comscrn",
  "traj",
  "stat",
] as const;

export type OptionKey = (typeof OPTION_KEYS)[number];

export type VehicleEvent = {
  when: Record<string, unknown>;
  set: Record<string, unknown>;
};

export type Vehicle = {
  type: string;
  name: string;
  family?: string;
  aero_deck?: string;
  prop_deck?: string;
  weather_deck?: string;
  sam_deck?: string;
  srmb_deck?: string;
  params: Record<string, number | string>;
  events: VehicleEvent[];
};

export type ModuleSpec = {
  name: string;
  phases: string[];
};

export type Scenario = {
  title: string;
  options: Record<string, boolean>;
  modules: ModuleSpec[];
  timing: Record<string, number>;
  end_time: number;
  family?: string;
  iseed?: number;
  vehicles: Vehicle[];
};

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value == null || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function optionalString(value: unknown): string | undefined {
  const s = asString(value);
  return s == null || s === "" ? undefined : s;
}

function parseOptions(raw: unknown): Record<string, boolean> {
  const rec = asRecord(raw);
  if (rec == null) return {};
  const options: Record<string, boolean> = {};
  for (const [key, value] of Object.entries(rec)) {
    options[key] = Boolean(value);
  }
  return options;
}

function parseModules(raw: unknown): ModuleSpec[] {
  if (!Array.isArray(raw)) return [];
  const modules: ModuleSpec[] = [];
  for (const item of raw) {
    const rec = asRecord(item);
    if (rec == null) continue;
    const name = asString(rec.name) ?? "";
    const phases = Array.isArray(rec.phases)
      ? rec.phases.filter((p): p is string => typeof p === "string")
      : [];
    modules.push({ name, phases });
  }
  return modules;
}

function parseTiming(raw: unknown): Record<string, number> {
  const rec = asRecord(raw);
  if (rec == null) return {};
  const timing: Record<string, number> = {};
  for (const [key, value] of Object.entries(rec)) {
    const n = asNumber(value);
    if (n != null) timing[key] = n;
  }
  return timing;
}

function parseParams(raw: unknown): Record<string, number | string> {
  const rec = asRecord(raw);
  if (rec == null) return {};
  const params: Record<string, number | string> = {};
  for (const [key, value] of Object.entries(rec)) {
    if (typeof value === "number" && Number.isFinite(value)) params[key] = value;
    else if (typeof value === "string") params[key] = value;
  }
  return params;
}

function parseEvents(raw: unknown): VehicleEvent[] {
  if (!Array.isArray(raw)) return [];
  const events: VehicleEvent[] = [];
  for (const item of raw) {
    const rec = asRecord(item);
    if (rec == null) continue;
    events.push({
      when: asRecord(rec.when) ?? {},
      set: asRecord(rec.set) ?? {},
    });
  }
  return events;
}

function parseVehicle(raw: unknown): Vehicle | null {
  const rec = asRecord(raw);
  if (rec == null) return null;
  const type = asString(rec.type);
  const name = asString(rec.name);
  if (type == null || name == null) return null;
  const vehicle: Vehicle = {
    type,
    name,
    params: parseParams(rec.params),
    events: parseEvents(rec.events),
  };
  const family = optionalString(rec.family);
  const aero_deck = optionalString(rec.aero_deck);
  const prop_deck = optionalString(rec.prop_deck);
  const weather_deck = optionalString(rec.weather_deck);
  const sam_deck = optionalString(rec.sam_deck);
  const srmb_deck = optionalString(rec.srmb_deck);
  if (family != null) vehicle.family = family;
  if (aero_deck != null) vehicle.aero_deck = aero_deck;
  if (prop_deck != null) vehicle.prop_deck = prop_deck;
  if (weather_deck != null) vehicle.weather_deck = weather_deck;
  if (sam_deck != null) vehicle.sam_deck = sam_deck;
  if (srmb_deck != null) vehicle.srmb_deck = srmb_deck;
  return vehicle;
}

export function scenarioFromJson(raw: unknown): Scenario {
  const rec = asRecord(raw) ?? {};
  const scenario: Scenario = {
    title: asString(rec.title) ?? "",
    options: parseOptions(rec.options),
    modules: parseModules(rec.modules),
    timing: parseTiming(rec.timing),
    end_time: asNumber(rec.end_time) ?? 0,
    vehicles: Array.isArray(rec.vehicles)
      ? rec.vehicles.map(parseVehicle).filter((v): v is Vehicle => v != null)
      : [],
  };
  const family = optionalString(rec.family);
  const iseed = asNumber(rec.iseed);
  if (family != null) scenario.family = family;
  if (iseed != null) scenario.iseed = iseed;
  return scenario;
}

function vehicleToJson(vehicle: Vehicle): Vehicle {
  const out: Vehicle = {
    type: vehicle.type,
    name: vehicle.name,
    params: { ...vehicle.params },
    events: vehicle.events.map((e) => ({
      when: { ...e.when },
      set: { ...e.set },
    })),
  };
  if (vehicle.family != null) out.family = vehicle.family;
  if (vehicle.aero_deck != null) out.aero_deck = vehicle.aero_deck;
  if (vehicle.prop_deck != null) out.prop_deck = vehicle.prop_deck;
  if (vehicle.weather_deck != null) out.weather_deck = vehicle.weather_deck;
  if (vehicle.sam_deck != null) out.sam_deck = vehicle.sam_deck;
  if (vehicle.srmb_deck != null) out.srmb_deck = vehicle.srmb_deck;
  return out;
}

export function scenarioToJson(scenario: Scenario): Scenario {
  const out: Scenario = {
    title: scenario.title,
    options: { ...scenario.options },
    modules: scenario.modules.map((m) => ({ name: m.name, phases: [...m.phases] })),
    timing: { ...scenario.timing },
    end_time: scenario.end_time,
    vehicles: scenario.vehicles.map(vehicleToJson),
  };
  if (scenario.family != null) out.family = scenario.family;
  if (scenario.iseed != null) out.iseed = scenario.iseed;
  return out;
}

function drawerVehicleError(raw: unknown): string | null {
  const rec = asRecord(raw);
  if (rec == null) return "vehicle must be an object";
  if (asString(rec.type) == null || asString(rec.name) == null) {
    return "vehicle missing type or name";
  }
  return null;
}

export function parseDrawerJson(
  text: string,
): { ok: true; scenario: Scenario } | { ok: false; error: string } {
  try {
    const parsed = JSON.parse(text) as unknown;
    const rec = asRecord(parsed);
    if (rec == null) return { ok: false, error: "scenario must be an object" };
    if (Array.isArray(rec.vehicles)) {
      for (const item of rec.vehicles) {
        const error = drawerVehicleError(item);
        if (error != null) return { ok: false, error };
      }
    }
    return { ok: true, scenario: scenarioFromJson(parsed) };
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    return { ok: false, error };
  }
}

export function commitPhases(draft: string): string[] {
  return draft
    .split(",")
    .map((p) => p.trim())
    .filter((p) => p.length > 0);
}

export function commitParamValue(
  params: Record<string, number | string>,
  key: string,
  draft: string,
): Record<string, number | string> {
  const next = { ...params };
  const trimmed = draft.trim();
  if (trimmed === "") {
    delete next[key];
    return next;
  }
  const num = Number(trimmed);
  next[key] = Number.isFinite(num) ? num : trimmed;
  return next;
}

export function commitNumericRecord(
  rec: Record<string, number>,
  key: string,
  draft: string,
): Record<string, number> {
  const next = { ...rec };
  const trimmed = draft.trim();
  if (trimmed === "") {
    delete next[key];
    return next;
  }
  const num = Number(trimmed);
  if (Number.isFinite(num)) next[key] = num;
  return next;
}
