export type PhysicsName =
  | "Position"
  | "Velocity"
  | "Attitude"
  | "Load"
  | "Rate"
  | "Aero"
  | "Propulsion"
  | "Guidance"
  | "Other";

export const PHYSICS_ORDER: readonly PhysicsName[] = [
  "Position",
  "Velocity",
  "Attitude",
  "Load",
  "Rate",
  "Aero",
  "Propulsion",
  "Guidance",
  "Other",
];

export const MODULE_ORDER: readonly string[] = [
  "environment",
  "kinematics",
  "newton",
  "aerodynamics",
  "forces",
  "euler",
  "attitude",
  "propulsion",
  "guidance",
  "control",
  "actuator",
  "seeker",
];

export const PHYSICS_BY_STEM: Record<string, PhysicsName> = {
  alt: "Position", hbe: "Position", latx: "Position", lonx: "Position", dbi: "Position",
  SBEL: "Position", SAEL: "Position", sbeg: "Position", SBEG: "Position",
  range_go: "Position", wp_grdrange: "Position", dwbh: "Position", STBG: "Position",
  dvbe: "Velocity", dvbi: "Velocity", dvba: "Velocity", mach: "Velocity", vmach: "Velocity",
  VBEL: "Velocity", VBEB: "Velocity", vbeg: "Velocity", VBEG: "Velocity",
  velocityx: "Velocity", altd: "Velocity",
  alphax: "Attitude", alphaix: "Attitude", alppx: "Attitude", alp: "Attitude",
  betax: "Attitude", betaix: "Attitude", phix: "Attitude", phipx: "Attitude",
  phibdx: "Attitude", phiblx: "Attitude", psix: "Attitude", psibd: "Attitude",
  psibdx: "Attitude", psiblx: "Attitude", psivdx: "Attitude", psivgx: "Attitude",
  psivlx: "Attitude", thtbdx: "Attitude", thtblx: "Attitude", thtvdx: "Attitude",
  thtvgx: "Attitude", thtvlx: "Attitude", gamma: "Attitude", psisbx: "Attitude",
  thtsbx: "Attitude",
  alx: "Load", anx: "Load", ayx: "Load", avx: "Load", FSPV: "Load", gmax: "Load", gminx: "Load",
  ppx: "Rate", qqx: "Rate", rrx: "Rate", qq: "Rate", omegax: "Rate", omega_rpm: "Rate",
  cl_ov_cd: "Aero", cla: "Aero", dma: "Aero", dmde: "Aero", pdynmc: "Aero", stmarg: "Aero",
  wnp: "Aero", wny: "Aero", zetp: "Aero", zety: "Aero", realp1: "Aero", realp2: "Aero",
  realy1: "Aero", realy2: "Aero", tpsp_ratio: "Aero",
  thrust: "Propulsion", thrst_stoch: "Propulsion", mass: "Propulsion", fmasse: "Propulsion",
  fmassr: "Propulsion", throttle: "Propulsion", fidle: "Propulsion", idle: "Propulsion",
  mprop: "Propulsion", power: "Propulsion", tav: "Propulsion", mil: "Propulsion",
  max: "Propulsion", cg: "Propulsion", xcg: "Propulsion", phis: "Propulsion", phisd: "Propulsion",
  alcomx: "Guidance", ancomx: "Guidance", altcom: "Guidance", phicx: "Guidance",
  phimvx: "Guidance", psivgcx: "Guidance", psivlcx: "Guidance", thtvgcx: "Guidance",
  delax: "Guidance", delex: "Guidance", delrx: "Guidance", delacx: "Guidance",
  delecx: "Guidance", delrcx: "Guidance", nl_gain: "Guidance", wp_flag: "Guidance",
  write: "Guidance", modes: "Guidance", mtargeting: "Guidance", time_go: "Guidance",
  tip: "Guidance", zetlagr: "Guidance",
  erq: "Other", etbl: "Other", lconv: "Other", sim_time: "Other",
};

const COMPONENT = /^(.+)([123])$/;

export function physicsOf(name: string): PhysicsName {
  const match = COMPONENT.exec(name);
  const stem = match != null && match[1] in PHYSICS_BY_STEM ? match[1] : name;
  return PHYSICS_BY_STEM[stem] ?? "Other";
}

export type CurveModuleGroup = { module: string; names: string[] };

export type CurveGroup = {
  physics: PhysicsName;
  loose: string[];
  modules: CurveModuleGroup[];
};

function moduleRank(name: string): [number, string] {
  const index = MODULE_ORDER.indexOf(name);
  if (index === -1) return [MODULE_ORDER.length, name];
  return [index, ""];
}

export function groupCurves(names: string[], modules: Record<string, string>): CurveGroup[] {
  const loose = new Map<PhysicsName, string[]>();
  const byModule = new Map<PhysicsName, Map<string, string[]>>();
  for (const name of names) {
    const physics = physicsOf(name);
    const module = modules[name];
    if (module == null || module === "") {
      const bucket = loose.get(physics) ?? [];
      bucket.push(name);
      loose.set(physics, bucket);
      continue;
    }
    const groups = byModule.get(physics) ?? new Map<string, string[]>();
    const bucket = groups.get(module) ?? [];
    bucket.push(name);
    groups.set(module, bucket);
    byModule.set(physics, groups);
  }
  const result: CurveGroup[] = [];
  for (const physics of PHYSICS_ORDER) {
    const modulesFor = byModule.get(physics);
    if ((loose.get(physics)?.length ?? 0) === 0 && (modulesFor == null || modulesFor.size === 0)) {
      continue;
    }
    const moduleNames = modulesFor == null ? [] : [...modulesFor.keys()];
    moduleNames.sort((a, b) => {
      const ra = moduleRank(a);
      const rb = moduleRank(b);
      if (ra[0] !== rb[0]) return ra[0] - rb[0];
      return ra[1] < rb[1] ? -1 : ra[1] > rb[1] ? 1 : 0;
    });
    result.push({
      physics,
      loose: loose.get(physics) ?? [],
      modules: moduleNames.map((module) => ({ module, names: modulesFor?.get(module) ?? [] })),
    });
  }
  return result;
}

export function mergedColumnModules(
  sources: { columns: string[]; modules?: Record<string, string> }[],
): Record<string, string> {
  const merged: Record<string, string> = {};
  for (const source of sources) {
    const table = source.modules ?? {};
    for (const name of source.columns) {
      if (name === "time" || merged[name]) continue;
      const module = table[name];
      if (module) merged[name] = module;
    }
  }
  return merged;
}
