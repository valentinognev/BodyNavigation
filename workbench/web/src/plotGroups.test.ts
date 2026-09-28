import { expect, it } from "vitest";
import {
  PHYSICS_BY_STEM,
  groupCurves,
  mergedColumnModules,
  physicsOf,
  type PhysicsName,
} from "./plotGroups";

const EXPECTED: Record<string, PhysicsName> = {
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

it("assigns every listed stem and uses a listed stem for a component", () => {
  expect(PHYSICS_BY_STEM).toEqual(EXPECTED);
  expect(physicsOf("SBEL1")).toBe("Position");
  expect(physicsOf("SBEL2")).toBe("Position");
  expect(physicsOf("sbeg1")).toBe("Position");
  expect(physicsOf("SBEG1")).toBe("Position");
  expect(physicsOf("FSPV3")).toBe("Load");
  expect(physicsOf("realp1")).toBe("Aero");
  expect(physicsOf("no_such")).toBe("Other");
});

it("nests modules under physics, loose names first, input order kept", () => {
  expect(
    groupCurves(["mach", "alt", "SBEL1", "foo", "hbe"], { alt: "newton", SBEL1: "newton", mach: "environment" }),
  ).toEqual([
    { physics: "Position", loose: ["hbe"], modules: [{ module: "newton", names: ["alt", "SBEL1"] }] },
    { physics: "Velocity", loose: [], modules: [{ module: "environment", names: ["mach"] }] },
    { physics: "Other", loose: ["foo"], modules: [] },
  ]);
});

it("orders known modules first and unknown modules A to Z", () => {
  expect(
    groupCurves(
      ["psix", "gamma", "alphax"],
      { psix: "targeting", gamma: "trajectory", alphax: "kinematics" },
    ),
  ).toEqual([
    {
      physics: "Attitude",
      loose: [],
      modules: [
        { module: "kinematics", names: ["alphax"] },
        { module: "targeting", names: ["psix"] },
        { module: "trajectory", names: ["gamma"] },
      ],
    },
  ]);
});

it("keeps the first vehicle module and fills columns only that vehicle lacked", () => {
  expect(
    mergedColumnModules([
      { columns: ["time", "alt", "SBEL1"], modules: { alt: "newton", SBEL1: "newton" } },
      { columns: ["alt", "mach"], modules: { alt: "kinematics", mach: "environment" } },
    ]),
  ).toEqual({ alt: "newton", SBEL1: "newton", mach: "environment" });
  expect(mergedColumnModules([{ columns: ["alt"] }])).toEqual({});
});
