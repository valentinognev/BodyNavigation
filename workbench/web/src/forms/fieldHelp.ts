import { sharedHelp, variableHelp } from "./variableHelp";

const SCHEMA: Record<string, string> = {
  title: "Run title from the TITLE line.",
  "scenario.family":
    "Family for vehicles that do not set their own. Empty uses the global type table.",
  "vehicle.family":
    "Family for this vehicle. Empty uses the scenario family, then the global type table.",
  iseed: "Seed for stochastic draws.",
  scrn: "Display vehicle data on the screen at scrn_step intervals.",
  events: "Write event messages to the screen.",
  plot: "Write a plot file for each vehicle.",
  doc: "Write module-variable definitions to the documentation file.",
  csv: "Also write plot, trajectory, and merged plot data as CSV.",
  tabout: "Also write the screen output to tabout.asc, without events.",
  merge: "Merge each vehicle plot file into one plot file.",
  comscrn: "Write combus data to the screen.",
  traj: "Write combus data to trajectory files for plotting.",
  stat: "Write statistic data to stat.asc.",
  end_time: "Time when the run stops, in seconds.",
  int_step: "Integration step, in seconds.",
  plot_step: "Interval between plot output, in seconds.",
  scrn_step: "Interval between screen output, in seconds.",
  traj_step: "Interval between trajectory-file output, in seconds.",
  com_step: "Interval between combus screen output, in seconds.",
  "module.name": "Module called in this order.",
  phases: "Which parts of the module run: def, init, exec, term.",
  "vehicle.type": "CADAC vehicle type token, such as CRUISE3 or PLANE6.",
  "vehicle.name": "Label for this vehicle.",
  params: "Initial values of this vehicle's module-variables.",
  when: "Watch criteria. Keys are module-variable names.",
  set: "Module-variables assigned when the event fires.",
  aero_deck: "Aerodynamic table file.",
  prop_deck: "Propulsion table file.",
  weather_deck: "Weather table file.",
  sam_deck: "SAM table file.",
  srmb_deck: "SRBM table file.",
};

export function schemaDescription(key: string): string | undefined {
  return SCHEMA[key];
}

export function variableDescription(
  program: string | null | undefined,
  name: string,
): string | undefined {
  if (program) {
    const hit = variableHelp[program]?.[name];
    if (hit) return hit;
  }
  return sharedHelp[name];
}

export function timingDescription(
  program: string | null | undefined,
  name: string,
): string | undefined {
  return schemaDescription(name) ?? variableDescription(program, name);
}
