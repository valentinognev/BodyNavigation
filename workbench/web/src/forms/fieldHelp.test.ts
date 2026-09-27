import { expect, it } from "vitest";
import { schemaDescription, variableDescription } from "./fieldHelp";

it("uses the open program's CADAC sentence", () => {
  expect(variableDescription("hyper3", "alphax")).toBe("Angle of attack - deg");
  expect(variableDescription("hyper3", "mprop")).toBe(
    "=0:none; =1:fixed-throttle; =2:auto-throttle",
  );
  expect(variableDescription("falcon5", "mprop")).toBe("Mode switch - ND");
});

it("falls back to the shared sentence when the program has no entry", () => {
  expect(variableDescription(null, "alphax")).toBe("Angle of attack - deg");
  expect(variableDescription("hyper3", "not_a_cadac_name")).toBeUndefined();
});

it("explains each scenario field", () => {
  expect(schemaDescription("title")).toBe("Run title from the TITLE line.");
  expect(schemaDescription("scenario.family")).toBe(
    "Family for vehicles that do not set their own. Empty uses the global type table.",
  );
  expect(schemaDescription("vehicle.family")).toBe(
    "Family for this vehicle. Empty uses the scenario family, then the global type table.",
  );
  expect(schemaDescription("iseed")).toBe("Seed for stochastic draws.");
  expect(schemaDescription("scrn")).toBe("Display vehicle data on the screen at scrn_step intervals.");
  expect(schemaDescription("events")).toBe("Write event messages to the screen.");
  expect(schemaDescription("plot")).toBe("Write a plot file for each vehicle.");
  expect(schemaDescription("doc")).toBe("Write module-variable definitions to the documentation file.");
  expect(schemaDescription("csv")).toBe("Also write plot, trajectory, and merged plot data as CSV.");
  expect(schemaDescription("tabout")).toBe("Also write the screen output to tabout.asc, without events.");
  expect(schemaDescription("merge")).toBe("Merge each vehicle plot file into one plot file.");
  expect(schemaDescription("comscrn")).toBe("Write combus data to the screen.");
  expect(schemaDescription("traj")).toBe("Write combus data to trajectory files for plotting.");
  expect(schemaDescription("stat")).toBe("Write statistic data to stat.asc.");
  expect(schemaDescription("end_time")).toBe("Time when the run stops, in seconds.");
  expect(schemaDescription("int_step")).toBe("Integration step, in seconds.");
  expect(schemaDescription("plot_step")).toBe("Interval between plot output, in seconds.");
  expect(schemaDescription("scrn_step")).toBe("Interval between screen output, in seconds.");
  expect(schemaDescription("traj_step")).toBe("Interval between trajectory-file output, in seconds.");
  expect(schemaDescription("com_step")).toBe("Interval between combus screen output, in seconds.");
  expect(schemaDescription("module.name")).toBe("Module called in this order.");
  expect(schemaDescription("phases")).toBe("Which parts of the module run: def, init, exec, term.");
  expect(schemaDescription("vehicle.type")).toBe("CADAC vehicle type token, such as CRUISE3 or PLANE6.");
  expect(schemaDescription("vehicle.name")).toBe("Label for this vehicle.");
  expect(schemaDescription("params")).toBe("Initial values of this vehicle's module-variables.");
  expect(schemaDescription("when")).toBe("Watch criteria. Keys are module-variable names.");
  expect(schemaDescription("set")).toBe("Module-variables assigned when the event fires.");
  expect(schemaDescription("aero_deck")).toBe("Aerodynamic table file.");
  expect(schemaDescription("prop_deck")).toBe("Propulsion table file.");
  expect(schemaDescription("weather_deck")).toBe("Weather table file.");
  expect(schemaDescription("sam_deck")).toBe("SAM table file.");
  expect(schemaDescription("srmb_deck")).toBe("SRBM table file.");
});
