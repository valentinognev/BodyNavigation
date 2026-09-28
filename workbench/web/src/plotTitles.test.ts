import { expect, it } from "vitest";
import { sharedHelp, variableHelp } from "./forms/variableHelp";
import { groundTrackTitle, plotDescription, seriesTitle, trajectoryTitle } from "./plotTitles";

it("drops the trailing unit clause and appends vs time", () => {
  expect(seriesTitle("hyper3", "alt")).toBe("Vehicle altitude vs time");
  expect(seriesTitle("hyper3", "alphax")).toBe("Angle of attack vs time");
  expect(seriesTitle("hyper3", "dvbe")).toBe("Vehicle speed vs time");
  expect(seriesTitle("agm6", "dvbe")).toBe("Missile speed vs time");
  expect(seriesTitle("aim5", "alt")).toBe("Vehicle altitude vs time");
  expect(seriesTitle("hyper3", "ABII")).toBe("Inertial acceleration vs time");
});

it("uses the symbol when the name has no sentence", () => {
  expect(seriesTitle("hyper3", "foo")).toBe("foo vs time");
  expect(seriesTitle(null, "foo")).toBe("foo vs time");
});

it("uses shared help when the program is missing", () => {
  expect(seriesTitle(null, "alphax")).toBe("Angle of attack vs time");
  expect(seriesTitle(undefined, "alt")).toBe("Vehicle altitude vs time");
  expect(groundTrackTitle(null)).toBe("Vehicle latitude vs vehicle longitude");
  expect(groundTrackTitle(undefined)).toBe("Vehicle latitude vs vehicle longitude");
});

it("names the hyper3 ground track from latitude and longitude", () => {
  expect(groundTrackTitle("hyper3")).toBe("Vehicle latitude vs vehicle longitude");
});

it("names a geographic trajectory from altitude, latitude, and longitude", () => {
  expect(trajectoryTitle("hyper3", "geographic")).toBe(
    "Vehicle altitude vs vehicle latitude vs vehicle longitude",
  );
  expect(trajectoryTitle(null, "geographic")).toBe(
    "Vehicle altitude vs vehicle latitude vs vehicle longitude",
  );
});

it("names a local trajectory from the SBEL sentence", () => {
  expect(trajectoryTitle("aim5", "local")).toBe(
    "Missile pos. wrt point E in local level axes",
  );
});

it("uses a vector stem for a component column", () => {
  expect(seriesTitle("hyper3", "FSPV")).toBe("Specific force in V-coord vs time");
  expect(seriesTitle("hyper3", "FSPV1")).toBe("Specific force in V-coord, component 1 vs time");
  expect(seriesTitle("aim5", "SBEL1")).toBe(
    "Missile pos. wrt point E in local level axes, component 1 vs time",
  );
  expect(seriesTitle("aim5", "SBEL2")).toBe(
    "Missile pos. wrt point E in local level axes, component 2 vs time",
  );
  expect(seriesTitle("aim5", "SBEL3")).toBe(
    "Missile pos. wrt point E in local level axes, component 3 vs time",
  );
  expect(seriesTitle("hyper3", "foo1")).toBe("foo1 vs time");
  expect(seriesTitle("agm6", "STCEL1")).toBe("Position track file of target #1 vs time");
});

it("describes a vector component with the stem sentence", () => {
  expect(plotDescription("aim5", "SBEL1")).toBe(
    "Missile pos. wrt point E in local level axes, component 1 - m",
  );
  expect(plotDescription("aim5", "SBEL2")).toBe(
    "Missile pos. wrt point E in local level axes, component 2 - m",
  );
  expect(plotDescription("aim5", "anx")).toBe("Normal load factor - g's");
});

it("uses the symbol when the sentence is only whitespace", () => {
  expect(seriesTitle("hyper3", "empty")).toBe("empty vs time");
});

it("keeps a side's symbol when that side has no sentence", () => {
  const latProgram = variableHelp.hyper3.latx;
  const latShared = sharedHelp.latx;
  delete variableHelp.hyper3.latx;
  delete sharedHelp.latx;
  try {
    // Latitude's own symbol, same rule as `not_a_name vs Vehicle longitude`.
    expect(groundTrackTitle("hyper3")).toBe("latx vs Vehicle longitude");
  } finally {
    variableHelp.hyper3.latx = latProgram;
    sharedHelp.latx = latShared;
  }

  const lonProgram = variableHelp.hyper3.lonx;
  const lonShared = sharedHelp.lonx;
  delete variableHelp.hyper3.lonx;
  delete sharedHelp.lonx;
  try {
    expect(groundTrackTitle("hyper3")).toBe("Vehicle latitude vs lonx");
  } finally {
    variableHelp.hyper3.lonx = lonProgram;
    sharedHelp.lonx = lonShared;
  }
});
