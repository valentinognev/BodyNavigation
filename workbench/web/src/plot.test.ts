import { expect, it } from "vitest";
import { defaultColumns, hasGroundTrack } from "./plot";

it("prefers alt mach", () => {
  expect(defaultColumns(["time", "FSPV1", "alt", "mach", "lonx"])).toEqual(["alt", "mach"]);
});
it("falls back", () => {
  expect(defaultColumns(["time", "foo", "bar", "baz", "qux"])).toEqual(["foo", "bar", "baz", "qux"]);
});
it("ground", () => {
  expect(hasGroundTrack(["latx", "lonx", "alt"])).toBe(true);
  expect(hasGroundTrack(["alt"])).toBe(false);
});
