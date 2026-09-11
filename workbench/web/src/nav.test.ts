import { expect, it } from "vitest";
import { runButtonDisabled, saveButtonDisabled } from "./nav";

it("disables Save and Run when program is null", () => {
  expect(saveButtonDisabled(null)).toBe(true);
  expect(runButtonDisabled(null, false, null)).toBe(true);
  expect(saveButtonDisabled("hyper3")).toBe(false);
  expect(runButtonDisabled(null, false, "hyper3")).toBe(false);
});
