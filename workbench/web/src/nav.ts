export type NavId =
  | "overview"
  | "modules"
  | "vehicles"
  | "timing"
  | "events"
  | "decks"
  | "results";

export const NAV_ITEMS: { id: NavId; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "modules", label: "Modules" },
  { id: "vehicles", label: "Vehicles" },
  { id: "timing", label: "Timing" },
  { id: "events", label: "Events" },
  { id: "decks", label: "Decks" },
  { id: "results", label: "Results" },
];

export type ParseErrorInfo = {
  message: string;
};

export function saveButtonDisabled(program: string | null): boolean {
  return program == null;
}

export function runButtonDisabled(
  parseError: ParseErrorInfo | null,
  runInFlight: boolean,
  program: string | null,
): boolean {
  return parseError != null || runInFlight || program == null;
}
