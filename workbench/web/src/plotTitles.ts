import { variableDescription } from "./forms/fieldHelp";

const UNIT_MARK = " - ";
const COMPONENT = /^(.+)([123])$/;

function definedSentence(program: string | null | undefined, name: string): string | undefined {
  const sentence = variableDescription(program, name);
  if (sentence == null || sentence.trim() === "") return undefined;
  return sentence.trim();
}

function withComponent(sentence: string, index: string): string {
  const clause = `, component ${index}`;
  const at = sentence.lastIndexOf(UNIT_MARK);
  if (at === -1) return `${sentence}${clause}`;
  return `${sentence.slice(0, at).trim()}${clause}${sentence.slice(at)}`;
}

export function plotDescription(
  program: string | null | undefined,
  name: string,
): string | undefined {
  const direct = definedSentence(program, name);
  if (direct) return direct;
  const match = COMPONENT.exec(name);
  if (match == null) return undefined;
  const stem = definedSentence(program, match[1]);
  if (stem == null) return undefined;
  return withComponent(stem, match[2]);
}

function columnPhrase(
  program: string | null | undefined,
  name: string,
): { text: string; prose: boolean } {
  const sentence = plotDescription(program, name);
  if (sentence == null) return { text: name, prose: false };
  const trimmed = sentence.trim();
  const at = trimmed.lastIndexOf(UNIT_MARK);
  const stem = (at === -1 ? trimmed : trimmed.slice(0, at)).trim();
  if (stem === "") return { text: name, prose: false };
  return { text: stem, prose: true };
}

function lowerFirst(text: string): string {
  return text.slice(0, 1).toLowerCase() + text.slice(1);
}

export function seriesTitle(program: string | null | undefined, yName: string): string {
  return `${columnPhrase(program, yName).text} vs time`;
}

export function groundTrackTitle(program: string | null | undefined): string {
  const lat = columnPhrase(program, "latx");
  const lon = columnPhrase(program, "lonx");
  const lonText = lat.prose && lon.prose ? lowerFirst(lon.text) : lon.text;
  return `${lat.text} vs ${lonText}`;
}

export function trajectoryTitle(
  program: string | null | undefined,
  kind: "geographic" | "local",
): string {
  if (kind === "local") {
    const phrase = columnPhrase(program, "SBEL");
    return phrase.prose ? phrase.text : "-SBEL3 vs SBEL1 vs SBEL2";
  }
  const alt = columnPhrase(program, "alt");
  const lat = columnPhrase(program, "latx");
  const lon = columnPhrase(program, "lonx");
  const latText = alt.prose && lat.prose ? lowerFirst(lat.text) : lat.text;
  const lonText = alt.prose && lon.prose ? lowerFirst(lon.text) : lon.text;
  return `${alt.text} vs ${latText} vs ${lonText}`;
}
