import type { Decision } from "./types";

export const CATEGORY_COLOR: Record<string, string> = {
  ripe: "var(--ripe)",
  unripe: "var(--unripe)",
  uncertain: "var(--uncertain)",
};

export const DECISION_COLOR: Record<Decision, string> = {
  ACCEPT: "var(--accept)",
  REJECT: "var(--reject)",
  REVIEW: "var(--review)",
};

/** Mirrors backend grading.category(): "unripe" must be checked before "ripe". */
export function categoryOf(label: string): string {
  const n = label.toLowerCase();
  if (n === "uncertain") return "uncertain";
  if (n.includes("unripe")) return "unripe";
  if (n.includes("ripe")) return "ripe";
  return n;
}

export const colorOf = (label: string) => CATEGORY_COLOR[categoryOf(label)] ?? "var(--uncertain)";

export const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export function clockTime(iso: string) {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
