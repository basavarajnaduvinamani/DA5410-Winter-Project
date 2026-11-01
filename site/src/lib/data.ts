import snapshot from "@/data/research.json";
import type { ResearchData, Metric } from "./types";
export const research = snapshot as ResearchData;
export const repo =
  "https://github.com/basavarajnaduvinamani/DA5410-Winter-Project";
export const code = `${repo}/tree/claude/site-2026-11`;
export const colab = `https://colab.research.google.com/github/basavarajnaduvinamani/DA5410-Winter-Project/blob/claude/site-2026-11/notebooks/CounterPrior_Mechanistic_Steering.ipynb`;
export const sourceUrl = (path: string) =>
  `${repo}/blob/claude/site-2026-11/${path}`;
export function value(n: number | null | undefined, digits = 3) {
  return n == null ? "Not available" : n.toFixed(digits);
}
export function metric(m: Metric | undefined) {
  return !m || m.estimate == null
    ? "Not available"
    : `${value(m.estimate)}${m.ci95 ? ` [${value(m.ci95[0])}, ${value(m.ci95[1])}]` : " [CI unavailable]"}`;
}
export function profileLabel() {
  return research.run
    ? `${research.run.profile} profile: ${research.run.testImages} test images`
    : "No verified GPU run";
}
