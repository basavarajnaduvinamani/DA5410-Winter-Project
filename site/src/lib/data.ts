import snapshot from "@/data/research.json";
import type { ResearchData, Metric } from "./types";
export const research = snapshot as ResearchData;
export const repo =
  "https://github.com/basavarajnaduvinamani/DA5410-Winter-Project";
const release = "v2.0-final";
export const code = `${repo}/tree/${release}`;
export const colab = `https://colab.research.google.com/github/basavarajnaduvinamani/DA5410-Winter-Project/blob/${release}/notebooks/CounterPrior_Mechanistic_Steering.ipynb`;
export const sourceUrl = (path: string) =>
  `${repo}/blob/${release}/${path}`;
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
