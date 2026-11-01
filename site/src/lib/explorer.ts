import type { Example } from "./types";
export function filterExamples(
  examples: Example[],
  split: string,
  filter: string,
) {
  return examples.filter(
    (e) =>
      (split === "all" || e.split === split) &&
      (filter === "all" ||
        (filter === "hallucinated" &&
          e.truth === "no" &&
          e.baselinePred === "yes") ||
        (filter === "fixed" &&
          e.baselinePred !== e.truth &&
          e.steeredPred === e.truth) ||
        (filter === "broken" &&
          e.baselinePred === e.truth &&
          e.steeredPred !== e.truth)),
  );
}
