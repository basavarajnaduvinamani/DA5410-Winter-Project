export function formatScore(score: number): string {
  return score > 0 && score < 0.001 ? "< 0.001" : score.toFixed(3);
}
