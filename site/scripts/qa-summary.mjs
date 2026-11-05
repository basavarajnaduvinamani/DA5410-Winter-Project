import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import crypto from "node:crypto";
const read = (p) => JSON.parse(readFileSync(p, "utf8"));
const tests = read("qa/playwright-results.json");
const lighthouse = read("qa/lighthouse-summary.json");
if (
  tests.stats.unexpected ||
  tests.stats.skipped ||
  tests.stats.flaky ||
  !tests.stats.expected
)
  throw new Error(
    "QA summary requires every browser check to pass without skips or retries",
  );
if (
  lighthouse.length !== 4 ||
  lighthouse.some(
    (p) => p.scores.accessibility < 90 || p.scores["best-practices"] < 90,
  )
)
  throw new Error("Mobile Lighthouse gate did not pass for every page");
const checks = [];
function walk(suite) {
  for (const spec of suite.specs ?? [])
    for (const t of spec.tests ?? [])
      checks.push({
        name: spec.title,
        browser: t.projectName,
        status: t.status,
      });
  for (const child of suite.suites ?? []) walk(child);
}
for (const suite of tests.suites) walk(suite);
const screenshots = Object.fromEntries(
  readdirSync("qa/screenshots")
    .filter((p) => p.endsWith(".png"))
    .sort()
    .map((p) => [
      p,
      crypto
        .createHash("sha256")
        .update(readFileSync(`qa/screenshots/${p}`))
        .digest("hex"),
    ]),
);
const report = {
  scope: "Software QA, not pretrained-model evaluation",
  recordedAt: new Date().toISOString(),
  playwright: { ...tests.stats, checks },
  accessibility: {
    scans: checks.filter((t) => t.name.startsWith("QA ")).length,
    violations: 0,
    assertion: "Every axe violation is a test failure",
  },
  lighthouse,
  screenshots,
  dependencies: read("package.json"),
};
writeFileSync("qa/verification.json", JSON.stringify(report, null, 2) + "\n");
console.log(
  `Verified ${tests.stats.expected} browser checks and ${Object.keys(screenshots).length} screenshots.`,
);
