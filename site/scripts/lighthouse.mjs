import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { chromium } from "playwright";
const base = process.env.BASE_URL || "http://127.0.0.1:4173";
mkdirSync("qa/lighthouse", { recursive: true });
const reports = [];
for (const [name, route] of [
  ["home", "/"],
  ["explorer", "/explorer/"],
  ["ablations", "/ablations/"],
  ["about", "/about/"],
]) {
  const output = `qa/lighthouse/${name}.json`;
  const result = spawnSync(
    process.execPath,
    [
      path.resolve("node_modules/lighthouse/cli/index.js"),
      base + route,
      "--quiet",
      "--output=json",
      `--output-path=${output}`,
      "--only-categories=accessibility,best-practices",
      "--chrome-flags=--headless --no-sandbox --disable-gpu",
    ],
    {
      stdio: "inherit",
      env: { ...process.env, CHROME_PATH: chromium.executablePath() },
    },
  );
  if (result.status !== 0) process.exit(result.status || 1);
  const r = JSON.parse(readFileSync(output, "utf8"));
  const scores = Object.fromEntries(
    Object.entries(r.categories).map(([k, v]) => [k, v.score * 100]),
  );
  reports.push({ page: route, url: r.finalDisplayedUrl, scores });
}
writeFileSync(
  "qa/lighthouse-summary.json",
  JSON.stringify(reports, null, 2) + "\n",
);
if (
  reports.some(
    (r) => r.scores.accessibility < 90 || r.scores["best-practices"] < 90,
  )
)
  process.exit(1);
console.log(JSON.stringify(reports, null, 2));
