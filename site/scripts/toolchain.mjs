import { readFileSync, writeFileSync } from "node:fs";
const pkg = JSON.parse(readFileSync("package.json", "utf8"));
const installed = Object.fromEntries(
  Object.keys({ ...pkg.dependencies, ...pkg.devDependencies }).map((name) => [
    name,
    JSON.parse(readFileSync(`node_modules/${name}/package.json`, "utf8"))
      .version,
  ]),
);
writeFileSync(
  "qa/toolchain.json",
  JSON.stringify(
    {
      recordedAt: new Date().toISOString(),
      node: process.version,
      platform: process.platform,
      npmUserAgent: process.env.npm_config_user_agent ?? null,
      source:
        "Installed package manifests; exact dependency graph is pinned in package-lock.json",
      installed,
    },
    null,
    2,
  ) + "\n",
);
