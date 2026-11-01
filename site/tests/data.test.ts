import { describe, it, expect } from "vitest";
import {
  readFileSync,
  mkdtempSync,
  mkdirSync,
  copyFileSync,
  writeFileSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import {
  buildResearch,
  validateData,
  validateExample,
} from "../scripts/export-data.mjs";
import { filterExamples } from "../src/lib/explorer";
import type { Example } from "../src/lib/types";
const base: Example = {
  id: "synthetic-unit-fixture",
  image: "COCO_val2014_000000000042.jpg",
  imageId: 42,
  imageSrc: "/examples/COCO_val2014_000000000042.jpg",
  imageSha256: "0".repeat(64),
  imageAlt: "Synthetic metadata fixture; never published",
  attribution: {
    creator: "Fixture author",
    source: "https://www.flickr.com/photos/fixture/42",
    license: "CC BY 4.0",
    licenseUrl: "https://creativecommons.org/licenses/by/4.0/",
  },
  split: "random",
  question: "Synthetic fixture question?",
  truth: "no",
  baseline: "yes",
  baselinePred: "yes",
  steered: "no",
  steeredPred: "no",
  probeScore: 0.7,
  probeThreshold: 0.5,
  gate: {
    triggered: null,
    alignment: null,
    threshold: null,
    source: "not recorded",
  },
};
describe("standalone deployment evidence", () => {
  function packaged(tamper: boolean) {
    const temp = mkdtempSync(path.join(tmpdir(), "counterprior-export-"));
    try {
      const site = path.join(temp, "site");
      for (const file of [
        "scripts/export-data.mjs",
        "public/research.json",
        "public/evidence-manifest.json",
        "src/data/research.json",
      ]) {
        const target = path.join(site, file);
        mkdirSync(path.dirname(target), { recursive: true });
        copyFileSync(file, target);
      }
      if (tamper) writeFileSync(path.join(site, "public/research.json"), "{}");
      return spawnSync(
        process.execPath,
        [path.join(site, "scripts/export-data.mjs")],
        { encoding: "utf8" },
      );
    } finally {
      rmSync(temp, { recursive: true, force: true });
    }
  }
  it("builds from the public package without parent research files", () => {
    const r = packaged(false);
    expect(r.stderr).toBe("");
    expect(r.status).toBe(0);
    expect(r.stdout).toContain("Validated the packaged public evidence");
  });
  it("rejects a changed packaged export", () => {
    const r = packaged(true);
    expect(r.status).not.toBe(0);
    expect(r.stderr).toContain("evidence hash mismatch");
  });
});
describe("public evidence contract", () => {
  it("rejects malformed confidence intervals before publishing", () => {
    const data = {
      ...structuredClone(buildResearch()),
      extra: { ci95: [0, 1, 2] },
    };
    expect(() => validateData(data)).toThrow(/two finite/);
    data.extra = { ci95: [1, 0] };
    expect(() => validateData(data)).toThrow(/ordered/);
  });
  it("matches the actual disjoint image manifest", () => {
    const data = buildResearch();
    const split = JSON.parse(
      readFileSync("../dataset/pope/splits_seed20261030.json", "utf8"),
    );
    const all = ["calibration", "validation", "test"].flatMap((k) => split[k]);
    expect(new Set(all).size).toBe(all.length);
    for (const k of ["calibration", "validation", "test"])
      expect(data.protocol.counts[k]).toBe(split[k].length);
  });
  it("never fills a missing run with demonstration answers or charts", () => {
    const data = buildResearch();
    if (data.status === "pending") {
      expect(data.run).toBeNull();
      expect(data.examples).toEqual([]);
      expect(data.calibrationCurve).toEqual([]);
      expect(data.results.baseline).toEqual({});
    }
    expect(JSON.stringify(data)).not.toContain("VERCEL_OIDC_TOKEN");
    expect(JSON.stringify(data)).not.toContain("Bearer ");
  });
  it("rejects fabricated pending results and non-finite numbers", () => {
    const data = structuredClone(buildResearch());
    data.status = "pending";
    data.run = null;
    data.examples = [base];
    expect(() => validateData(data)).toThrow(/Pending/);
    const other = structuredClone(buildResearch());
    other.protocol.seed = NaN;
    expect(() => validateData(other)).toThrow(/Non-finite/);
  });
  it("requires local image assets and complete redistribution attribution", () => {
    expect(validateExample(base)).toEqual(base);
    expect(() =>
      validateExample({ ...base, imageSrc: "https://example.com/image.jpg" }),
    ).toThrow(/local/);
    expect(() =>
      validateExample({ ...base, imageSrc: "/examples/../secret" }),
    ).toThrow(/local/);
    expect(() =>
      validateExample({
        ...base,
        attribution: { ...base.attribution, creator: "" },
      }),
    ).toThrow(/attribution/);
  });
  it("does not infer an unrecorded gate from the probe score", () => {
    expect(validateExample(base).gate.triggered).toBeNull();
    expect(() =>
      validateExample({
        ...base,
        gate: { ...base.gate, triggered: "unknown" },
      }),
    ).toThrow(/unknown/);
  });
  it("caps the public example set", () => {
    const data = structuredClone(buildResearch());
    data.examples = Array.from({ length: 25 }, () => base);
    expect(() => validateData(data)).toThrow(/24/);
  });
});
describe("explorer outcomes (synthetic fixtures only)", () => {
  const fixed = base;
  const broken = {
    ...base,
    id: "broken-fixture",
    split: "popular",
    truth: "yes",
    baselinePred: "yes",
    steeredPred: "no",
  };
  const unchanged = {
    ...base,
    id: "unchanged-fixture",
    split: "adversarial",
    truth: "no",
    baselinePred: "no",
    steeredPred: "no",
  };
  const examples = [fixed, broken, unchanged];
  it("selects baseline hallucinations, fixes and regressions separately", () => {
    expect(filterExamples(examples, "all", "hallucinated")).toEqual([fixed]);
    expect(filterExamples(examples, "all", "fixed")).toEqual([fixed]);
    expect(filterExamples(examples, "all", "broken")).toEqual([broken]);
  });
  it("intersects split and outcome without mutating the saved set", () => {
    expect(filterExamples(examples, "popular", "fixed")).toEqual([]);
    expect(filterExamples(examples, "popular", "broken")).toEqual([broken]);
    expect(examples).toHaveLength(3);
  });
  it("handles the honest empty production state", () =>
    expect(filterExamples([], "random", "fixed")).toEqual([]));
});
