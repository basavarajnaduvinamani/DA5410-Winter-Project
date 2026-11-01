import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import crypto from "node:crypto";
const defaultRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const json = (p) => JSON.parse(readFileSync(p, "utf8"));
export const sha = (p) =>
  crypto.createHash("sha256").update(readFileSync(p)).digest("hex");
export function validateExample(e) {
  if (
    !e ||
    !Number.isSafeInteger(e.imageId) ||
    !e.imageSrc?.startsWith("/examples/") ||
    e.imageSrc.includes("..")
  )
    throw new Error("Example image must be a local licensed COCO asset");
  if (
    !e.attribution?.creator ||
    !e.attribution?.license ||
    !/^https:\/\//.test(e.attribution?.source || "") ||
    !/^https:\/\//.test(e.attribution?.licenseUrl || "")
  )
    throw new Error("Incomplete attribution; exclude this image");
  if (
    !["random", "popular", "adversarial"].includes(e.split) ||
    !["yes", "no"].includes(e.truth)
  )
    throw new Error("Invalid POPE example");
  if (
    e.probeScore !== null &&
    (!Number.isFinite(e.probeScore) || e.probeScore < 0 || e.probeScore > 1)
  )
    throw new Error("Invalid probe probability");
  if (![true, false, null].includes(e.gate?.triggered))
    throw new Error("Unrecorded gates must stay unknown");
  return e;
}
export function validateData(data) {
  const finite = (v) => {
    if (typeof v === "number" && !Number.isFinite(v))
      throw new Error("Non-finite data is not publishable");
    if (v && typeof v === "object") {
      if (
        "ci95" in v &&
        v.ci95 !== null &&
        (!Array.isArray(v.ci95) ||
          v.ci95.length !== 2 ||
          v.ci95.some((n) => !Number.isFinite(n)) ||
          v.ci95[0] > v.ci95[1])
      )
        throw new Error(
          "Confidence intervals require two finite, ordered bounds",
        );
      Object.values(v).forEach(finite);
    }
  };
  finite(data);
  if (
    !["pending", "measured"].includes(data.status) ||
    !data.protocol?.counts ||
    data.schemaVersion !== 1
  )
    throw new Error("Invalid research export");
  if (data.examples.length > 24)
    throw new Error("At most 24 licensed example images may be published");
  data.examples.forEach(validateExample);
  if (
    data.status === "pending" &&
    (data.run !== null ||
      data.examples.length ||
      Object.keys(data.results.baseline).length ||
      data.calibrationCurve.length)
  )
    throw new Error("Pending exports must not contain fabricated results");
  if (
    data.status === "measured" &&
    (!data.run?.id ||
      !Number.isInteger(data.run.testImages) ||
      data.run.testImages <= 0 ||
      !data.run.model ||
      !data.run.revision ||
      !["quick", "full"].includes(data.run.profile) ||
      !["error", "false_yes"].includes(data.run.target))
  )
    throw new Error("Measured data requires a verified run and sample size");
  return data;
}
export function buildResearch(root = defaultRoot) {
  const read = (name) => json(path.join(root, name));
  const config = read("configs/pope_llava.json");
  const splits = read("dataset/pope/splits_seed20261030.json");
  const sources = read("dataset/pope/sources.json");
  const audit = read("dataset/audit_2026-10-30.json");
  const retained = read("dataset/counterprior_corrected.json");
  const modern = read("configs/pope_qwen_replication.json");
  const descriptorPath = path.join(root, "results/published_run.json");
  const metricsPath = path.join(root, "results/REPORT_metrics.json");
  const descriptor = existsSync(descriptorPath) ? json(descriptorPath) : null;
  let metrics = null;
  if (descriptor) {
    if (
      descriptor.status !== "verified" ||
      !existsSync(metricsPath) ||
      sha(metricsPath) !== descriptor.metrics_sha256
    )
      throw new Error(
        "Published metrics do not match the verified run manifest",
      );
    metrics = json(metricsPath);
    if (metrics.run_id !== descriptor.run_id)
      throw new Error("Run identity mismatch");
  } else if (existsSync(metricsPath))
    throw new Error(
      "Unverified result file: run scripts/ingest_run.py before publishing",
    );
  const counts = Object.fromEntries(
    ["calibration", "validation", "test"].map((k) => [
      k,
      splits.counts?.[k] ?? splits[k]?.length,
    ]),
  );
  const pendingReason = existsSync(
    path.join(root, "results/2026-11-01/gpu_status.json"),
  )
    ? read("results/2026-11-01/gpu_status.json").status
    : "not_run";
  const data = {
    schemaVersion: 1,
    status: metrics ? "measured" : "pending",
    statusDetail: metrics
      ? "Generated from a verified saved run."
      : pendingReason === "setup_failed_before_inference"
        ? "The Colab setup stopped before inference. A headless plotting fix is available; model results remain unverified."
        : "The protocol is frozen. No pretrained-model run has been verified.",
    builtFrom: [
      "configs/pope_llava.json",
      "configs/pope_qwen_replication.json",
      "dataset/pope/splits_seed20261030.json",
      "dataset/pope/sources.json",
      "dataset/audit_2026-10-30.json",
      "dataset/counterprior_corrected.json",
      ...(metrics
        ? ["results/REPORT_metrics.json", "results/published_run.json"]
        : []),
    ],
    protocol: {
      counts,
      quickCounts: config.profiles.quick.max_images,
      seed: config.seed,
      bootstrapIterations: config.bootstrap_iterations,
      target: config.detection.target,
      model: config.model_id,
      revision: config.revision,
      questionHashes: Object.fromEntries(
        Object.entries(sources.files).map(([k, v]) => [k, v.sha256]),
      ),
      audit: {
        original: audit.length,
        retained: retained.length,
        excluded: audit.length - retained.length,
      },
      source: "dataset/pope/splits_seed20261030.json",
    },
    run: descriptor
      ? {
          id: descriptor.run_id,
          profile: descriptor.profile,
          testImages: descriptor.test_images,
          model: descriptor.model_id,
          revision: descriptor.revision,
          target: descriptor.target,
          hardware: descriptor.hardware ?? null,
          runtime: descriptor.runtime ?? null,
          source: "results/published_run.json",
        }
      : null,
    summary: metrics?.summary ?? [],
    results: {
      baseline: metrics?.baseline ?? {},
      detection: metrics?.detection ?? null,
      steering: metrics?.steering ?? {},
    },
    calibrationCurve: metrics?.provenance?.calibration_curve ?? [],
    selectedLayers: metrics?.provenance?.selected_layers ?? {},
    hypotheses: metrics?.hypotheses ?? [
      {
        id: "detection",
        status: "inconclusive",
        finding:
          "No verified run. Error detection cannot yet be compared with answer-token confidence.",
        source: "results/REPORT.md",
      },
      {
        id: "geometry",
        status: "inconclusive",
        finding:
          "No held-out evidence yet for an oriented PCA projection. A separable plot would not establish a universal truth axis.",
        source: "docs/PROTOCOL.md",
      },
      {
        id: "steering",
        status: "inconclusive",
        finding:
          "False-yes reduction and recognition retention have not been measured under this protocol.",
        source: "results/REPORT.md",
      },
    ],
    examples: [],
    modern: {
      status: "Not executed",
      model: modern.model_id,
      revision: modern.revision,
      source: "configs/pope_qwen_replication.json",
    },
  };
  const examplesPath = path.join(root, "results/examples_manifest.json");
  if (metrics && existsSync(examplesPath)) {
    const m = json(examplesPath);
    if (m.run_id !== data.run.id)
      throw new Error("Example manifest belongs to another run");
    data.builtFrom.push("results/examples_manifest.json");
    data.examples = m.examples.map(validateExample);
    for (const e of data.examples) {
      const asset = path.join(root, "site/public", e.imageSrc);
      if (!existsSync(asset) || sha(asset) !== e.imageSha256)
        throw new Error(
          "Missing or changed licensed image asset; run npm run prepare:examples",
        );
    }
  }
  return validateData(data);
}
export function exportResearch(root = defaultRoot) {
  const data = buildResearch(root);
  const output = path.join(root, "site/src/data/research.json");
  mkdirSync(path.dirname(output), { recursive: true });
  writeFileSync(output, JSON.stringify(data, null, 2) + "\n");
  const pub = path.join(root, "site/public/research.json");
  mkdirSync(path.dirname(pub), { recursive: true });
  writeFileSync(pub, JSON.stringify(data, null, 2) + "\n");
  writeFileSync(
    path.join(root, "site/public/evidence-manifest.json"),
    JSON.stringify(
      {
        schemaVersion: 1,
        exportSha256: sha(pub),
        sources: Object.fromEntries(
          data.builtFrom.map((p) => [p, sha(path.join(root, p))]),
        ),
        scope:
          "Hashes identify the inputs used for this public export; they do not verify unseen model weights or image bytes.",
      },
      null,
      2,
    ) + "\n",
  );
  console.log(
    `Exported ${data.status} evidence; ${data.examples.length} licensed examples. Sources: ${data.builtFrom.join(", ")}`,
  );
  return data;
}
if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  if (existsSync(path.join(defaultRoot, "configs/pope_llava.json")))
    exportResearch();
  else {
    const site = path.resolve(
      path.dirname(fileURLToPath(import.meta.url)),
      "..",
    );
    const pub = path.join(site, "public/research.json");
    const manifest = json(path.join(site, "public/evidence-manifest.json"));
    if (
      manifest.exportSha256 !== sha(pub) ||
      sha(path.join(site, "src/data/research.json")) !== sha(pub)
    )
      throw new Error(
        "Packaged evidence hash mismatch; regenerate from the repository root",
      );
    validateData(json(pub));
    console.log(
      "Validated the packaged public evidence export and source manifest.",
    );
  }
}
