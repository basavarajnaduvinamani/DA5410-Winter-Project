export interface Metric {
  estimate: number | null;
  // JSON imports infer arrays; the exporter enforces exactly two ordered bounds.
  ci95: readonly number[] | null;
  image_clusters?: number;
  n?: number;
  valid_bootstrap?: number;
}
export interface Baseline {
  n: number;
  images: number;
  accuracy: Metric;
  precision: Metric;
  recall: Metric;
  f1: Metric;
  yes_ratio: Metric;
  false_yes_rate: Metric;
}
export interface Steering {
  n: number;
  strength: number;
  accuracy: Metric;
  delta_accuracy: Metric;
  false_yes_rate: Metric;
  delta_false_yes_rate: Metric;
  recall: Metric;
  retention: Metric;
  delta_recall?: Metric;
  versus_controls_false_yes?: Record<string, Metric>;
  per_split_metrics?: Record<
    string,
    { false_yes_rate: Metric; recall: Metric }
  >;
  incomplete?: string;
}
export interface Example {
  id: string;
  image: string;
  imageId: number;
  imageSrc: string;
  imageSha256: string;
  imageAlt: string;
  attribution: {
    title?: string | null;
    creator: string;
    source: string;
    license: string;
    licenseUrl: string;
  };
  split: string;
  question: string;
  truth: string;
  baseline: string;
  baselinePred: string;
  steered: string;
  steeredPred: string;
  probeScore: number | null;
  probeThreshold: number;
  gate: {
    triggered: boolean | null;
    alignment: number | null;
    threshold: number | null;
    source: string;
  };
}
export interface ResearchData {
  schemaVersion: number;
  status: "pending" | "measured";
  statusDetail: string;
  builtFrom: string[];
  summary: string[];
  protocol: {
    counts: Record<string, number>;
    quickCounts: Record<string, number>;
    seed: number;
    bootstrapIterations: number;
    target: string;
    model: string;
    revision: string;
    questionHashes: Record<string, string>;
    audit: { original: number; retained: number; excluded: number };
    source: string;
  };
  run: {
    id: string;
    profile: string;
    testImages: number;
    model: string;
    revision: string;
    target: string;
    hardware: string | null;
    runtime: string | null;
    source: string;
  } | null;
  results: {
    baseline: Record<string, Baseline>;
    detection: {
      detectors: Record<string, { auroc: Metric; auprc: Metric }>;
      comparisons?: Record<string, Metric>;
      n: number;
    } | null;
    steering: Record<string, Steering>;
  };
  calibrationCurve: {
    layer: number;
    probe: number | null;
    meanDifference: number | null;
    pca: number | null;
  }[];
  selectedLayers: Record<string, number>;
  hypotheses: { id: string; status: string; finding: string; source: string }[];
  examples: Example[];
  modern: { status: string; model: string; revision: string; source: string };
}
