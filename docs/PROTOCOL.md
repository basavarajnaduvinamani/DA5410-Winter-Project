### FILE: PROTOCOL.md

status: prepared CPU-verified protocol; full LLaVA inference and clean GPU installation remain UNKNOWN. [results/2026-10-30]
Colab: https://colab.research.google.com/github/basavarajnaduvinamani/DA5410-Winter-Project/blob/upgrade/2026-10-30/notebooks/CounterPrior_Mechanistic_Steering.ipynb [notebooks/CounterPrior_Mechanistic_Steering.ipynb]

## Execution
runtime: select a Colab GPU; launcher uses uv 0.9.7 to provision Python 3.11.9 and an isolated environment, syncing the Linux transitive dependency lock with required hashes; uv pip freeze records installed versions. [requirements-colab.txt] [requirements-colab.lock] [scripts/colab_run.sh]
command from repository root: bash scripts/colab_run.sh [scripts/colab_run.sh]
config: configs/llava.json; immutable model revision, seed, generation budget and bootstrap settings are recorded in every prediction. [configs/llava.json]
raw artifacts: output directory contains identity.json, environment.json, environment-freeze.txt, splits.json, per-case raw decoder states, predictions.jsonl, review exports, selection.json and errors.log on failure. Previous predictions are never overwritten. [scripts/run_experiment.py]
calibration review: first pass collects baseline answers then pauses; fill blind_review_<count>.jsonl, using full-resolution source images; a separate operator retains blind_key.private.json. Reviewers should not see method, split or automatic labels. [scripts/blind_review.py]
mapping command: .venv-colab/bin/python -m scripts.blind_review --filled FILLED_JSONL --key RUN/blind_key.private.json --output RUN/reviews.jsonl [scripts/blind_review.py]
resume command: .venv-colab/bin/python -m scripts.run_experiment --config configs/llava.json --output RUN --reviews RUN/reviews.jsonl [scripts/run_experiment.py]
final review: review all generated answers, including every steered/ablation answer; preserve earlier reviews; combine operator-mapped JSONL rows without duplicate prediction IDs. [scripts/analyze_run.py]
analysis command: .venv-colab/bin/python -m scripts.analyze_run --run RUN --reviews ALL_REVIEWS_JSONL --output RUN/metrics.json [scripts/analyze_run.py]
archive: download the entire RUN directory before ending Colab; no automatic push from Colab. [scripts/colab_run.sh]

## Frozen methodology
split unit: decoded RGB pixels including dimensions, SHA256 grouped; duplicate images and all prompts for an image stay in one partition. Splits use only image identities/category and a fixed seed, before inference. [src/protocol.py]
stratification: each category combination is split by image group; sparse strata are assigned to calibration; unavailable held-out categories have null estimates. [src/protocol.py]
probe target: actual baseline hallucination after human adjudication, not an instruction to say yes/no; incorrect positive-control answers are not automatically hallucinations. [scripts/run_experiment.py]
selection: layer chosen by calibration-only leave-one-image-group-out probe AUROC, then gate balanced accuracy; direction is PCA on raw terminal states oriented toward non-hallucinated answers. Gate threshold maximizes calibration out-of-fold balanced accuracy on cosine alignment. Ties prefer earlier layers. [src/protocol.py]
selection caution: calibration scores are selection diagnostics, not unbiased generalization estimates. No validation/test labels are used to choose layer, threshold, alpha, probe or axis. [src/protocol.py]
failure: if calibration training folds lack both classes, stop and acquire more audited calibration images. No forced labels, test reuse or fabricated scores. [src/protocol.py]
probe: logistic regression C=1, probability cutoff 0.5; frozen after calibration; positive class is hallucination. [src/protocol.py]
steering: alpha fixed in config; cosine(h,v) threshold in [-1,1]; intervene on last position per batch member at prefill and every decoding step; no whole-sequence mutation or batch-mean gate. [src/rotational_steering.py]
layers: zero-based raw decoder block outputs; embeddings excluded; reported historical Layer 17/25 cannot be equated without extraction provenance. [src/probe.py]
logit lens: apply final decoder RMS normalization exactly once before output head; require single-token candidate strings. [src/probe.py]
statistics: report validation and test separately; image-cluster percentile bootstrap, paired baseline/steered resampling; interval conditional on the frozen calibration fit, not refitting on test. Single-class AUROC and empty categories are null; valid bootstrap count and image count accompany each interval. [src/protocol.py] [scripts/analyze_run.py]
utility: per-category accuracy and hallucination rate; positive-control accuracy and paired retention among baseline-correct controls; hallucination delta is steered minus baseline, so negative favors steering. [scripts/analyze_run.py]
limitations: very small provisional benchmark; one retained counter-attribute image provides no held-out attribute estimate; manual all-answer adjudication and independent label review required. CIs do not imply population validity. [dataset/audit_2026-10-30.json]

## Optional matched ablations
command: add --ablations to the resume command; fixed selected layer, gate threshold, alpha and split; random direction, shuffled calibration labels, mean difference, vector addition and constant SLERP. [scripts/run_experiment.py]
interpretation: ablations reuse the primary method's gate instead of optimizing per direction; vector addition uses unit-vector strength alpha while SLERP uses angular fraction alpha, so these doses are not equal in geometric magnitude. [scripts/run_experiment.py]
newer model: UNKNOWN; this runner supports LLaVA only. Qwen/InternVL require a separately verified architecture adapter and GPU run. [src/model_loader.py]
primary implementation references: https://huggingface.co/llava-hf/llava-1.5-7b-hf ; https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/llava [LIVE retrieved 2026-10-30]
