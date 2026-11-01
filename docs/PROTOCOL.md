### FILE: PROTOCOL.md

status: prepared CPU-verified protocol; full LLaVA inference and clean GPU installation remain UNKNOWN. [results/2026-10-30]
round 2 (2026-10-31): POPE is the PRIMARY benchmark; the custom set below is a secondary small-scale probe set. See 'POPE primary protocol' at the end. [scripts/run_pope.py]
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

## POPE primary protocol (round 2, 2026-10-31)
benchmark: POPE random/popular/adversarial, 3,000 questions each, same 500 COCO val2014 images; question files vendored at AoiDragon/POPE@08d957b with sha256 pins. [dataset/pope/sources.json]
images: COCO val2014, fetched per image or from val2014.zip; fetch_pope writes a sha256 manifest of the bytes used. [scripts/fetch_pope.py]
command: bash scripts/colab_run.sh setup|timing|run|report (notebook cells 3-6); resumable from RUN/predictions.jsonl and RUN/states. [scripts/colab_run.sh]
prompt: 'USER: <image>\n{question}\nAnswer the question using a single word or phrase. ASSISTANT:'; greedy, max_new_tokens 3. [src/adapters.py]
scoring: official POPE parser (first sentence; 'No'/'no'/'not' => no); accuracy, precision, recall, F1 (yes positive), yes-ratio, false-yes rate on label=no. [src/pope.py]
split: image-disjoint 20/10/70% calibration/validation/test by sha256(seed:image); seed 20261030; 100/50/350 images; IDs committed. quick profile takes the first 40/20/100 in the same order. [dataset/pope/splits_seed20261030.json]
detection target: baseline answer wrong (config detection.target='error'; 'false_yes' restricts to object hallucination). [src/pope_pipeline.py]
detection selection: per layer, 5-fold GroupKFold (groups = images) out-of-fold AUROC on calibration only for (a) standardised logistic probe C=1, (b) mean-difference cosine projection, (c) class-balanced PCA-axis projection; each detector takes its best calibration layer (ties to earlier). Logit baselines (-log p, entropy of first answer token) are not fitted. [src/detection.py]
held-out: validation/test scores saved per item; per-layer test AUROC curve is descriptive only and never used for selection. [scripts/make_report.py]
steering: layer = probe layer; directions fitted on calibration states at that layer: PCA (balanced) for SLERP, mean difference for addition, Gaussian random and shuffled-label mean difference as controls (a shuffled-label PCA axis would equal the real one up to sign, since PC1 does not use labels); cosine gate threshold per direction maximises calibration balanced accuracy. [src/pope_pipeline.py]
strength: SLERP alpha in {0.05,0.1,0.2,0.3}; addition beta in {0.05,0.1,0.2,0.3} times mean calibration norm; chosen by validation accuracy (ties to smaller); controls reuse the SLERP alpha. Strength 0 is not a candidate, so a steered test row always exists; validation_sweep.json records whether it beat baseline. [configs/pope_llava.json]
test: each method answers every test question once with frozen settings; utility = recall on present objects and retention of baseline-correct answers. [src/pope_pipeline.py]
statistics: 95% percentile image-cluster bootstrap (1,000); paired deltas resample identical images for both arms; conditional on frozen selection. [src/evaluation.py]
report: scripts/make_report.py reads only saved run files and writes results/REPORT.md, results/REPORT_metrics.json, results/figures/*.png and the README block between RESULTS markers. [scripts/make_report.py]
manual review: POPE is scored automatically; the blinded manual review pause applies only to the custom set. [scripts/run_experiment.py]
cross-model: --adapter qwen2_5_vl | internvl selects chat-template adapters (stub-tested only; revision unpinned until verified). [src/adapters.py]


## Analysis declarations for the research site
Declared 2026-11-01 before inspecting pretrained-model outcomes. The owner's first attempt stopped in setup before inference. These declarations do not retroactively turn historical experiments into preregistered runs.

- Endpoint: the current `configs/pope_llava.json` detector target is `error` (any wrong baseline answer), not exclusively a hallucinated yes. Preserve it for the active run and label it on every result. False-yes rate on absent objects is the steering endpoint.
- Detection: compare the frozen probe with answer-token negative log-probability using a paired image-cluster bootstrap of the AUROC difference. Supported improvement requires a wholly positive interval; wholly negative is not supported; an interval including zero or insufficient class support is inconclusive. Report entropy and all projection baselines alongside it. Do not select a layer from the descriptive test-layer curve.
- Geometry: an oriented PCA projection is associated with the configured error target only if its held-out AUROC interval is wholly above chance. Below chance is not supported; a crossing/undefined interval is inconclusive. This does not establish a universal truth direction or a causal truth variable.
- Steering: a reduction requires the paired false-yes delta interval for SLERP minus baseline to be wholly below zero. Report recall and paired retention alongside it, with no post-hoc utility cutoff. A reduction with recall loss is a qualified trade-off, never an unqualified safety improvement. Compare both random-direction and shuffled-label controls using paired differences; if they perform comparably, the geometric mechanism remains inconclusive.
- A quick-profile result is exploratory and labelled with its actual held-out image count next to every table and chart. Report all executed profiles/runs and never choose the better test result as the headline.

## Optional modern-model replication: registered before execution
Model: `Qwen/Qwen2.5-VL-7B-Instruct`, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`; official model metadata reports Apache-2.0. Source: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct . Configuration: `configs/pope_qwen_replication.json`.
Protocol: baseline and detection only, same POPE files, official scorer, frozen image splits and seed; group-CV layer choice on calibration; use `--stages baseline,calibrate,report`. No validation strength sweep or test steering. Use a separate Drive run folder. Begin only after the main LLaVA run and only on spare GPU time. T4 suitability is to be verified, not claimed from stub tests. GPU status: not executed.
Command: `.venv-colab/bin/python -m scripts.run_pope --config configs/pope_qwen_replication.json --profile quick --stages baseline,calibrate,report --output /content/drive/MyDrive/counterprior/qwen-quick --coco-dir /content/drive/MyDrive/counterprior/coco_val2014`.
The adapter remains CPU/stub-verified only until this command succeeds on the actual checkpoint. A separate generated report must identify the model; never pool its questions or headline with LLaVA.
