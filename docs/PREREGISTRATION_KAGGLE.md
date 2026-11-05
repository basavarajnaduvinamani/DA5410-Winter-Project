# CounterPrior v2: private Kaggle continuation

Written 2026-11-05 before any lean-study calibration results are inspected. This
amends execution and scope under the owner's Kaggle A-to-Z master prompt. The
original Colab files, 5,293-question manifest and 2,092 media clips remain backups.
No research outcome is reported here. Qwen replication is pending and outside the
initial Gemma timing window. Historical LLaVA claims remain retracted.

## Fixed sample, evidence and passes

Gemma: initial n=2,000, calibration 400 / validation 400 / test 1,200. Seed 20261103.
Assign connected recovered-source groups (or disclosed released-clip fallback)
once to 20/20/60 candidate pools. Sample inside each pool by task x yes/no label,
proportional largest remainder and seeded question-hash order. Preserve original
splits as provenance. New split/sample manifests are separate immutable records.
No source group crosses splits. Original-source disjointness is unverified when
lineage is unrecoverable; swapped released clips may share sources across splits.

Use the existing eight frames and complete 16 kHz mono audio. Do not regenerate,
truncate or resample them differently. Native processor padding handles different
audio lengths and image shapes; swaps are not assumed to have equal token length.
K=2: full, cf0, cf1, blank, no_media. Matching gets full/no_media only. Donors use
the existing seeded clip/task rule, first two of four candidate draws, recomputed
inside the new group-disjoint pools. Donor real/swap status may be unverified;
retain the existing explicit limitation. Neither blank nor no_media can be cut.

## Predeclared calibration-only add-on

Score the initial sample in one run, calibration then validation then test within
each of six fixed hash shards. Test predictions are sealed and never analysed
before FROZEN.json. After complete calibration, estimate each primary task's
negative-label false-yes rate and its grouping-unit bootstrap CI from calibration
only. Multiply its point estimate by the initial test-negative inventory solely
for sample planning. If either primary task expects fewer than 40 test false-yes
cases, allow ONE add-on up to n=3,000 (600/600/1,800). Group assignment, seed,
stratification, donor rule and existing questions remain unchanged. The add-on
scores only additional pairs. No validation/test prediction may influence this
decision, including a decision to stop after the add-on. Actual cells below 40
false-yes cases remain underpowered, even at n=3,000. Analysis uses the final
predeclared sample. Record the decision before opening sealed test predictions.

## Numerical execution and timing auto-gate

Six worker processes, two per participant, one complete fp16 model per T4.
CUDA_VISIBLE_DEVICES maps each process to one physical card; never shard weights
across cards. Prefix caching failed parity and remains disabled. Question owner is
stable_hash(question_id) % 6. Participant indices 0/1/2 own workers 0-1/2-3/4-5.
Each worker runs one 10-question fresh-forward smoke, 10 generation agreements,
and batch-versus-fresh-unbatched-GPU parity on five calibration questions. Keep
max absolute vocabulary-logit and all-layer last-token hidden difference <=0.02,
identical answers and finite fp16 states. A fitting batch that fails parity is a
failure; do not relax tolerance. Pick the largest fitting, parity-verified batch
up to all five passes; serialize only when larger batches genuinely do not fit.

Owner timing: 50 stratified calibration questions across its two T4 processes.
Report observed p50/p90 batch wall time, allocated per-pass time, checkpoint cost,
model loading separately and grouping-unit bootstrap uncertainty where defined.
Allocated pass time is not an isolated-pass latency measurement. The explicit
exception to the ban on extrapolated research results is this private **planning
forecast**, required by the owner: forecast n=2,000 on six workers, label startup,
parallelism and failure assumptions, never publish it as an experimental result.
Auto-proceed only if parity passes and forecast <=2.5 hours. Otherwise halt; for
an excessive forecast propose one owner-reviewed cut: test 1,200 -> 900 (total
1,700), preserving calibration, validation, blank and no_media. Do not apply that
cut automatically. A parity failure needs repair, not a claim that a sample cut
fixes numerical correctness. The ~2-hour target is not a measured guarantee.

## Statistical choices and CPU-only analysis

Retain the existing diagnosis, probe preprocessing and controls. Primary endpoint:
paired test Delta AUROC of the probe-augmented model over [answer margin +
counterfactual contrast + no-media margin + calibration object yes propensity],
among Yes answers. Use 2,000 paired source-group bootstrap resamples, fixed seed.
Contrast is full margin minus mean of the two swapped margins. Also report the
blank-modality AVCD-style baseline. AVCD and AVHBench are prior work, not ours.
This is AVCD-style queried-modality contrast, not an implementation claim about
the published attentive-mask mechanism.

Secondary filter comparison (before freeze): compare the probe filter with the blank-modality filter on held-out false-Yes rates using a paired source-group bootstrap with 2,000 resamples. Each threshold uses the same validation recall-loss budget of at most 2 percentage points. Report held-out recall drop and its 95% CI for every fitted filter. When a filter cannot reduce false Yes answers on validation within budget, report “no beneficial operating point within the validation recall budget”. Filters withhold flagged Yes answers.

Calibration fits models; validation selects layer, regularisation and thresholds.
No new activation steering method is introduced. Centre features using unlabeled
calibration states and residualize against answer margin; retain shuffled-label
and Yes-minus-No projection controls. Choose policy thresholds on validation.
Ship a demo flag only if the paired false-yes reduction CI excludes zero in the
beneficial direction AND the measured recall drop is <=2 percentage points.
Report every cell's false-yes count; fewer than 40 means underpowered. Null wording:
"No supported improvement was measured"; a CI crossing zero is inconclusive,
not proof of equivalence or safety. Missing classes are reported explicitly.

Calibration MDE planning: use source-group out-of-fold predictions at a fixed
last layer/C=0.1 (no test or validation choice), paired bootstrap Delta AUROC CI
width, and the explicitly approximate normal formula (1.96+0.842)*SE, with
SE=CI_width/(2*1.96), two-sided alpha=.05 and assumed power=.80. This is a planning
approximation, not a simulated power guarantee. If undefined, report undefined;
if >.05, disclose insufficient sensitivity under limitations. Its numerical value
is pending calibration; never invent it before the run.

The complete label audit plus 50 object spot-checks is required before analysis
freeze. Audited exclusions apply in analysis, not inference. Write FROZEN.json
with model, data, code, audit and analysis-state hashes. HALT 2: owner reviews
frozen choices/validation numbers. Evaluate sealed test exactly once afterwards.
HALT 3: owner reviews test CIs, power flags and ship verdict before site work.

## Storage, privacy and provenance

One private versioned Kaggle dataset; Drive is retained as backup. No public media,
audio or frames: public content is aggregates plus our own text only. Each real
participant uses their own accounts and token; credentials never enter files or
logs. Save full-only all-layer float16 NPY shards of 250 questions, JSONL per pair,
flush every 25 completed questions and on normal exit. Abrupt termination can
repeat uncommitted work. Resume only from attached saved private output versions;
skip completed pairs. Stop normally before 11 hours so the commit can save.
Coverage checks require every final sampled pair exactly once with matching run
identity and correct assignment. No partial run becomes metrics.json.

Keep prior-work references from the original reviewed preregistration. Verify
original papers before adding any new novelty claim or public citation. No new
claim is made here. This is independent continuation in November 2026, separate
from the November 2025-February 2026 LLaVA internship.
