# CounterPrior v2 preregistration

Independent continuation, November 2026. Written before any v2 inference or test access.
Status: protocol for owner review; no results. The owner's explicit GO is required.
The LLaVA internship, November 2025–February 2026, is separate. No previous steering
claim is reused as evidence for this study.

## Question and scope

Do the named small omni model checkpoints confuse visual and acoustic evidence?
Does queried-modality contrast help, and do internal states add information beyond
confidence, contrast, language-only confidence, and object-specific model-Yes propensity?
This is a research demo motivated by assistive use. No safety or on-device latency
claim follows from a Colab T4 evaluation. Model families are selected for potential
device relevance; no phone execution is being measured.

Models: `google/gemma-4-E2B-it` and `Qwen/Qwen2.5-Omni-3B`, Thinker only.
Resolve each to an immutable Hub SHA before loading. Gemma is Apache-2.0;
the Qwen 3B checkpoint has a Qwen Research license, requiring documented applicable
research-use terms. Do not substitute models. Qwen load plus smoke has a hard
30-minute timebox; a failure permits Gemma-only results with Qwen marked unfinished.
Free T4, fp16 parameters and inputs; SDPA for text/vision, Gemma audio eager
(its chunked audio attention does not implement SDPA). No bf16, offload or quantization.
Fit, finiteness and parity are gates, not assumptions. Log actual memory/time.

## Data gates and lineage

Use the official AVHBench released subset, not the full-benchmark leaderboard.
Download links and the README's example QA schema are in `AVH_REVIEW.md`.
After GO: print the first three raw records without changing their fields, every
key/task name, and counts BEFORE filtering; save an explicit field/container/task
mapping if the real release differs (no guessed aliases). Inventory files, save hashes, and
inspect archive terms/README plus the paper's dataset-use discussion. Public
downloadability and the paper's own copyright license are not dataset-use permission.
`terms_review.json` accepts dataset decision `research_use_permitted` with saved
applicable terms, or owner decision `no_licence_stated_private_research_eval` with
saved inspected evidence/hash, source URL and reviewer. The latter grants no public
media use: never show AVHBench audio, video or frames on the site, only aggregate
numbers and our own text. Model entries still require saved license text and the
applicable passage. Pending/denied decisions block bulk runs, not early diagnostics.

The released `video_id` is insufficient to prove original-source disjointness.
Auto-generate `lineage.json`: use explicit source metadata (including clip sidecars)
and explicitly tagged `audio_<id>__video_<id>` filenames to recover both swap origins.
Union clips sharing a recovered original source into connected components. Where
origins cannot be recovered, group by released `video_id`; this does not stop the run.
The exact stated limitation, carried into manifest and `metrics.json`, is:
`lineage_unrecoverable: grouping by released clip; swapped clips may share sources across splits`.
These fallback groups are not certified independent source videos. Seed-20261103 permute groups into
60% calibration, 20% validation, remainder test (floor first two sizes, minimum one).
Require ten components and at least four eligible donor clips per split; print exact
realized counts. A giant connected component or missing task/class is a limitation,
never a reason to silently switch to row-level splitting.

For each video/task choose four distinct clips not marked swapped from its own split,
excluding its connected source component, with a deterministic hash seed. Same
donors for all questions sharing video/task; same manifest for both models.
Where real/swapped status is explicitly available, use it. Where absent, retain a
separate unknown-status marker and disclose that donor real origin is unverified;
do not describe such a donor collection as verified real clips.
Donor audio/video is an intervention input only: original target labels are never
asserted to be correct for a random donor. CIs for contrast are conditional on the
fixed donor pool and do not account for sampling a new donor collection.

Derive object keys automatically with saved `object-v1`: casefold/normalize spaces,
anchored noun-phrase extraction for "Is there a/an", "Is the ... making sound/visible"
and "Can/Do you hear/see"; unmatched prompts keep an exact normalized question key.
Spot-check up to 50 deterministic question keys by hand in the audit sheet; flag
errors without changing the frozen inference manifest mid-run.
Audit up to 100 deterministically sampled swapped QA items (also sample clips of
unknown swap status when needed), without model outputs, in parallel with GPUs.
Preserve released labels. Mark valid / exclude / uncertain,
reviewer and rationale (including off-screen sound). Exclude invalid and uncertain
audited questions from the primary analysis; retain them in the audit record.
Report audited n, error rate with source-cluster bootstrap CI, uncertain count,
and selection rule. If fewer than 100 exist, report all available; do not imply 100.
The audit does not certify labels of unaudited items. Partial sheets are accepted
during inference; inference scores ALL rows and never hashes the live audit sheet.
Only completed reviewed exclusions apply in `load_completed()`/analysis. Complete
label and object checks before `freeze_analysis()`; freeze its separate audit hash
and reject later analysis on an edited sheet. Audit completion never gates model load.

## Evidence and scoring

Eight uniformly indexed frames, each aspect-preserving within 224x224, plus the
entire mono 16 kHz WAV track, presented as ordered images and a separate audio input.
No implicit audio from video, transcript, or truncated track. Hash media once and
share the same decoded evidence across models. This tests sparse frames, not full
temporal video understanding. If memory fails, switch the entire study (both models)
to one middle frame, new run directory and new matching smoke; do not mix formats.

Prompt: original QA preceded by `Question:`, followed by `Answer Yes or No only.`;
native chat template, thinking disabled where supported. Enumerate every non-special
vocabulary token whose decoded, whitespace-stripped, casefolded string is yes/no.
Class score is logsumexp over unique IDs, margin = Yes score minus No score.
Answer Yes iff margin > 0; exact ties are No. Save mapping and raw class scores.
This is a constrained first-token decision, not an unrestricted generated answer.

One score forward pass captures the last query-token state from the embedding
output and all decoder layers. Published `hidden_states[-1]` may include final
normalization; record that distinction and use the library-defined representation
consistently. Language-model head sees only the last position to avoid full-sequence
vocabulary logits. Media-prefix caching uses a deep copy per question, explicit
model-specific position IDs, and no modality re-encoding in the suffix.

Early `pre_smoke(cfg, raw_qa, media_dir)` runs before terms/lineage/manifest: fixed-seed
20 raw QA rows from at least three videos, prepared privately, the same full-pass
parity/finiteness/VRAM checks and ten one-token generation agreement checks. Save
`<model>/pre_smoke.json`, engineering-only, never a result or formal bulk gate.
Raw diagnostic items may later land in test; do not use their responses to tune
the protocol or statistical choices. Formal smoke still uses calibration only.

Formal calibration-only smoke: 20 QA items on the full pass, plus the first item on every
other pass; compare fresh full pass and cached
suffix, all vocabulary logits and all last-token states max absolute diff <=0.02,
identical answers. On the first ten full items, greedy `generate(max_new_tokens=1)`
must produce an actual Yes/No variant agreeing with the class decision. Non-Yes/No
outputs fail the gate. This one-token diagnostic is the only generation exception.
Print peak allocated VRAM, finite-output checks, timing, package versions, model SHA.
Passing these checks is required; none has yet been run.

### November 4 engineering amendment: explicit full-forward fallback

Owner-provided early diagnostics failed the prefix gate on one Gemma QA item.
Do not relax the registered tolerance or claim that prefix caching was verified.
`verified_prefix` mode remains subject to the original cache gate. An explicit
`full_forward` fallback uses the original fresh full pass for every question, with
no prefix reuse or encode-once claim. Save the actual failed diagnostic and a
per-model `scoring_amendment.json` before bulk/test inference. Its hash and mode
enter run identity. This companion amendment also applies to the saved Drive copy
of the preregistration; it must not silently overwrite an already frozen analysis.

In fallback mode, the same <=0.02 numerical and answer-agreement checks compare
two genuine full passes; label them repeatability, never prefix parity. Keep the
ten generation checks, finiteness, VRAM, the independent formal manifest smoke,
and pre-test analysis freeze. Measure observed latency over the complete early
smoke, report n/CI, and assess time before launching bulk work. Timing includes
processor preparation, forward pass and feature extraction, but not source media
preprocessing or model loading. No population result follows from these diagnostics.
All research/statistical choices and media evidence remain unchanged. The fallback
is a computation/speed trade-off, not a successful fix of the cache discrepancy.

Passes: full; four queried-modality swaps; queried-modality black frames/silence;
question only. AV Matching is secondary and uses full + no-media only because it
has no single queried sense. Contrast = full margin minus mean swapped margin;
secondary blank contrast = full margin minus blank margin. These are **AVCD-style
adaptations**, not the attention-guided masking or adaptive decoding of published AVCD.
An exact published-AVCD comparison is an explicitly unimplemented limitation.

## A. Diagnosis

Per model and task: accuracy, false-yes rate = accepted Yes / ground-truth No,
recall = accepted Yes / ground-truth Yes. Full and no-media are separate paired
conditions. Report numerators/denominators, grouping-unit n (source components only
where recovered; otherwise released clips, with the overlap limitation), and
95% percentile cluster-bootstrap CIs with 2,000 resamples, fixed seed 20261103.
Use paired source resampling for differences. Undefined class draws are discarded;
CI is unavailable unless at least 90% are valid and at least two clusters exist.
No cross-model pooled headline; leaderboard numbers are context, not matched scores.

## B. Contrast gate and product decision

Frozen per-model/task validation thresholds: among all unique contrast scores
(plus finite boundary values accepting all/none), minimize false-yes rate subject
to recall loss <=0.02 versus full answers. Ties maximize accepted-Yes coverage,
then choose the numerically smallest threshold. Keep a full Yes only if contrast
>= threshold; rejected Yes becomes `I'm not sure`, never a claimed correct No.
Original No answers remain No. Blank thresholds are chosen independently by the
same rule, secondary only. Do not select random-swap versus blank using test.

On test, the demo flag ships only if the paired false-yes reduction CI lower bound
is >0 and recall loss point estimate <=0.02. Report the recall-loss CI too; this
point-estimate rule does not establish noninferiority with confidence. Otherwise
`did_not_ship`. Each task and model gets its own verdict. Missing CI fails shipping.
No aggregate rollout claim when a task fails. Ship here means an optional demo
configuration, not a production or assistive-safety release.

Report false-yes before/after, recall before/after, and share of all questions
answered (No plus accepted Yes). Selective risk is error rate among answered
questions; abstentions are never scored as correct. Threshold-swept risk/coverage
is descriptive test analysis; it cannot be used to replace the frozen threshold.

## C. Internal probe

Population: full-pass Yes answers; outcome 1=false-yes, 0=true-yes. Calibration fits,
validation selects, test evaluates once. Minimum needed for an adequately populated
cell is 40 false-yes examples; below this always label `underpowered`, never
`doesn't work`. Missing either class makes AUROC and supervised fit unavailable.
Report both class counts and source clusters. Forty is a reporting floor, not a
formal guarantee of power or a prospective effect-size calculation.

Per task, compare a standardized L2 logistic baseline with four inputs: full margin,
random-swap contrast, no-media margin, and Laplace-smoothed per-object model-Yes
propensity estimated from calibration only. Calibration training propensities leave
the row's whole source component out; validation/test use all calibration. Unseen
object propensity is 1/2. Ground-truth test labels never build the base rate.

For every available library-defined layer, center using unlabeled calibration
activations per task, then regress each centered feature on an intercept and full
margin (ridge 1e-4 on margin, 1e-8 intercept). Apply only the frozen calibration
transform elsewhere; no test centering. Standardize residual features in the fitted
logistic pipeline. Hyperparameter grid C={0.01,0.1,1}; baseline and augmented
models both select maximum validation AUROC, ties favor smaller C. Select layer by
augmented validation AUROC, ties earlier layer. Do not refit after validation.

Headline: paired test ΔAUROC of **baseline + residual probe over baseline**. Also
report probe-only AUROC, baseline AUROC, and an honest null/negative/inconclusive
interpretation. This nested comparison answers whether features add information;
it does not imply a pure probe outperforms contrast alone. Within-margin quintiles
use validation Yes-margin quantile edges; report per-bin n and false-yes count.

Controls: (1) calibration labels permuted at fixed seed within the fitted task,
same layer and C, evaluated on untouched test labels; (2) remove the normalized
Yes-minus-No mean unembedding direction from residual features before fitting,
same selected layer, own C chosen on validation. Direction is a linear readout
control, not a complete account of Gemma logit softcapping or normalization.
All probe controls are secondary; only the nested main ΔAUROC is the primary test.
Model/task CIs are pointwise; no family-wide significance claim or cherry-picked win.

Product probe policy uses augmented hallucination-risk probability, frozen validation
threshold chosen by the same false-yes/recall rule (risk <= threshold accepts Yes).
Compare it with no fix and random-swap contrast using paired test quantities.
Do not call an offline readout deployable before device latency is measured.

## D. Transfer and geometry (secondary, first analysis scope cut)

Use source-task probe-only readout at its frozen layer on the other task's test Yes
answers, applying target-task unlabeled calibration centering and the source fitted
readout. Target in-task readout is a reference ceiling, not an oracle bound.
Report AUROC and ratio (transfer AUROC-.5)/(in-task AUROC-.5), undefined if the
reference <=.5, with paired cluster CIs and counts; do not clip unfavorable ratios.
Geometry is calibration-only: false-yes minus true-yes mean direction per task/layer,
cross-task cosine and cosine to the Yes-No unembedding, cluster CIs and 2,000
within-task shuffled-label null draws. Per-layer results are exploratory, no selected
significant layer claim. Missing classes or small samples are explicit limitations.

## Reliability, provenance, and stop conditions

One shared private Drive folder, separate per-model/stage/partition journals. Never
concurrently run the same partition. Fixed partition count across restarts. Per-row
fp16 feature file is durable before its JSONL commit; compact derived fp16 NPZ shards
every 200 new questions and at stage completion. Preserve recovery files. Only a
torn final line can be repaired; interior corruption, duplicate IDs, changed code,
packages, model, manifest, terms or evidence cause a stop/new study directory.
Audit edits during inference do not change its signature; analysis freezes the
completed sheet separately and rejects post-freeze edits.
Drive is not a transactional store; an exclusive lock is best effort on Drive FUSE.
After a dead runtime, verify it is stopped before manually removing its stale lock.

Ten minutes without a new durable row dumps all Python stacks and ends that
runtime kernel (a wedged CUDA call cannot reliably be interrupted by an exception).
Qwen load+smoke additionally has a hard 1800-second total deadline. Gemma uses a
3600-second outer inactivity watchdog. Touch the outer watchdog after model loading;
Qwen's independent deadline still caps total load/check time. The inner smoke
watchdog is touched after each completed check. Before test
inference, serialize calibration models, validation thresholds/layers, hashes, and
freeze marker. Any incomplete pass blocks metrics export; no partial extrapolation.
`metrics.json` contains only measured complete test results, with CIs/n. Notebooks
are saved with no outputs. After notebook authoring, STOP until owner types GO.
