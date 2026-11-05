# CounterPrior v2 notebook review

2026-11-03. Independent continuation. Authoring phase only; HALT pending owner GO.

## Deliverables and operation

- `notebooks/avh_gemma.ipynb`: isolated Gemma T4 environment.
- `notebooks/avh_qwen.ipynb`: isolated Qwen Thinker environment, load/smoke timebox.
- `notebooks/avh_common.py`: shared gates, preprocessing, inference, recovery, CPU analysis.
- `docs/PREREGISTRATION.md`: protocol to copy unchanged into the shared private study folder.

The notebooks contain operating instructions and a pending owner terms template.
Set OWNER_RELEASE only after the owner types GO in chat. After raw inventory, run
the early engineering smoke before manual gates. Bulk inference requires the terms
decision, auto-generated lineage/object keys, manifest/media and formal smoke.
Audit runs in parallel and gates analysis freeze only. Account A prepares data once; Account
B reads identical artifacts; spare C takes disjoint stages/partitions. No local model
weights, dataset download, test inspection, or experimental analysis ran in authoring.

The declared T4 precision is fp16; model-card default bf16 is explicitly overridden.
There is no offloading, quantization, FlashAttention-2 or silent model replacement.
Whether either model fits and remains numerically stable on this T4 protocol is unknown.
Qwen's 30-minute timebox includes model weight loading and smoke; a cold download may
use that budget. Failed gates stop rather than quietly change the experiment.

## Verified sources and corrections

Sources checked 2026-11-03. Metadata does not establish hardware fit or novelty.

| Source | Verified implication for this plan |
|---|---|
| [Gemma E2B-it model card](https://huggingface.co/google/gemma-4-E2B-it) | Exact repository exists, Apache-2.0, multimodal inputs. Effective size is not a VRAM measurement. |
| [Gemma Transformers API](https://huggingface.co/docs/transformers/main/model_doc/gemma4) | Native processor, conditional generation class, audio inputs; use fp16 explicitly and test. |
| [Qwen 3B model card](https://huggingface.co/Qwen/Qwen2.5-Omni-3B), [license](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/main/LICENSE) | Exact checkpoint exists with Qwen Research license, not Apache-2.0. Save/review applicable terms. |
| [Qwen API](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_omni) | Thinker-only model avoids speech generation; separate image/audio inputs supported. |
| [AVHBench official README](https://github.com/kaist-ami/AVHBench) | Released subset has QA fields video_id/task/text/label; real and swapped clips. README does not supply an explicit dataset license or source-lineage map. |
| [AVHBench paper](https://arxiv.org/html/2410.18325v2) | Benchmark being followed up, not our task invention; paper copyright is not permission for underlying clips. |
| [AVCD](https://arxiv.org/abs/2505.20862) | Attention-guided modality perturbation and adaptive contrastive decoding. Random donor swaps and uniform blanks are **AVCD-style adaptations**, not its published implementation. |
| [OmniDPO, 2509.00723](https://arxiv.org/html/2509.00723v1) | Already evaluates Qwen2.5-Omni on AVHBench. Inspected text does not establish the 3B variant. No first-Qwen claim. Training-based alignment is distinct from this frozen-backbone readout. |
| [When Eyes and Ears Disagree, 2511.10059](https://arxiv.org/html/2511.10059v1) | Verified title/ID; audiovisual confusion and Qwen-based mitigation are prior work. |
| [OmniHalluc-L, 2606.03614](https://arxiv.org/html/2606.03614v1) | Counterfactual long-video event binding and perturbation reliability calibration are prior work; our sparse-frame AVHBench protocol differs. |
| [REVE, 2609.26028](https://arxiv.org/html/2609.26028v1) | Reuses encoder states for acoustic-event verification; internal-state reuse is not a new concept. Our queried decoder states are a distinct test. |
| [HALP, 2603.05465](https://arxiv.org/html/2603.05465v1) | Pre-generation internal-representation hallucination probes already exist. Proposed contribution is their incremental information on this cross-sense task, if measured. |

All requested paper IDs resolved to the stated topics. No exhaustive priority search
was completed, so no "first" claim is permitted for Gemma or any other component.
Published results are not copied into this study's `metrics.json`.

Official private release download identifiers:

- Videos: `10-Qp8zxA3ITT-ileEnCgJkf5Nzx1wry7`.
- QA: `1KcYDAv9lLy3hsx5rWdfRqMFV2NYcZ94W`.

Download, count, inspect archive README/terms and save evidence after GO. Current
item count and dataset-use permission remain unverified. Neither is fabricated.

## Important review limitations

- GPU forward/cache/generation behavior is **unrun**; compatibility is not certified.
- Gemma selects a stable Transformers release at runtime using `pip index versions`,
  versioned Gemma 4 docs and a real class import. Qwen verifies its Thinker class on
  4.57.1. Selected versions, package sets, revision SHAs and hashes are saved/checked.
- A real phone, end-to-end assistive task, and accessibility efficacy are untested.
- The random-swap CIs condition on the frozen donor collection. Donor reuse induces
  dependence not covered by resampling only target source components.
- A minority false-yes floor of 40 is a reporting rule, not a power calculation.
- Final hidden-state normalization and nonlinear Gemma logit capping limit a
  mechanistic interpretation of the unembedding projection control.
- The code enumerates single-token Yes/No variants. Ten one-token greedy checks
  gate that approximation; it does not sum probabilities of arbitrary multi-token answers.
- No exact published AVCD result can be claimed from this implementation. A matching
  attentive-mask implementation would be a separately reviewed extension.
- Unrecoverable lineage uses explicitly limited released-clip grouping; never claim
  verified source-disjoint test splits in that case. Unknown swap status and donor
  real-origin uncertainty are also recorded. The no-license owner decision permits
  private evaluation only, with no public AVHBench media or frames.
- Test thresholds never retune. "Ship" denotes a research-demo flag only. Recall
  loss <=2 points uses the requested point estimate; its uncertainty is also reported.
- Secondary geometry is CPU-expensive and can be skipped per the approved cut order.

## Panel decisions

PANEL — scope and execution gates
Options: A) write reviewable notebooks and stop; B) run experiments and refresh the site now.
VP Product: make the experiment reviewable. Research: preregister before test access. ML Eng: verify real APIs and measure fit. Data Sci: freeze source-level splits. Platform: preserve existing work. Design/Privacy: gate dataset use on terms.
Disagreement: none; explicit halt takes precedence over the later schedule.
Call: A, on claude/avh-2026-11; no experiments before GO. Reversible? yes.
Resume impact: PRE-DOC / SDE: reproducible protocol and auditable execution.

PANEL — protocol and checkpoint design
Options: A) source-connected splits, frozen validation choices, separate worker journals; B) row-level splits and shared append file.
VP Product: false claims and detections together. Research: label adaptation accurately. ML Eng: cache parity gate. Data Sci: isolate both source origins and donors. Platform: one writer per journal. Design/Privacy: keep media private pending terms.
Disagreement: Platform favors separate journals because concurrent Drive appends can corrupt shared files.
Call: A; one shared folder with independent journals and deterministic partitions. Reversible? yes, before collection.
Resume impact: APM / PRE-DOC / SDE: ship rule, leakage controls and recovery.

These blocks were recorded under the initial AGENTS instructions; AGENTS v2 now
reserves subsequent blocks for major hard-to-undo decisions.

## Initial authoring validation (before Round 1)

The complete Python `tests/` suite passed: 55 checks, including 14 new AVH tests.
Both authored notebooks pass schema validation; every code cell compiles, has a
null execution count and empty outputs. The first cell's GO refusal is tested.
`docs/AVH_VALIDATION.json` records artifact hashes, exact command, counts and limits.
These are deterministic software checks; CIs are not applicable. No research metric
or empirical result is inferred from these synthetic contract fixtures.

Initial test attempts exposed sandbox temporary-directory atomic-replace restrictions
and unscoped collection of existing Lighthouse browser-profile files under `site/`.
The successful final command uses a project-contained pytest base directory and
explicitly collects the Python `tests/` suite. Its exit code is zero; a pyreadline
console-destructor warning appeared at interpreter shutdown. No site code was changed.

Strongest result: reviewable artifacts with frozen protocol and passing CPU contracts.
Weakest gap: dataset rights/lineage and T4 execution remain unverified.
Next upgrade: after owner GO, resolve the data gates and execute measured GPU smoke.

## Round 1 — requested diff summary

1. Inference never reads the live audit or excludes its rows. Partial sheets are
   accepted; completed reviewed exclusions apply in analysis. Completion gates
   freeze; the completed audit hash is frozen separately and later edits rejected.
2. `pre_smoke(cfg, raw_qa, media_dir)` selects 20 raw QA from at least three videos,
   prepares media, loads the model and uses the shared parity/generation/VRAM/fp16
   checks. Its engineering-only artifact never replaces the formal smoke.
3. Inventory prints raw records/keys/tasks/counts first; schema differences require
   an explicit saved mapping. It preserves metadata for lineage recovery.
4. Lineage/object keys auto-generate. Explicit metadata/tagged filename sources
   are recovered where available; otherwise the exact released-clip overlap
   limitation is retained in preregistration, manifest and final metrics. The audit
   includes up to 50 deterministic object-key checks without per-question hand entry.
5. Owner no-license private-evaluation decision is accepted for the dataset only.
   Saved model license text is still required. Metrics record the no-public-media policy.
6. Cell 3 queries real package releases, checks versioned Gemma docs and the selected
   class import; Qwen imports its Thinker class on 4.57.1. Nothing was installed locally.
7. Gemma outer timeout is 3600 seconds, Qwen retains its 1800-second total deadline;
   the outer watchdog is touched after loading, inner checks touch their stall watchdog.
8. Prefix/suffix slicing retains token-type and multimodal-type tensors, including
   `token_type_ids`, for both model paths.
9. Both notebook headers include the six-line Tonight's order. Account C takes
   over stopped partitions or distinct stages only. Model-load reuse avoids a second download/load.

Statistics, parity tolerances, journaling, frozen validation choices and power rules
are retained. No model or dataset run occurred. Final Round 1 software validation
is recorded in `AVH_VALIDATION.json`; synthetic checks are not research findings.
The final Python suite passed all 63 checks, including 22 AVH checks. Both revised
notebooks validate with empty outputs and null execution counts. Early/formal GPU
smoke, runtime package installation and real dataset inventory remain unrun.

PANEL: allow early diagnostics and qualified lineage fallback
Options: A) keep blocking provenance gate; B) requested early smoke plus disclosed fallback.
Key views: Research/Data Science: clip grouping cannot establish source-disjointness. Platform: audit edits must not invalidate journals. Privacy: owner no-license decision means private evaluation only.
Call: B as requested; preserve overlap limitation and media prohibition. Reversible? yes, before results.
Resume impact: PRE-DOC / SDE: distinguish verified provenance from practical fallback.

## November 4 runtime repair — model-license lookup

The owner ran Gemma setup and reported successful selection/import verification of
Transformers 5.22.1 / `Gemma4ForConditionalGeneration`. Metadata resolution then
failed because `resolve_model()` assumed a repository file named `LICENSE`.
The actual pinned revision `4adf15748de9e1a0097b6274cee9c15836c41c37` links its
license from `README.md` instead. This was an authoring bug, not a model-fit failure.

The corrected runtime inspects repository file names before downloading a bundled
license. For Gemma without one, it verifies the pinned model card's explicit
Apache-2.0 declaration and Google license link, fetches that publisher page, checks
the expected license text and saves its provenance/hash plus pinned card evidence.
Unverified cards, empty licenses and unexpected redirects still stop. The live
metadata/license check succeeded on the exact revision above. No model weights
were downloaded by that check. Statistical choices and inference code are unchanged.

Owner recovery: replace only `reviewed/avh_common.py` in the shared Drive folder,
reload it through the existing notebook `spec.loader.exec_module(avh)`, and finish
the metadata/license/config assignments. Do not rerun installation or reset the
runtime merely for this repair. Continue to section 4 inventory afterward.

## November 4 runtime repair — Gemma audio attention backend

The owner's early smoke downloaded weights and stopped during model construction:
`Gemma4AudioModel` rejects an explicitly global SDPA backend in Transformers 5.22.1.
The [versioned implementation](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/gemma4/modeling_gemma4.py)
declares its chunked audio attention eager-only. The corrected loader uses a
per-component map: SDPA for text/vision, eager for audio, and records the map in the
environment. Qwen's SDPA setting is unchanged. fp16 and all formal smoke checks remain.
The preregistration now states this explicit implementation constraint. It changes
the audio backend, not the model, data or statistical plan; cache parity and fp16
finiteness still have to be measured. No successful Gemma forward pass or T4 fit is
claimed from a completed download or this source-level repair.

## November 4 prefix-cache diagnostics and full-forward fallback

The owner supplied one-item diagnostics: default prefix vs full errors were
0.046875 logits / 0.125 hidden; text-only math gave 0.0703125 / 0.125; disabling
reduced FP16 GEMM accumulation gave 0.0625 / 0.125. Yes/No answers agreed; this
does not pass the <=0.02 gate. The latter diagnostic's reported margins matched,
but that does not establish hidden-feature parity. Global math attention OOMed
in the vision encoder. These are single-item engineering observations, not
research metrics, accuracy estimates or a validated runtime guarantee.

PANEL: use the full forward pass as a fallback
Options: A) continue debugging prefix reuse; B) fresh full pass per question.
Key views: Research/Data Science: B uses the reference computation and preserves statistics. ML/Platform: measure speed before bulk.
Call: B explicitly, caching disabled and failed gate recorded. Reversible? yes, via configuration.
Resume impact: PRE-DOC / SDE: disclose failed optimization and independently validate fallback.

Implementation and consequences are in ADR 0012. The retained scorer can switch
dispatch without reloading weights, retaining original model/media configuration.
The fallback early smoke checks 20 full-pass repeats plus ten generated tokens,
reports observed latency and marks prefix_cache_verified=false. It does not replace
formal manifest smoke. Performance/success on actual T4 fallback is pending.
