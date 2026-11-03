# CounterPrior v2 notebook review

2026-11-03. Independent continuation. Authoring phase only; HALT pending owner GO.

## Deliverables and operation

- `notebooks/avh_gemma.ipynb`: isolated Gemma T4 environment.
- `notebooks/avh_qwen.ipynb`: isolated Qwen Thinker environment, load/smoke timebox.
- `notebooks/avh_common.py`: shared gates, preprocessing, inference, recovery, CPU analysis.
- `docs/PREREGISTRATION.md`: protocol to copy unchanged into the shared private study folder.

The notebooks contain operating instructions and deliberately invalid templates for
terms and lineage. Set OWNER_RELEASE only after the owner types GO in chat. Before
inference, complete documented usable terms, original-source mapping, the blind
swapped-item audit, and measured smoke gates. Account A prepares data once; Account
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
- Pinning uses Gemma Transformers 5.20.2 and Qwen 4.57.1 in separate runtimes;
  actual package sets, revision SHAs and source hashes are saved and checked.
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
- Source lineage and terms may be absent from the released package. The notebooks
  stop in that case; inventing either would undermine the experiment.
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

## Final authoring validation

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
