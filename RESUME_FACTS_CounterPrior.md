# CounterPrior v2 — facts at the notebook review halt

Date: 2026-11-03. Independent continuation, November 2026. The internship period
November 2025–February 2026 covers earlier LLaVA work only.

## Measured facts

Software checks are deterministic artifact counts, so sampling CIs do not apply.
No v2 model sample, effect size, CI, latency, or VRAM measurement exists yet.

| Fact | Number / status | Source file |
|---|---|---|
| Requested model notebooks saved, unexecuted and schema-valid | 2 | `notebooks/avh_gemma.ipynb`, `notebooks/avh_qwen.ipynb`, `docs/AVH_VALIDATION.json` |
| Shared review-gated runtime authored | Present; GPU behavior unrun | `notebooks/avh_common.py` |
| Python tests in final suite | 55 passed, 0 failures, 0 errors; n=55 checks, CI not applicable | `docs/AVH_VALIDATION.json` |
| New AVH contract tests included in that suite | 14 passed; n=14 checks, CI not applicable | `tests/test_avh.py`, `docs/AVH_VALIDATION.json` |
| Study protocol with source-connected splits and validation freeze | Authored before v2 inference | `docs/PREREGISTRATION.md` |
| Dataset item count, usable terms and original-source lineage | Not established; inference gated | `docs/AVH_REVIEW.md` |
| fp16 T4 fit, cache parity, generation agreement, model performance | Not run; pending GO | `docs/AVH_VALIDATION.json` |
| Incremental probe AUROC, false-yes/recall effects, replication | In progress; no measured number | `docs/AVH_VALIDATION.json` |

## Candidate lines — ongoing work only

- APM: **(Continued independently, Nov 2026)** CounterPrior v2 — in progress:
  defined a validation-selected abstention policy and a false-yes/recall ship rule;
  product effects are not yet measured.
- Pre-doc: **(Continued independently, Nov 2026)** CounterPrior v2 — in progress:
  preregistered a source-disjoint audiovisual hallucination study and an internal
  probe comparison against AVCD-style contrast; evaluation is pending.
- SDE: **(Continued independently, Nov 2026)** Authored two review-gated Colab
  notebooks and a shared checkpointing runtime; 14 new CPU contract tests pass.
  T4 compatibility and multi-account recovery remain unvalidated.

These are drafting facts, not completed research accomplishment claims. Replace
ongoing language only after complete runs populate traceable `metrics.json` files.

## Do not say

- Do not revive the retracted LLaVA accuracy, steering, control or attention-head claims.
- Do not date this work inside the internship or imply it was supervised there.
- Do not claim any v2 sample size, performance improvement, T4 fit, device latency,
  completed replication, working three-account recovery, or empirical probe advantage.
- Do not present AVHBench, published AVCD, pre-generation probing, or encoder-state
  reuse as our inventions. Random swaps and uniform blanks are AVCD-style adaptations.
- Do not claim an exact AVCD reproduction, a first evaluation, production readiness,
  solved hallucination, or safety for blind and low-vision users.
- Do not convert synthetic unit-test results into research measurements.
- Do not assume public download links license the data or public example media.

## Interview questions and honest answers

1. **What have you established so far?** The review artifacts and CPU code contracts
   are complete. Neither model has run in this phase; there are no research results.
2. **How will you avoid leakage in swapped audiovisual clips?** Require both original
   source identities, form connected components before splitting, and choose donors
   inside each split. Without trustworthy provenance, the experiment stops.
3. **What makes this different from AVCD or HALP?** It proposes an incremental-information
   test on cross-sense errors in the selected small checkpoints, against an explicitly
   labeled contrast adaptation. Those published methods are prior work. Whether the
   probe adds useful information remains an empirical question.

Strongest result: reviewable notebooks, preregistered gates and passing CPU contracts.
Weakest gap: usable data provenance/terms and actual GPU behavior remain unverified.
Next upgrade: after owner GO, establish those data gates and execute the T4 smoke.
