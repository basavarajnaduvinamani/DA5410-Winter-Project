# CounterPrior v2 — facts at the notebook review halt

Date: 2026-11-04, Gemma runtime repairs and explicit full-forward fallback. Independent continuation, November 2026. The internship period
November 2025–February 2026 covers earlier LLaVA work only.

## Measured facts

Software checks are deterministic artifact counts, so sampling CIs do not apply.
No complete v2 research result exists. The owner supplied single-item engineering observations and a release inventory; these do not establish population performance.

| Fact | Number / status | Source file |
|---|---|---|
| Requested model notebooks saved, unexecuted and schema-valid | 2 | `notebooks/avh_gemma.ipynb`, `notebooks/avh_qwen.ipynb`, `docs/AVH_VALIDATION.json` |
| Gemma license lookup repaired and live-checked at owner-reported revision | Succeeded; metadata/license only, no weights | `docs/AVH_VALIDATION.json`, `notebooks/avh_common.py` |
| Shared review-gated runtime authored | Present; prefix gate failed on an owner-run engineering item; fallback validation pending | `notebooks/avh_common.py` |
| Python tests in final suite | 69 passed, 0 failures, 0 errors; n=69 checks, CI not applicable | `docs/AVH_VALIDATION.json` |
| New AVH contract tests included in that suite | 28 passed; n=28 checks, CI not applicable | `tests/test_avh.py`, `docs/AVH_VALIDATION.json` |
| Protocol with recovered-source grouping, disclosed clip fallback and validation freeze | Authored before v2 inference | `docs/PREREGISTRATION.md` |
| Dataset item count, terms decision and original-source lineage | Inventory completed; terms decision/lineage recovery pending | `docs/AVH_REVIEW.md` |
| Round 1 requested fixes | Authored; owner cached early smoke stopped on parity | `docs/AVH_REVIEW.md`, `notebooks/avh_common.py` |
| fp16 T4 fit, cache parity, generation agreement, model performance | Model loaded on T4; cache errors exceed gate on n=1; full 20-item smoke incomplete | `docs/AVH_VALIDATION.json` |
| Incremental probe AUROC, false-yes/recall effects, replication | In progress; no measured number | `docs/AVH_VALIDATION.json` |

| Failed Gemma prefix-cache engineering gate | n=1 item; max hidden difference 0.125 versus <=0.02 limit; no population CI | `docs/AVH_CACHE_STATUS.json` |
| Explicit full-forward fallback | Authored; actual GPU repeatability/timing pending | `docs/adr/0012-explicit-full-forward-fallback.md` |

## Candidate lines — ongoing work only

- APM: **(Continued independently, Nov 2026)** CounterPrior v2 — in progress:
  defined a validation-selected abstention policy and a false-yes/recall ship rule;
  product effects are not yet measured.
- Pre-doc: **(Continued independently, Nov 2026)** CounterPrior v2 — in progress:
  preregistered an audiovisual hallucination study with recovered-source grouping
  or an explicitly limited released-clip fallback, and an internal
  probe comparison against AVCD-style contrast; evaluation is pending.
- SDE: **(Continued independently, Nov 2026)** Authored two review-gated Colab
  notebooks and a shared checkpointing runtime; 28 AVH CPU contract tests pass.
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
- Do not claim verified prefix reuse or encode-once optimization for full_forward mode; the cached path failed its gate.
- Do not assume public download links license the data or public example media.
- Do not claim source-disjointness when lineage is unrecoverable. The owner's
  no-license private-evaluation decision never authorizes public AVHBench media/frames.

## Interview questions and honest answers

1. **What have you established so far?** The review artifacts and CPU code contracts
   are complete. Gemma loaded on the owner's T4, but its cached smoke failed numerical parity; there are no complete research results.
2. **How will you avoid leakage in swapped audiovisual clips?** Require both original
   source identities where recoverable, form connected components before splitting,
   and choose donors inside each split. Unrecoverable lineage uses clip grouping
   with a stated possible source-overlap limitation; that fallback cannot rule out leakage.
3. **What makes this different from AVCD or HALP?** It proposes an incremental-information
   test on cross-sense errors in the selected small checkpoints, against an explicitly
   labeled contrast adaptation. Those published methods are prior work. Whether the
   probe adds useful information remains an empirical question.

Strongest result: reviewable notebooks, preregistered gates and passing CPU contracts.
Weakest gap: fallback GPU validation, original-source recovery and owner terms decision remain pending.
Next upgrade: execute the full-forward early smoke on the retained T4 model, then assess measured latency before bulk work.
