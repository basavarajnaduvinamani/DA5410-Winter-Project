# Explicit full-forward fallback after prefix parity failure

Date: 2026-11-04. Status: authored and CPU-tested; GPU fallback validation pending.

## Context

Owner-supplied single-item Gemma diagnostics on T4 agree on Yes/No, but fail the
registered <=0.02 absolute-logit/hidden-state cache gate. The original diagnostic
has max errors 0.046875 / 0.125; text-only math 0.0703125 / 0.125; disabling reduced
FP16 GEMM reductions 0.0625 / 0.125. These are engineering observations on one item,
not population model-performance estimates. A global math-attention diagnostic
also OOMed in vision. No successful cached bulk run is claimed.

## Options

A. Continue searching for numerical/backend changes to satisfy prefix parity.
B. Keep prefix mode blocked and explicitly score complete inputs with fresh passes.

## Decision

B. Add `scoring_mode=full_forward` as an explicit fallback, preserving the failed
prefix evidence in a per-model preregistration amendment before any bulk/test run.
The original `verified_prefix` mode retains its original gate. Full-forward mode
never requests a prefix cache and rejects attempts to do so. Its smoke compares
two actual fresh passes, with generation agreement, finiteness and memory checks;
it never calls that comparison prefix parity. A retained model can be reused.

## Consequences

Full-forward mode repeats media/text encoding for each question and may be slower.
Measure complete smoke timing before committing compute to bulk inference. Do not
claim encode-once, verified prefix reuse, or a successful cache optimization for this
mode. Frame/audio evidence, labels, splits, statistical choices, controls, audit and
analysis freeze remain the same. Modes and amendment hashes enter run identity and
metrics, preventing mixed-mode journals. Formal smoke remains separate from early
diagnostics. This amendment is pre-test; it changes an optimization, not thresholds.
