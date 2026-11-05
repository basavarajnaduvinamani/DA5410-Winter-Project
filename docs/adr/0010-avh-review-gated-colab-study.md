# AVH review-gated study, separate from LLaVA

Date: 2026-11-03. Independent continuation. Status: implemented for review, unrun.

## Context

The owner requests two T4 notebooks and a shared runtime, then an explicit halt
until GO. Existing LLaVA work and site must not be altered in this phase. The
proposed AVCD baseline description does not match published attentive masking.
Original source provenance for released swapped clips is not yet established.

## Options

A. Separate per-model environments, shared frozen evidence and manifests,
source-connected splits, private per-partition journals, calibration/validation
freeze before test; accurately named AVCD-style contrast.
B. Single environment, release-ID splits, concurrent appends to one journal,
and describe random swaps as the published AVCD implementation.

## Decision

A. Keep the requested Gemma/Qwen checkpoints without substitutions. Require real
license evidence and both source origins for swaps. Enforce GO in notebook cells
and runtime entry points. Require measured fp16 T4 fit/cache/generation smoke.
Use separate worker journals under one shared Drive folder, durable question rows,
recovery features, and derived 200-question shards. Freeze analysis before test.
Do not publish metrics from incomplete passes. Save the tested ADR in the study commit.

## Consequences

Unknown dataset-use rights or source mapping can stop the study. Drive locks are
best effort, not distributed transactions. More disk space buys restart safety.
Random-swap and blank baselines are adaptations; an exact AVCD reproduction is
not implemented or claimed. No fit or empirical research finding is claimed now.
No site refresh, deployment, merge, or GPU execution occurs before owner GO.
