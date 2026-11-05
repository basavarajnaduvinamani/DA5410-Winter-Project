# Early diagnostic smoke and analysis-only audit

Date: 2026-11-03. Implements only the owner's Round 1 review changes to ADR 0010.

## Context

The first authoring round placed audit and full provenance before loading a model,
preventing prompt T4 compatibility diagnosis. The owner requests relaxed lineage
and private dataset-use decisions, with explicit limits, and a second review halt.

## Options

A. Keep all initial manual gates before loading.
B. Early engineering-only smoke, formal manifest smoke before bulk, audit at freeze,
auto object keys and disclosed released-clip fallback when origins are unavailable.

## Decision

B, as explicitly requested. Inference journals/signatures do not depend on audit
edits; all rows are scored. Completed exclusions apply in analysis, and its audit
hash freezes separately. Partial audit cannot freeze. Early smoke has a distinct
artifact and does not open bulk inference. The existing statistical rules remain.

## Consequences

Recovered origins retain connected grouping; clip fallback cannot guarantee source
disjointness and is disclosed in the protocol and metrics. Dataset no-license private
research evaluation is an owner decision, not a license claim: no public AVHBench
audio/video/frames. Model license evidence remains required. Both notebooks stop
for owner review again, without executing GPU work in this revision.
