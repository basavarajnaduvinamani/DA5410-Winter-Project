# Preserve the active protocol and separate newer-model replication

## Context
The existing LLaVA configuration trains an any-error detector, whereas the motivating question mentions false-yes hallucination. The owner has started the unchanged notebook. The setup failed before inference, with an inherited inline Matplotlib backend. No outcomes have been inspected.

## Options
A. Preserve the run and clearly distinguish answer error from object hallucination.
B. Change its target and restart; this creates a different experiment.
C. Add a separate modern-model, detection-only replication after LLaVA.

## Decision
Use A and C. Publish the exact configured target. Register a separate Qwen2.5-VL-7B-Instruct checkpoint under its verified Apache-2.0 licence. The smaller 3B checkpoint has a custom Qwen Research licence, so it is not silently treated as Apache-2.0. Use the same question files, official scorer, seed, image partition and error-detection target; no modern steering run is claimed. Four-bit loading is a candidate for T4, not a measured memory guarantee.

## Consequences
An any-error AUROC cannot be described as a false-yes-only detector. Modern-model results have their own run identity, table and status. Failure, class insufficiency and negative outcomes are reported. Selection remains calibration/validation-only; test is never used to retune.
