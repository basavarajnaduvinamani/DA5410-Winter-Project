# CounterPrior v2 release records

- [Kaggle preregistration](../../docs/PREREGISTRATION_KAGGLE.md)
- [Gemma serial scoring amendment](../../docs/GEMMA_SERIAL_AMENDMENT.md)
- [Gemma secondary comparison and freeze](gemma-FROZEN.json)
- [Gemma run approval](gemma-run-approval.json)
- [Gemma held-out summary](gemma-reported-test-summary.json) and [exploratory summary](gemma-exploratory-summary.json)
- [Qwen freeze](qwen-FROZEN.json), [evaluation receipt](qwen-test_evaluation.json), [metrics](qwen-metrics.json), and [raw modality comparison](qwen-raw_modality_results.json)
- [LLaVA verified metrics](../REPORT_metrics.json)

The Gemma held-out summary records rounded values from the reviewed evaluation. The original Gemma metrics JSON and evaluation receipt were unavailable in the local release workspace. The Gemma freeze is transcribed from the saved freeze output. The two worker snapshots preserve the runtime files used for scoring. The public Qwen notebook reads the private sample manifest from an input dataset.
