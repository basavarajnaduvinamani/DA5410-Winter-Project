# CounterPrior

Do small multimodal models ground their answers in the image, audio or video they receive? CounterPrior tests this with object traps, input swaps, and probes that read model states before an answer.

LLaVA-1.5 began as an AI4Bharat internship project (Nov 2025–Feb 2026). CounterPrior-50 was released in September 2026. The verified POPE rerun and the Gemma 4 E2B and Qwen2.5-Omni-3B audio/video work are independent follow-ups from November 2026.

| Model and split | Result | 95% CI | Qualification |
|---|---|---|---|
| LLaVA-1.5 · POPE test, 1,800 questions | Accuracy 0.863. Probe AUROC 0.719 vs confidence 0.745 | Accuracy [0.840, 0.888] | Steering did not beat controls |
| Gemma 4 E2B · AVHBench test, sound | Probe ΔAUROC +0.027 | [−0.231, +0.290] | 15 false Yes, underpowered |
| Gemma 4 E2B · AVHBench test, video | Probe ΔAUROC −0.101 | [−0.217, +0.005] | 24 false Yes, underpowered |
| Qwen2.5-Omni-3B · AVHBench test, sound | Probe ΔAUROC +0.101 | [+0.008, +0.188] | 47 false Yes. Filter exceeded the recall cap |
| Qwen2.5-Omni-3B · AVHBench test, video | Probe ΔAUROC +0.004 | [−0.127, +0.130] | 26 false Yes, underpowered |

Exploratory Gemma video swaps moved sound-question margins 1.75× as much as audio swaps [1.51, 2.08]. Audio-only and Listen First missed the validation gate. Early pilot figures were withdrawn after a scorer audit. The 3.6% flip claim was withdrawn as evidence of sound removal because donors generally lacked verified absence labels. Released clip IDs substitute for missing source lineage. See the [release records](results/v2/README.md).

Reproduce with the [Kaggle guide](README_KAGGLE.md) and [preregistration](docs/PREREGISTRATION_KAGGLE.md). Obtain the release from [AVHBench’s maintainers](https://github.com/kaist-ami/AVHBench) and keep the study input private. Use the [POPE configuration](configs/pope_llava.json) for the image run. [Research site](https://counterprior.vercel.app).

Prior work: [AVHBench](https://arxiv.org/abs/2410.18325), [AVCD](https://arxiv.org/abs/2505.20862), [POPE](https://aclanthology.org/2023.emnlp-main.20/), [OmniDPO](https://ojs.aaai.org/index.php/AAAI/article/view/39104), [HALP](https://aclanthology.org/2026.eacl-long.287/), [REVE](https://arxiv.org/abs/2609.26028), [OmniHalluc-L](https://aclanthology.org/2026.emnlp-main.882/), [When Eyes and Ears Disagree](https://arxiv.org/abs/2511.10059).
