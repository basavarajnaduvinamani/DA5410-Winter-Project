# CounterPrior design plan

Subject: uncertainty in multimodal answers. Reader: a research lead or pre-doc interviewer with two minutes. Lead with a recorded answer change, then show the held-out outcomes.

| Token | Hex | Role |
| --- | --- | --- |
| Mist | `#F2F5FA` | Page background |
| White | `#FFFFFF` | Reading and chart surface |
| Ink | `#18243B` | Text |
| Slate | `#4C5971` | Supporting text and chart ticks |
| Signal blue | `#1947B8` | Selected control, links, focus and probe estimates |
| Blue wash | `#E1EBFF` | Hero instrument and active states |

Geist Sans carries the interface and headings, with tabular figures for measurements. Left-align the title, controls and prose. Keep the charts on a shared grid that stacks on mobile.

```text
CounterPrior                       Results / Explorer / Ablations / About

Research question                  Saved example
Short context and attribution       Image | Question + answer toggle
                                    Confidence | Probe score

Results       [LLaVA absolute AUROC] [Qwen difference + interval]
              Exploratory Gemma sentence

Method        How we fit and evaluated the probes
Limitations   What the record can support
Code          Repository and complete results
```

1. Make the evidence interactive before asking readers to interpret a metric.
2. Give the negative result the same space as the positive one.
3. Let measured values, missing fields and uncertainty look different.

## Review before building

The first sketch used a margin index and rules between chapters, which read like a generic broadsheet. Replace the index and rules with vertical space and one asymmetrical, blue-wash instrument. Use a cool sans palette in place of cream, serif and terracotta. Drop stat tiles, all-caps labels, numbered chapters, arrow links and motion. The controls do one job and expose exactly which saved observation is displayed.

The public site data records one changed answer under LLaVA steering, including the original probe score. It contains aggregate Gemma swap measurements but no public per-item swap with answer, confidence and probe score. Use the saved steering example and disclose unavailable scores. Keep the existing swap diagram with the complete results. Show the Qwen comparison against its recorded full baseline, which includes confidence.

## Review round

Lead with both measured results. Move the saved LLaVA example below them as a native disclosure, and load its image lazily. The charts use static SVG, and the explorer loads on navigation. Use one self-hosted font and omit the landing page's client bootstrap.
