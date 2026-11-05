### FILE: ADJUDICATION.md

policy: 2026-10-30-v1 [src/adjudication.py]
unit: complete answer to one audited image/question. [src/adjudication.py]
labels: correct | hallucinated | incorrect | unresolved; positive-control failures measure lost sightedness. [src/adjudication.py]
contradiction: an affirmed absent target fails even when another clause says empty; negation must scope the target. [src/adjudication.py]
automatic scoring: conservative triage, not a semantic oracle; mixed attributes, uncertainty, refusals and paraphrases require manual review. [src/adjudication.py]
review: review EVERY answer against pixels, prompt and complete response, blinded to method/split; append JSONL reviews with prediction_id, label, reviewer and reason. Never edit raw predictions. Missing reviews or unresolved answers prevent reportable metrics. [scripts/analyze_run.py]
hallucination rate: explicit hallucinated answers / all answers per category; incorrect answers are separately counted. [scripts/analyze_run.py]
sightedness: correct positive controls / all positive controls; also report paired steered retention among baseline-correct controls; zero denominator is null. [scripts/analyze_run.py]
dataset: originals preserved; exclusions in dataset/audit_2026-10-30.json. Surviving counter-attribute coverage cannot support a three-way stratified split. Do not infer category generalization. [dataset/audit_2026-10-30.json]
