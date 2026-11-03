"""Build the two review-only Colab notebooks. Does not import/run model code."""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(text).strip() + "\n"}


def build(model):
    version = "5.20.2" if model == "gemma" else "4.57.1"
    cells = [md(f"""
        # CounterPrior v2 — {model.title()}
        Independent continuation, November 2026. **REVIEW COPY — NOT RUN.**

        **HALT until the owner types GO in chat.** This file has no measured outcomes.
        One model per Colab T4 runtime; fp16 only. Read `docs/AVH_REVIEW.md` and
        `docs/PREREGISTRATION.md` before use. Dataset terms and source provenance
        can block the study; do not fill missing lineage with released numeric IDs.

        After GO, upload the reviewed `avh_common.py` and `PREREGISTRATION.md` into
        `MyDrive/CounterPrior_v2/2026-11-03/reviewed/`. Both notebooks must use the
        identical copies. Account A prepares data once; Account B uses those artifacts.
        Use a new study folder for any changed evidence format, code, or protocol.
        Attach a free T4. Do not run both model notebooks in the same runtime.
        Runtime outputs, media, licenses and audit remain in the private Drive folder.
        No automatic site changes, uploads to public hosting, PR, or deployment.
    """), code('''
        OWNER_RELEASE = ""  # Set to GO only after the owner has typed GO in chat.
        if OWNER_RELEASE != "GO":
            raise RuntimeError("HALT for owner review; do not run subsequent cells")
    '''), md("""
        ## 1. Install isolated model environment
        This cell is idempotent. Start a fresh runtime if a different model environment
        was previously imported. Colab's installed CUDA PyTorch is preserved and logged;
        the model class, fp16 tensors, GPU residence, and package set are checked later.
        No FlashAttention-2 dependency (T4). The shared module is imported only after install.
    """), code(f'''
        import os, subprocess, sys
        assert OWNER_RELEASE == "GO"
        os.environ["MPLBACKEND"] = "Agg"
        if "transformers" in sys.modules:
            raise RuntimeError("Fresh runtime required before installing model dependencies")
        packages = ["transformers=={version}", "accelerate==1.10.1", "numpy==2.2.5",
                    "scikit-learn==1.7.1", "Pillow==11.3.0", "av==16.0.1",
                    "soundfile==0.13.1", "librosa==0.11.0", "gdown==5.2.0", "joblib==1.5.2"]
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *packages], check=True)
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)
    '''), md("""
        ## 2. Mount shared Drive and select a work partition
        One writer per model/stage/partition. Normally A=Gemma, B=Qwen, workers=1.
        To use C for the same model, assign it different stages, or decide workers=2
        **before starting that model** and use indices 0/1. Keep workers fixed on resume.
        Account C cannot append to the same journal as A or B.
    """), code(f'''
        from google.colab import drive
        from pathlib import Path
        import importlib.util, json
        drive.mount("/content/drive")
        ROOT = Path("/content/drive/MyDrive/CounterPrior_v2/2026-11-03")
        MODEL = "{model}"
        FRAMES = 8  # If OOM: new shared study directory, FRAMES=1 for both models, repeat smoke.
        WORKERS, WORKER_INDEX = 1, 0
        ROOT.mkdir(parents=True, exist_ok=True)
        source = ROOT / "reviewed" / "avh_common.py"
        prereg = ROOT / "reviewed" / "PREREGISTRATION.md"
        assert source.is_file() and prereg.is_file(), "Copy reviewed runtime + preregistration to shared Drive first"
        spec = importlib.util.spec_from_file_location("avh_common", source)
        avh = importlib.util.module_from_spec(spec)
        sys.modules["avh_common"] = avh
        spec.loader.exec_module(avh)
        avh.require_go(OWNER_RELEASE)
        target = ROOT / "PREREGISTRATION.md"
        if target.exists() and avh.file_hash(target) != avh.file_hash(prereg):
            raise RuntimeError("Preregistration changed; use a new study folder")
        target.write_bytes(prereg.read_bytes())
        print("Shared study:", ROOT, "runtime SHA256:", avh.file_hash(source))
    '''), md("""
        ## 3. Resolve exact model revision and inspect its license
        Saves an immutable revision and actual license text. The existing revision
        is reused on restart. Qwen 3B has a research license; do not describe it as
        Apache-2.0. This cell fetches metadata/license only, not weights.
    """), code('''
        metadata_path = ROOT / MODEL / "model_metadata.json"
        if metadata_path.exists():
            metadata = avh.read_json(metadata_path)
        else:
            metadata = avh.resolve_model(MODEL)
            avh.atomic_json(metadata_path, metadata)
        license_path = ROOT / f"{MODEL}_LICENSE.txt"
        license_path.write_text(metadata["license_text"], encoding="utf-8")
        print(metadata["repo"], metadata["revision"], "license:", license_path)
        print(metadata["license_text"])
        cfg = avh.Config(str(ROOT), MODEL, metadata["revision"], frames=FRAMES,
                         workers=WORKERS, worker_index=WORKER_INDEX)
    '''), md("""
        ## 4. Download, inspect, and inventory the official subset (A only)
        Set PREPARE_DATA on exactly one runtime. Downloads are never inferred to be
        licensed. The extractor rejects traversal, archive symlinks and device files.
        Inspect all README/license/terms files. The release must be usable for research;
        unresolved or unusable terms mean STOP. No model weights have been loaded.
    """), code('''
        PREPARE_DATA = False  # Account A sets True after GO; B/C leave False.
        if PREPARE_DATA:
            hashes = avh.download_release(ROOT, OWNER_RELEASE)
            avh.atomic_json(ROOT / "release_hashes.json", hashes)
            unpacked = avh.extract_release(ROOT)
            qa, inventory = avh.inventory_release(unpacked)
            avh.atomic_json(ROOT / "release_inventory.json", inventory)
            avh.atomic_json(ROOT / "released_qa.json", qa)
        elif (ROOT / "release_inventory.json").exists():
            print(json.dumps(avh.read_json(ROOT / "release_inventory.json"), indent=2))
        else:
            print("WAIT: Account A must prepare and inspect the official release")
    '''), md("""
        ## 5. Terms and lineage gate — deliberate manual review
        Fill `terms_review.json` from actual saved text, never a guessed license.
        Each dataset/model entry needs decision=research_use_permitted, evidence_file
        (relative to ROOT), sha256, exact applicable_passage, source_url, reviewer.
        Dataset media may have separate upstream terms. If no usable terms can be
        established, **STOP and report**. Permission to publish example media is separate.

        `lineage.json` is a list: video_id (string), source_ids (all original audio
        and video origins), is_swapped (boolean), provenance (where mapping came from),
        object_by_question (exact question -> audited object/event key).
        Unknown mapping is a stop, not permission to treat clip IDs as independent.
        The blank templates below are deliberately invalid until reviewed.
    """), code('''
        terms_path = ROOT / "terms_review.json"
        if not terms_path.exists():
            template = {key: {"decision": "pending", "evidence_file": "", "sha256": "",
                        "applicable_passage": "", "source_url": "", "reviewer": ""}
                        for key in ("dataset", "gemma", "qwen")}
            avh.atomic_json(terms_path, template)
        if not (ROOT / "lineage.json").exists() and (ROOT / "released_qa.json").exists():
            qa = avh.read_json(ROOT / "released_qa.json")
            vids = sorted({str(r["video_id"]) for r in qa})
            template = [{"video_id": v, "source_ids": [], "is_swapped": None,
                         "provenance": "", "object_by_question": {r["text"]: "" for r in qa
                          if str(r["video_id"]) == v and r["task"] in avh.TASKS}} for v in vids]
            avh.atomic_json(ROOT / "lineage.json", template)
        avh.check_terms(ROOT, MODEL)  # Raises until real evidence is supplied.
        print("Terms gate passed for", MODEL)
    '''), md("""
        ## 6. Freeze manifest; blind swapped-item audit; shared media (A only)
        Author the lineage record from verified source mapping first. This cell never
        guesses it. Both models read exactly the same manifest and frame/audio bytes.
        Audit sheet contains no model outputs. Complete decision/reviewer/note, listening
        for off-screen sounds and checking the released label. Invalid/uncertain audited
        QA is excluded and preserved in the report. Error rate is reported with a CI.
    """), code('''
        if PREPARE_DATA:
            qa = avh.read_json(ROOT / "released_qa.json")
            lineage = avh.read_json(ROOT / "lineage.json")
            manifest = avh.make_manifest(ROOT, qa, lineage)
            print("Audit worksheet:", avh.create_audit(ROOT, manifest))
            print("Splits:", {s: sum(r["split"] == s for r in manifest["rows"])
                              for s in ("calibration", "validation", "test")})
            index_path = ROOT / f"media_index_f{FRAMES}.json"
            if not index_path.exists():
                avh.prepare_media(ROOT, ROOT / "release" / "unpacked", manifest, frames=FRAMES)
        manifest = avh.read_json(ROOT / "manifest.json")
        index = avh.read_json(ROOT / f"media_index_f{FRAMES}.json")
        print("Audited exclusions:", len(avh.check_audit(ROOT, manifest)))
        print(json.dumps(avh.read_json(ROOT / "audit_summary.json"), indent=2))
    '''), md("""
        ## 7. Load fp16 model and mandatory smoke gates
        Nothing below is established by local CPU checks. Model loading may fail T4
        memory, fp16 numerical stability, cache compatibility, or token/generation
        agreement. Stop with the actual failure. Qwen load+smoke is timeboxed to
        30 minutes; a timeout prints stacks and ends the kernel. Gemma also has a load
        watchdog. Never silently switch precision, offload, library, or evidence size.
        A validated one-frame fallback is a new study configuration for both models.
    """), code('''
        import numpy as np
        import time
        started = time.monotonic()
        with avh.Watchdog(1800):
            scorer = avh.OmniScorer(cfg)
            smoke = avh.smoke_test(scorer, cfg, manifest, index, OWNER_RELEASE)
        seconds = time.monotonic() - started
        print("LOAD + SMOKE seconds:", seconds)
        if MODEL == "qwen" and seconds > 1800:
            raise RuntimeError("Qwen timebox exceeded; stop this family")
        weights = scorer.model.get_output_embeddings().weight
        yes = weights[scorer.tokens["yes"]].float().mean(0)
        no = weights[scorer.tokens["no"]].float().mean(0)
        unembedding = (yes-no).detach().cpu().numpy()
        np.save(ROOT / MODEL / "yes_no_unembedding.npy", unembedding)
        avh.atomic_json(ROOT / MODEL / "environment.json", scorer.env)
        print("Smoke passed. Peak VRAM bytes:", max(r["peak_vram_bytes"] for r in smoke["checks"]))
    '''), md("""
        ## 8. Calibration + validation passes, each restartable
        Every committed JSONL question is flushed immediately. A temporary per-question
        fp16 NPZ precedes the row; derived feature shards compact each 200 questions.
        Logs show measured seconds/question and peak VRAM every 100 new rows.
        Default stages below cover all passes. Spare account C can take distinct stages;
        do not concurrently run the same partition/stage. When a dead runtime leaves a
        lock, confirm it is stopped before manually removing only that stale lock file.
    """), code('''
        from dataclasses import replace
        STAGES = list(avh.PASSES)  # Assign disjoint subsets if a spare runs this model.
        for stage in STAGES:
            stage_cfg = replace(cfg, stage=stage)
            print(avh.run_stage(scorer, stage_cfg, manifest, index, OWNER_RELEASE,
                                splits=("calibration", "validation")))
    '''), md("""
        ## 9. CPU fit and validation freeze — before any test inference
        Strict completeness checks require every planned calibration/validation pass,
        including work from C. All fits/statistics are CPU from saved logits/features.
        This saves transforms, layers, C, thresholds and hashes. Existing freezes are
        reused, never overwritten. Underpowered probe cells are recorded honestly.
    """), code('''
        freeze = avh.freeze_analysis(ROOT, cfg, manifest, index, scorer.env, unembedding)
        print(json.dumps(freeze, indent=2))
    '''), md("""
        ## 10. Held-out passes (explicit switch, after frozen choices)
        This is a protocol gate, not another request for chat permission after GO.
        No test rows may be used to revise the prompt, tokenizer mapping, layer,
        threshold, preprocessing, exclusions, or scope. Any such change needs a new study.
    """), code('''
        RUN_HELD_OUT = False  # Set True only after cell 9 successfully freezes choices.
        if RUN_HELD_OUT:
            for stage in STAGES:
                print(avh.run_stage(scorer, replace(cfg, stage=stage), manifest, index,
                                    OWNER_RELEASE, splits=("test",)))
        else:
            print("Held-out inference remains closed")
    '''), md("""
        ## 11. Final CPU metrics export — complete test only
        `metrics.json` is created only from complete matching artifacts. No placeholder
        result values. Missing/undefined CIs stay null; false-yes count <40 is always
        underpowered. Verdicts are per task. No production or assistive-safety claim.
        Code does not publish anything or alter the existing site.
    """), code('''
        if RUN_HELD_OUT:
            metrics = avh.evaluate_test(ROOT, cfg, manifest, index, scorer.env, OWNER_RELEASE)
            print("Measured metrics:", ROOT / MODEL / "metrics.json")
            print(json.dumps(metrics, indent=2))
        else:
            print("No test results; metrics.json has not been generated")
    '''), md("""
        ## 12. Optional secondary transfer + geometry (first analysis cut)
        Expensive CPU bootstraps; run only if the schedule permits. Frozen-layer
        transfer and calibration-only geometry are secondary, never main-layer selection.
        No completed metric is fabricated if this part is skipped.
    """), code('''
        RUN_SECONDARY = False
        if RUN_SECONDARY and RUN_HELD_OUT:
            secondary = avh.geometry_and_transfer(ROOT, cfg, manifest, index, scorer.env, unembedding)
            print("Secondary metrics:", ROOT / MODEL / "secondary_metrics.json")
        else:
            print("Secondary transfer/geometry: not run")
    '''), md("""
        **End of review artifact. STOP.** The authoring task ends with these notebooks;
        no experiment runs are authorized until owner GO. After actual runs, report
        measured complete outcomes and their uncertainty, including negative findings.
        Site work, draft PR and any deployment belong to the later phase of the brief.
    """)]
    for i, cell in enumerate(cells):
        cell["id"] = f"{model}-{i:02d}"
        cell["source"] = cell["source"].splitlines(keepends=True)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}, "accelerator": "GPU",
        "colab": {"name": f"avh_{model}.ipynb", "provenance": [], "gpuType": "T4"}},
        "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks" / f"avh_{model}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    for family in ("gemma", "qwen"):
        print(build(family))
