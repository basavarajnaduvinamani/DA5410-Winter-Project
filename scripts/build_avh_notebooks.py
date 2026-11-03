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
    cells = [md(f"""
        # CounterPrior v2 — {model.title()}
        Independent continuation, November 2026. **REVIEW COPY — NOT RUN.**

        **Tonight's order**
        A) Account A: cells 1–4 with PREPARE_DATA=True → send inventory printout to owner.
        B) Accounts A and B in parallel: pre_smoke for Gemma and Qwen.
        C) Owner fills terms_review.json; lineage/object keys auto-generated; manifest + media.
        D) Formal smoke → calibration/validation passes (audit filled in parallel).
        E) Freeze → test passes → metrics.
        F) Account C only takes over a stalled notebook (same partition after the stale lock is cleared) or distinct stages.

        **HALT until the owner types GO in chat.** This file has no measured outcomes.
        One model per Colab T4 runtime; fp16 only. Read `docs/AVH_REVIEW.md` and
        `docs/PREREGISTRATION.md` before use. Terms gate bulk inference; unavailable
        lineage uses an explicitly disclosed released-clip grouping fallback.

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
        ## 1. Configure the isolated model environment
        Start a fresh runtime for this model. Dependency selection and real class-import
        verification happen in cell 3. Colab's CUDA PyTorch is preserved and logged.
        No FlashAttention-2 dependency (T4). PREPARE_DATA belongs to Account A only.
    """), code('''
        import os, subprocess, sys
        assert OWNER_RELEASE == "GO"
        os.environ["MPLBACKEND"] = "Agg"
        PREPARE_DATA = False  # Account A sets True after GO; B/C leave False.
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)
    '''), md("""
        ## 2. Mount shared Drive and select a work partition
        One writer per model/stage/partition. Normally A=Gemma, B=Qwen, workers=1.
        C runs distinct stages or takes over a stopped runtime's exact partition after
        its stale lock is cleared. Never run the same partition/stage concurrently.
        Keep workers fixed on resume. Default is one partition per model.
    """), code(f'''
        from google.colab import drive
        from pathlib import Path
        import importlib.util, json
        drive.mount("/content/drive")
        ROOT = Path("/content/drive/MyDrive/CounterPrior_v2/2026-11-03")
        MODEL = "{model}"
        FRAMES = 8  # If OOM: new shared study directory, FRAMES=1 for both models, repeat smoke.
        SCORING_MODE = "verified_prefix"  # Gemma fallback after saved failure: "full_forward".
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
        ## 3. Verify package release/class, resolve model revision and save license
        Print pip's real release list. For Gemma, choose the newest stable release
        whose versioned official model docs include Gemma4ForConditionalGeneration;
        then install and verify its real class import. Qwen checks its Thinker class
        on 4.57.1. Import errors propagate verbatim. Save the selection for restarts.
        Model metadata/license only at this step, no model weights.
    """), code('''
        import re, urllib.request, urllib.error
        from packaging.version import Version
        if "transformers" in sys.modules:
            raise RuntimeError("Fresh runtime required before dependency selection/install")
        indexed = subprocess.run([sys.executable, "-m", "pip", "index", "versions", "transformers"],
                                 check=True, text=True, stdout=subprocess.PIPE)
        print(indexed.stdout)
        selection_path = ROOT / MODEL / "transformers_selection.json"
        if selection_path.exists():
            selection = avh.read_json(selection_path)
            chosen = selection["version"]
            print("Reusing frozen package selection:", selection)
        elif MODEL == "gemma":
            match = re.search(r"Available versions:\\s*(.+)", indexed.stdout)
            if match is None:
                raise RuntimeError("Cannot parse pip release list; inspect actual output")
            versions = sorted({v.strip() for v in match[1].split(",") if not Version(v.strip()).is_prerelease},
                              key=Version, reverse=True)
            chosen = None
            for version in versions:
                docs_url = f"https://huggingface.co/docs/transformers/v{version}/en/model_doc/gemma4"
                try:
                    with urllib.request.urlopen(docs_url, timeout=30) as response:
                        docs = response.read().decode("utf-8")
                        resolved = response.geturl()
                except urllib.error.HTTPError as error:
                    if error.code == 404:
                        continue
                    raise
                if f"/v{version}/" in resolved and "Gemma4ForConditionalGeneration" in docs:
                    chosen = version
                    break
            if chosen is None:
                raise RuntimeError("No listed stable release has verified versioned Gemma 4 class docs")
            selection = {"version": chosen, "docs_url": docs_url, "class": "Gemma4ForConditionalGeneration"}
        else:
            chosen = "4.57.1"
            selection = {"version": chosen, "class": "Qwen2_5OmniThinkerForConditionalGeneration"}
        packages = [f"transformers=={chosen}", "accelerate==1.10.1", "numpy==2.2.5",
                    "scikit-learn==1.7.1", "Pillow==11.3.0", "av==16.0.1",
                    "soundfile==0.13.1", "librosa==0.11.0", "gdown==5.2.0", "joblib==1.5.2"]
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *packages], check=True)
        class_name = selection["class"]
        subprocess.run([sys.executable, "-c", f"import transformers; from transformers import {class_name}; "
                        f"print('Verified transformers', transformers.__version__, '{class_name}')"], check=True)
        avh.atomic_json(selection_path, selection)
        print("CHOSEN TRANSFORMERS:", chosen, "VERIFIED CLASS:", class_name)
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
                         workers=WORKERS, worker_index=WORKER_INDEX, scoring_mode=SCORING_MODE)
    '''), md("""
        ## 4. Download, inspect, and inventory the official subset (A only)
        Set PREPARE_DATA on exactly one runtime. Downloads are never inferred to be
        licensed. The extractor rejects traversal, archive symlinks and device files.
        Prints first three unfiltered records, every key/task name and counts before
        normalization. If schema differs, set the explicit mapping from that printout;
        no field/task aliases are guessed. No model weights have been loaded.
    """), code('''
        QA_SCHEMA_MAPPING = None  # canonical -> actual field; None means canonical identity only.
        QA_RECORDS_KEY = None  # Explicit container key if JSON is a dict rather than a list.
        QA_TASK_MAPPING = None  # Explicit released task name -> canonical task name, if needed.
        if PREPARE_DATA:
            hashes = avh.download_release(ROOT, OWNER_RELEASE)
            avh.atomic_json(ROOT / "release_hashes.json", hashes)
            unpacked = avh.extract_release(ROOT)
            qa, inventory = avh.inventory_release(unpacked, QA_SCHEMA_MAPPING, QA_RECORDS_KEY, QA_TASK_MAPPING)
            avh.atomic_json(ROOT / "release_inventory.json", inventory)
            avh.atomic_json(ROOT / "released_qa.json", qa)
        elif (ROOT / "release_inventory.json").exists():
            print(json.dumps(avh.read_json(ROOT / "release_inventory.json"), indent=2))
        else:
            print("WAIT: Account A must prepare and inspect the official release")
    '''), md("""
        ## Early diagnostic smoke — before terms, lineage or manifest
        Run on A and B in parallel after the inventory is available. Select 20 raw
        QA rows from at least three videos, fixed seed, prepare only their media,
        load the model and run parity/generation/VRAM/fp16 checks. No analysis result.
        This diagnostic cannot replace the later formal manifest smoke.
        Qwen has a hard 1800-second load/check timebox; Gemma a 3600-second inactivity
        timeout touched after loading. Keep the returned scorer for the formal smoke.
    """), code('''
        raw_qa = avh.read_json(ROOT / "released_qa.json")
        scorer, pre_report = avh.pre_smoke(cfg, raw_qa, ROOT / "release" / "unpacked",
                                         scorer=globals().get("scorer"))
        print(json.dumps(pre_report, indent=2))
        print("Engineering diagnostic only:", ROOT / MODEL / "pre_smoke.json")
    '''), md("""
        ## 5. Terms gate — owner review while the early diagnostic runs
        Dataset decision: research_use_permitted OR owner decision
        no_licence_stated_private_research_eval. The latter authorizes private evaluation
        only: never public AVHBench audio, video or frames, only aggregate numbers and
        our own text. Save inspected README/terms evidence, hash, source URL and reviewer.
        The no-license decision needs no invented applicable license passage.
        Model entries still require saved license text, exact applicable passage and
        research_use_permitted. This gates bulk runs, not the early diagnostic.
    """), code('''
        terms_path = ROOT / "terms_review.json"
        if not terms_path.exists():
            template = {key: {"decision": "pending", "evidence_file": "", "sha256": "",
                        "applicable_passage": "", "source_url": "", "reviewer": ""}
                        for key in ("dataset", "gemma", "qwen")}
            avh.atomic_json(terms_path, template)
        avh.check_terms(ROOT, MODEL)  # Raises until real evidence is supplied.
        print("Terms gate passed for", MODEL)
    '''), md("""
        ## 6. Freeze manifest; blind swapped-item audit; shared media (A only)
        Auto-recover explicit source IDs/tagged filename origins. Otherwise group by
        released clip and record possible cross-split source overlap; no lineage STOP.
        Auto-derive object keys with the saved deterministic rule. Spot-check 50 in
        audit.csv (object_check=correct/flagged). Audit label decisions and object checks
        run in parallel with inference; completion is required only at analysis freeze.
    """), code('''
        if PREPARE_DATA:
            qa = avh.read_json(ROOT / "released_qa.json")
            lineage = avh.auto_lineage(ROOT, qa, ROOT / "release" / "unpacked")
            manifest = avh.make_manifest(ROOT, qa, lineage)
            print("Audit worksheet:", avh.create_audit(ROOT, manifest))
            print("Lineage limitations:", manifest["limitations"])
            print("Splits:", {s: sum(r["split"] == s for r in manifest["rows"])
                              for s in ("calibration", "validation", "test")})
            index_path = ROOT / f"media_index_f{FRAMES}.json"
            if not index_path.exists():
                avh.prepare_media(ROOT, ROOT / "release" / "unpacked", manifest, frames=FRAMES)
        manifest = avh.read_json(ROOT / "manifest.json")
        index = avh.read_json(ROOT / f"media_index_f{FRAMES}.json")
        print("Analysis-only exclusions completed so far:", len(avh.check_audit(ROOT, manifest)))
        print(json.dumps(avh.read_json(ROOT / "audit_summary.json"), indent=2))
    '''), md("""
        ## 7. Formal manifest-based smoke gate (reuse early-loaded model)
        Nothing below is established by local CPU checks. Model loading may fail T4
        memory, fp16 numerical stability, cache compatibility, or token/generation
        agreement. Stop with the actual failure. Qwen load+smoke is timeboxed to
        30 minutes; a timeout prints stacks and ends the kernel. Gemma also has a load
        watchdog is 3600 seconds, touched after any model load. Never silently switch precision, offload, library, or evidence size.
        A validated one-frame fallback is a new study configuration for both models.
    """), code('''
        import numpy as np
        import time
        started = time.monotonic()
        timeout = 1800 if MODEL == "qwen" else 3600
        with avh.Watchdog(timeout, deadline=started+1800 if MODEL == "qwen" else None) as dog:
            if "scorer" not in globals():
                scorer = avh.OmniScorer(cfg)
            dog.touch()
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
        Inference scores ALL rows; audit exclusions apply only to CPU analysis.
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
        reused, never overwritten. Complete the audit and object spot-check now;
        its hash is frozen separately from inference journals. Underpowered cells stay explicit.
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
