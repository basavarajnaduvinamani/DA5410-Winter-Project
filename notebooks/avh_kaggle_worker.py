"""Fresh-process GPU worker. CUDA_VISIBLE_DEVICES is set by the notebook parent.

No account login automation. No test statistics. One model instance on local cuda:0.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import avh_common as avh
import avh_kaggle as kg


def guard_empty_sdpa_queries(mask_factory, bool_dtype):
    """Keep fully masked padding queries finite so they cannot contaminate real tokens."""
    def guarded(*args, **kwargs):
        mask = mask_factory(*args, **kwargs)
        if mask is not None and mask.dtype == bool_dtype:
            return mask | ~mask.any(dim=-1, keepdim=True)
        return mask
    return guarded


class BatchScorer:
    """Fresh full forwards with native processor padding; never prefix-cache reuse."""
    def __init__(self, scorer):
        self.scorer = scorer
        self.torch = scorer.torch
        scorer.processor.tokenizer.padding_side = "left"

    def score(self, row, names, index, diagnostic=False):
        import numpy as np
        torch, s = self.torch, self.scorer
        torch.cuda.synchronize()
        tick = time.monotonic()
        texts, images, image_groups, audios = [], [], [], []
        for name in names:
            ims, audio, _ = s.media(row, name, index)
            content = ([{"type": "image"} for _ in ims] + [{"type": "audio"}]) if ims else []
            content += [{"type": "text", "text": "\nQuestion: " + row["question"] + "\nAnswer Yes or No only."}]
            texts.append(s.processor.apply_chat_template([{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False))
            # Gemma validates one image group per text row, including empty groups
            # for no-media rows. A flat image list means ONE sample, not B samples.
            image_groups.append(list(ims) if ims else [])
            if ims:
                images.extend(ims)
                audios.append(audio)
        args = {"text": texts, "padding": True, "return_tensors": "pt"}
        if images:
            args.update(images=image_groups if s.cfg.model == "gemma" else images,
                        audio=audios, sampling_rate=16000)
        if s.cfg.model == "qwen":
            args["use_audio_in_video"] = False
        encoded = s.processor(**args)
        inputs = {k: v.to(device="cuda", dtype=torch.float16 if v.is_floating_point() else v.dtype)
                  for k, v in encoded.items() if isinstance(v, torch.Tensor)}
        if s.cfg.model == "qwen":
            lengths = inputs.get("feature_attention_mask")
            lengths = lengths.sum(-1) if lengths is not None else None
            positions, _ = s.model.get_rope_index(inputs["input_ids"], inputs.get("image_grid_thw"),
                None, inputs["attention_mask"], False, lengths, None)
            inputs["position_ids"] = positions
            if hasattr(s.model, "rope_deltas"):
                s.model.rope_deltas = None
        else:
            inputs["position_ids"] = (inputs["attention_mask"].long().cumsum(-1) - 1).clamp_min(0)
        need_hidden = diagnostic or "full" in names
        with torch.inference_mode():
            output = s.model(**inputs, use_cache=False, return_dict=True, output_hidden_states=need_hidden)
        logits = output.logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Nonfinite fp16 logits for passes {names}; padding={s.processor.tokenizer.padding_side}")
        hidden = torch.stack([h[:, -1].float() for h in output.hidden_states], dim=1) if need_hidden else None
        if hidden is not None and (not torch.isfinite(hidden).all() or not torch.isfinite(hidden.half()).all()):
            raise FloatingPointError("Nonfinite/overflowed fp16 hidden states")
        yes = torch.logsumexp(logits[:, s.tokens["yes"]], -1)
        no = torch.logsumexp(logits[:, s.tokens["no"]], -1)
        result = []
        for i, name in enumerate(names):
            margin = float(yes[i] - no[i])
            result.append({"pass": name, "margin": margin, "answer": int(margin > 0),
                "yes_logsumexp": float(yes[i]), "no_logsumexp": float(no[i]),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "hidden": hidden[i].cpu().numpy() if hidden is not None and (name == "full" or diagnostic) else None,
                "logits": logits[i].cpu().numpy() if diagnostic else None})
        del output, inputs, encoded, logits, hidden, yes, no
        torch.cuda.synchronize()
        elapsed = time.monotonic() - tick
        return result, elapsed


def check_pair(reference, candidate):
    import numpy as np
    ld = float(np.max(np.abs(reference["logits"] - candidate["logits"])))
    hd = float(np.max(np.abs(reference["hidden"] - candidate["hidden"])))
    if ld > .02 or hd > .02 or reference["answer"] != candidate["answer"]:
        raise RuntimeError(f"Batch/fresh parity failed: logits={ld}, hidden={hd}")
    return {"logits_max_abs_diff": ld, "hidden_max_abs_diff": hd}


def question_batches(batch, row, index, size, names=None, diagnostic=False):
    names = list(names or kg.passes_for(row))
    result, timings = [], []
    for start in range(0, len(names), size):
        part = names[start:start + size]
        outputs, seconds = batch.score(row, part, index, diagnostic=diagnostic)
        result.extend(outputs)
        timings.append({"passes": part, "seconds": seconds, "batch_size": len(part)})
    return result, timings


def startup_smoke(batch, calibration, index, worker_id):
    """Check full passes on ten questions and all required passes on five."""
    s, torch = batch.scorer, batch.torch
    serial = s.cfg.model == "gemma"
    items = sorted(calibration, key=lambda r: (r["task"] == avh.TASKS[2],
                   avh.digest([avh.SEED, "worker-smoke", worker_id, r["id"]])))[:10]
    if len(items) != 10:
        raise RuntimeError("Need 10 calibration items for the worker smoke")
    repeat, references = [], {}
    for i, row in enumerate(items):
        first, _ = batch.score(row, ["full"], index, diagnostic=True)
        second, _ = batch.score(row, ["full"], index, diagnostic=True)
        repeat.append(check_pair(first[0], second[0]))
        if i < 10:
            generated = s.generation_check(row, index)
            if generated["generated_answer"] != first[0]["answer"]:
                raise RuntimeError("Generate/scoring disagreement")
        if i < 5:
            refs = {}
            for name in kg.passes_for(row):
                refs[name] = batch.score(row, [name], index, diagnostic=True)[0][0]
            references[row["id"]] = refs
    candidates, accepted = {}, []
    for size in ((1,) if serial else range(1, 6)):
        try:
            diffs = []
            for row in items[:5]:
                outputs, _ = question_batches(batch, row, index, size, diagnostic=True)
                diffs.extend(check_pair(references[row["id"]][r["pass"]], r) for r in outputs)
            candidates[str(size)] = {"status": "passed", "comparisons": len(diffs),
                "max_logits_diff": max(r["logits_max_abs_diff"] for r in diffs),
                "max_hidden_diff": max(r["hidden_max_abs_diff"] for r in diffs)}
            accepted.append(size)
        except torch.cuda.OutOfMemoryError:
            if serial:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            candidates[str(size)] = {"status": "out_of_memory"}
    if not accepted:
        raise RuntimeError("No fitting parity-verified batch size")
    return {"passed": True, "n_questions": 10, "generation_items": 10,
        "reference_items": 5, "reference_kind": "fresh_unbatched_gpu",
        "prefix_cache_verified": False, "comparison_kind": (
            "serial_full_forward_repeatability" if serial else "full_forward_repeatability_and_batch_parity"),
        "batch_size": max(accepted), "validated_batch_sizes": accepted,
        "repeatability": repeat, "candidates": candidates}


def stable_environment(scorer):
    """Hash numerical runtime, retain the complete raw package list as separate provenance."""
    import importlib.metadata
    names = ("torch", "transformers", "numpy", "Pillow", "av", "soundfile", "librosa", "accelerate")
    result = {k: scorer.env[k] for k in ("python", "torch", "transformers", "gpu", "cuda", "attention_backend")}
    result["packages"] = {name: importlib.metadata.version(name) for name in names}
    result.update(scoring_mode="full_forward", fp16_reduced_precision_reduction=
                  scorer.torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    return result


def worker(job):
    # Import torch only here, after this fresh process has its GPU environment.
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(job["local_gpu"]):
        raise RuntimeError("Worker GPU assignment mismatch")
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Exactly one visible T4 per worker required")
    if job["reference"] != "fresh_unbatched_gpu":
        raise RuntimeError("Reference decision pending")
    dataset, out = Path(job["dataset"]), Path(job["output"])
    out.mkdir(parents=True, exist_ok=True)
    bundle = avh.read_json(dataset / "kaggle_bundle.json")
    verified = avh.read_json(job["cpu_verified"])
    if verified["status"] != "cpu_verified" or verified["bundle_sha256"] != avh.digest(bundle):
        raise RuntimeError("Run the CPU bundle verification first")
    if verified["media_index_sha256"] != avh.digest(avh.read_json(dataset / "kaggle_media_index.json")):
        raise RuntimeError("Relocatable media index changed after CPU verification")
    model = job["model"]
    if model == "qwen" and (not job.get("gemma_complete") or job.get("qwen_attempt") != 1):
        raise RuntimeError("Qwen runs only once on spare workers after Gemma completes")
    manifest = avh.read_json(job["manifest"])
    if manifest["model"] != model or manifest["parent_manifest_sha256"] != bundle["parent_manifest_sha256"]:
        raise RuntimeError("Wrong sample/study family")
    if job["mode"] in ("run", "addon"):
        gate = job.get("approval", {})
        if (gate.get("status") != "approved" or gate.get("bundle_sha256") != avh.digest(bundle)
                or gate.get("n_workers") != job["n_workers"]):
            raise RuntimeError("Owner timing auto-gate is missing or mismatched")
    work = out / "study_evidence"
    work.mkdir(exist_ok=True)
    serial_amendment = Path(__file__).with_name("GEMMA_SERIAL_AMENDMENT.md")
    if model == "gemma":
        shutil.copyfile(serial_amendment, work / serial_amendment.name)
    for name in ("terms_review.json", "release_inventory.json", "gemma_LICENSE.txt", "qwen_LICENSE.txt"):
        shutil.copyfile(dataset / name, work / name)
    (work / model).mkdir(exist_ok=True)
    for name in ("model_metadata.json", "scoring_amendment.json"):
        shutil.copyfile(dataset / model / name, work / model / name)
    amendment = avh.read_json(dataset / model / "scoring_amendment.json")
    shutil.copyfile(dataset / model / amendment["prefix_evidence_file"], work / model / amendment["prefix_evidence_file"])
    avh.check_terms(work, model)
    meta = avh.read_json(dataset / model / "model_metadata.json")
    cfg = avh.Config(str(work), model, meta["revision"], k=2, workers=job["n_workers"],
        worker_index=job["worker_id"], scoring_mode="full_forward")
    avh.record_scoring_amendment(cfg)
    index = kg.resolved_index(dataset)
    started = time.monotonic()
    deadline = started + (1800 if model == "qwen" else 10.5 * 3600)
    calibration = [r for r in manifest["rows"] if r["split"] == "calibration"]
    with avh.Watchdog(3600, deadline=deadline) as dog:
        if model == "gemma":
            # This registry override lives only in the dedicated Gemma worker.
            from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
            masks = ALL_MASK_ATTENTION_FUNCTIONS
            masks["sdpa"] = guard_empty_sdpa_queries(masks["sdpa"], torch.bool)
        scorer = avh.OmniScorer(cfg)
        load_seconds = time.monotonic() - started
        avh.atomic_json(out / "raw_environment.json", scorer.env)
        batch = BatchScorer(scorer)
        dog.touch()
        smoke_start = time.monotonic()
        smoke = startup_smoke(batch, calibration, index, job["worker_id"])
        smoke_seconds = time.monotonic() - smoke_start
        if model == "qwen" and time.monotonic() - started > 1800:
            raise RuntimeError("Qwen one-attempt load/smoke gate exceeded; replication pending")
        avh.atomic_json(out / "smoke.json", smoke)
        dog.deadline = started + 10.5 * 3600
        dog.seconds = 600
        dog.touch()
        identity = {"schema": kg.SCHEMA, "bundle_sha256": avh.digest(bundle),
            "selection_family": {k: manifest[k] for k in ("parent_manifest_sha256", "group_assignment_sha256", "seed", "k")},
            "model": model, "model_revision": cfg.revision, "n_workers": cfg.workers,
            "code_git_hash": job.get("code_git_hash", bundle["code_git_hash"]), "code_dirty": job.get("code_dirty", True),
            "code_sha256": avh.digest({name: avh.file_hash(Path(__file__).parent / name)
                                       for name in ("avh_common.py", "avh_kaggle.py", "avh_kaggle_worker.py")}),
            "environment": stable_environment(scorer), "reference": job["reference"],
            "numerical_limit": .02, "scoring_mode": "full_forward",
            "sdpa_empty_query_guard": model == "gemma"}
        if model == "gemma":
            identity["serial_amendment_sha256"] = avh.file_hash(serial_amendment)
            identity["comparison_kind"] = smoke["comparison_kind"]
            identity["batch_size"] = smoke["batch_size"]
        if job["mode"] in ("timing", "auto"):
            # Engineering-only: 50 owner-account questions split 25/25 over its two GPUs.
            items = kg.stratified_take(calibration, 50, avh.SEED)
            items = items[job["local_gpu"]::2]
            timings = []
            for row in items:
                outputs, parts = question_batches(batch, row, index, smoke["batch_size"])
                write_start = time.monotonic()
                # Real fp16 serialization is timed; this diagnostic never enters the research journal.
                full = next(r for r in outputs if r["pass"] == "full")
                import numpy as np
                np.save(out / "timing_hidden.npy", full["hidden"].astype(np.float16))
                with (out / "timing_pairs.jsonl").open("a") as f:
                    for r in outputs:
                        f.write(json.dumps({k: v for k, v in r.items() if k not in ("hidden", "logits")}) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                write_seconds = time.monotonic() - write_start
                timings.append({"question_id": row["id"], "source_group": row["source_group"],
                    "task": row["task"], "batches": parts, "checkpoint_seconds": write_seconds,
                    "question_seconds": sum(p["seconds"] for p in parts) + write_seconds})
                dog.touch()
            avh.atomic_json(out / "timing.json", {"kind": "engineering_timing_not_research_results",
                "model": model, "worker_id": job["worker_id"], "identity": identity,
                "launch_id": job["launch_id"], "load_seconds": load_seconds, "smoke_seconds": smoke_seconds,
                "batch_size": smoke["batch_size"], "rows": timings})
            (out / "timing_hidden.npy").unlink(missing_ok=True)
        if job["mode"] == "auto":
            wait_start = time.monotonic()
            while not Path(job["gate_file"]).exists():
                if time.monotonic() - wait_start > 600:
                    raise RuntimeError("Timing aggregation did not produce an auto-gate within 10 minutes")
                time.sleep(1)
            gate = avh.read_json(job["gate_file"])
            if gate["status"] != "approved":
                raise RuntimeError("Timing auto-gate halted; inspect report before any bulk run")
            dog.touch()
        if job["mode"] != "timing":
            gate = avh.read_json(job["gate_file"]) if job["mode"] == "auto" else job["approval"]
            if gate["code_sha256"] != identity["code_sha256"]:
                raise RuntimeError("Worker code differs from the owner timing gate")
            if gate["identity_sha256"] != avh.digest(identity):
                raise RuntimeError("Worker data/revision/numerical environment differs from the owner timing gate")
            splits = ("calibration", "validation", "test")
            items = [r for r in manifest["rows"] if r["split"] in splits
                     and kg.worker_for(r["id"], cfg.workers) == cfg.worker_index]
            split_order = {"calibration": 0, "validation": 1, "test": 2}
            items.sort(key=lambda r: (split_order[r["split"]], r["video_id"], r["id"]))
            if job["mode"] == "addon":
                initial_ids = {r["id"] for r in avh.read_json(dataset / "lean_manifest_n2000.json")["rows"]}
                items = [r for r in items if r["id"] not in initial_ids]
            status = "complete"
            with kg.PairJournal(out / "predictions", identity, cfg.worker_index, job["account_index"], cfg.workers, manifest) as journal:
                receipt_written = False
                for row in items:
                    if not receipt_written and manifest["n"] == 2000:
                        receipt = kg.calibration_receipt(journal, manifest, cfg.worker_index, cfg.workers, job["launch_id"])
                        if receipt is not None:
                            avh.atomic_json(out / "calibration_receipt.json", receipt)
                            receipt_written = True
                    names = [name for name in kg.passes_for(row) if (row["id"], name) not in journal.done]
                    if not names:
                        continue
                    # Exit normally before Kaggle's session cap so output version can finish saving.
                    if time.monotonic() > started + 10 * 3600:
                        status = "checkpointed_session_budget"
                        break
                    outputs, _ = question_batches(batch, row, index, smoke["batch_size"], names=names)
                    for r in outputs:
                        journal.append(row, r["pass"], {k: v for k, v in r.items()
                            if k not in ("pass", "hidden", "logits")}, hidden=r["hidden"])
                    journal.question_complete()
                    if journal.dirty_questions == 0:
                        print(json.dumps({"worker_id": cfg.worker_index, "completed_pairs": len(journal.done)}), flush=True)
                    dog.touch()
                if not receipt_written and manifest["n"] == 2000:
                    receipt = kg.calibration_receipt(journal, manifest, cfg.worker_index, cfg.workers, job["launch_id"])
                    if receipt is not None:
                        avh.atomic_json(out / "calibration_receipt.json", receipt)
            avh.atomic_json(out / "completion.json", {"status": status, "mode": job["mode"],
                "manifest_sha256": avh.digest(manifest), "worker_id": cfg.worker_index,
                "test_analysis": "sealed until CPU FROZEN.json and owner HALT 2 approval"})
        weights = scorer.model.get_output_embeddings().weight
        direction = (weights[scorer.tokens["yes"]].float().mean(0) - weights[scorer.tokens["no"]].float().mean(0))
        import numpy as np
        np.save(out / "yes_no_unembedding.npy", direction.detach().cpu().numpy())
    print(json.dumps({"worker_id": job["worker_id"], "mode": job["mode"], "finished": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    worker(avh.read_json(args.job))
