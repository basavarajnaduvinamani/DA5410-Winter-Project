"""Private Kaggle transport and deterministic lean-study contracts. CPU import only.

The original Colab manifest/media are never overwritten. Prediction shards are
private research artifacts. A test prediction is not permission to analyze it.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import avh_common as avh

PASSES = ("full", "cf0", "cf1", "blank", "no_media")
SCHEMA = "counterprior-kaggle-v1"


def immutable_json(path, value):
    path = Path(path)
    if path.exists() and avh.read_json(path) != value:
        raise RuntimeError(f"Refusing to change frozen record: {path}")
    if not path.exists():
        avh.atomic_json(path, value)
    return value


def worker_for(question_id, n_workers):
    if n_workers < 1:
        raise ValueError("Positive worker count required")
    return int(avh.digest(str(question_id))[:16], 16) % n_workers


def account_workers(account_index, n_workers):
    if n_workers < 2 or n_workers % 2 or not 0 <= account_index < n_workers // 2:
        raise ValueError("Require two GPUs per account and a valid account index")
    return (account_index * 2, account_index * 2 + 1)


def passes_for(row):
    return ("full", "no_media") if row["task"] == avh.TASKS[2] else PASSES


def stratified_take(rows, n, seed):
    """Largest-remainder task x label allocation; fixed hash order within strata."""
    if n > len(rows) or n < 1:
        raise ValueError("Sample size exceeds the available source-disjoint pool")
    strata = collections.defaultdict(list)
    for row in rows:
        strata[(row["task"], row["label"])].append(row)
    target = {s: n * len(items) / len(rows) for s, items in strata.items()}
    counts = {s: int(value) for s, value in target.items()}
    order = sorted(strata, key=lambda s: (-(target[s] - counts[s]), s))
    for s in order[:n - sum(counts.values())]:
        counts[s] += 1
    chosen = []
    for s in sorted(strata):
        chosen.extend(sorted(strata[s], key=lambda r: avh.digest([seed, "sample", r["id"]]))[:counts[s]])
    return sorted(chosen, key=lambda r: r["id"])


def lean_manifest(parent, n=2000, model="gemma", seed=avh.SEED):
    """New manifest; old split is retained as provenance, not silently overwritten.

    All released groups receive one fixed 20/20/60 assignment independent of n.
    Sampling happens inside each group-disjoint pool. Same-hash donor rule is
    recomputed within these new pools, using the first two of four seeded donors.
    """
    import numpy as np
    if model not in avh.MODEL_IDS or n not in (1000, 2000, 3000):
        raise ValueError("Only the requested Gemma/Qwen lean sizes are supported")
    if (model == "gemma" and n not in (2000, 3000)) or (model == "qwen" and n != 1000):
        raise ValueError("Gemma n=2000/3000; optional Qwen n=1000")
    rows = parent["rows"]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Parent manifest contains duplicate question IDs")
    groups = sorted({r["source_group"] for r in rows})
    order = np.random.default_rng(seed).permutation(groups).tolist()
    a, b = int(.2 * len(order)), int(.4 * len(order))
    assignment = {g: "calibration" if i < a else "validation" if i < b else "test"
                  for i, g in enumerate(order)}
    sizes = {"calibration": n // 5, "validation": n // 5, "test": n * 3 // 5}
    selected = []
    for split, count in sizes.items():
        pool = [r for r in rows if assignment[r["source_group"]] == split]
        selected.extend(dict(r, parent_split=r["split"], split=split)
                        for r in stratified_take(pool, count, seed))
    video_meta = {r["video_id"]: r for r in rows}
    for row in selected:
        candidates = sorted(v for v, meta in video_meta.items()
                            if assignment[meta["source_group"]] == row["split"]
                            and meta["source_group"] != row["source_group"] and not meta["is_swapped"])
        if len(candidates) < 4:
            raise ValueError("Need four candidate donor clips in each new split")
        rng = np.random.default_rng(int(avh.digest([seed, row["video_id"], row["task"]])[:16], 16))
        row["donors"] = rng.choice(candidates, 4, replace=False).tolist()[:2]
    return {"schema": SCHEMA, "model": model, "n": n, "seed": seed, "k": 2,
        "parent_manifest_sha256": avh.digest(parent), "group_assignment_sha256": avh.digest(assignment),
        "split_counts": sizes, "test_scoring_policy": "sealed_precompute",
        "limitations": parent.get("limitations", []), "object_rule": parent.get("object_rule"),
        "sampling_rule": "group-disjoint 20/20/60 pools; proportional task x label largest remainder; seeded hash order",
        "donor_rule": "same-split released groups, seeded first two of four; unverified swap status explicitly retained",
        "rows": sorted(selected, key=lambda r: r["id"])}


def portable_index(index, media_root):
    """Never change media bytes or the source index; create a relocatable companion."""
    result = {}
    for vid, entry in index.items():
        directory = Path(entry["directory"])
        target = Path(media_root) / "media" / directory.parent.name / directory.name
        if not target.is_dir():
            raise FileNotFoundError(target)
        result[vid] = dict(entry, directory=target.relative_to(media_root).as_posix())
    return result


def resolved_index(dataset_root):
    root = Path(dataset_root).resolve()
    index = avh.read_json(root / "kaggle_media_index.json")
    result = {}
    for vid, entry in index.items():
        directory = (root / entry["directory"]).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Media path escapes the private dataset")
        result[vid] = dict(entry, directory=str(directory))
    return result


def export_private_study(root, output, support_dir, code_git_hash, code_dirty=True, read_workers=8, local_staging=None):
    """Run on CPU Colab. Zip prepared bytes, evidence, old results and new support.

    Does not delete/edit Drive files or upload anything. No tokens or HF cache.
    Bounded concurrent reads hide Drive's per-file latency. The ZIP is built on
    local CPU disk, then copied to Drive in one sequential transfer. A cleanly
    interrupted older export can contribute verified entries without re-reading
    them individually from Drive. Never use a GPU/TPU for this operation.
    """
    root, output, support = Path(root), Path(output), Path(support_dir)
    if output.exists():
        raise FileExistsError("Preserve the existing export; choose a new version filename")
    if len(code_git_hash) != 40 or any(c not in "0123456789abcdef" for c in code_git_hash):
        raise ValueError("Real source repository commit hash required")
    index = avh.read_json(root / "media_index_f8.json")
    pindex = portable_index(index, root)
    files, media_hashes = {}, {}
    required = ["manifest.json", "media_index_f8.json", "lineage.json", "terms_review.json",
        "release_inventory.json", "released_qa.json", "qa_resolution.json", "audit.csv"]
    for model in avh.MODEL_IDS:
        required += [f"{model}/model_metadata.json", f"{model}/scoring_amendment.json"]
    terms = avh.read_json(root / "terms_review.json")
    required.extend(entry["evidence_file"] for entry in terms.values())
    for name in sorted(set(required)):
        source = (root / name).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file():
            raise ValueError(f"Missing/unsafe study evidence: {name}")
        files[name] = source
    for model in avh.MODEL_IDS:
        amendment = avh.read_json(root / model / "scoring_amendment.json")
        name = f"{model}/{amendment['prefix_evidence_file']}"
        files[name] = root / name
        for name in ("pre_smoke.json", "smoke_f8.json", "environment.json", "yes_no_unembedding.npy"):
            path = root / model / name
            if path.is_file():
                files[f"prior_colab/{model}/{name}"] = path
    for vid, entry in pindex.items():
        for name, expected in entry["file_hashes"].items():
            source = root / entry["directory"] / name
            archive_name = f"{entry['directory']}/{name}"
            files[archive_name] = source
            media_hashes[archive_name] = expected
    prereg = support / "PREREGISTRATION_KAGGLE.md"
    if not prereg.is_file():
        raise FileNotFoundError(prereg)
    files["runtime/PREREGISTRATION_KAGGLE.md"] = prereg
    for name in ("PREREGISTRATION.md", "lineage_rules.json", "publication_policy.json", "shared_data_summary.json"):
        if (root / name).is_file():
            files[f"prior_colab/{name}"] = root / name
    manifest = avh.read_json(root / "manifest.json")
    generated = {"kaggle_media_index.json": pindex,
        "lean_manifest_n2000.json": lean_manifest(manifest, n=2000),
        "lean_manifest_n3000.json": lean_manifest(manifest, n=3000)}
    generated_bytes = {name: json.dumps(value, indent=2).encode("utf-8") for name, value in generated.items()}
    hashes = {name: media_hashes.get(name) or avh.file_hash(path) for name, path in files.items()}
    hashes.update({name: hashlib.sha256(value).hexdigest() for name, value in generated_bytes.items()})
    payload = {"schema": SCHEMA, "private_only": True, "code_git_hash": code_git_hash,
        "code_dirty": bool(code_dirty), "parent_manifest_sha256": avh.digest(manifest),
        "original_media_index_sha256": avh.digest(index),
        "files": hashes}
    if not 1 <= read_workers <= 16:
        raise ValueError("CPU read concurrency must be between 1 and 16")
    output.parent.mkdir(parents=True, exist_ok=True)
    local = Path(local_staging) if local_staging else (
        Path("/content/counterprior-export-cache") if Path("/content").is_dir() else output.parent / ".cpu-export-staging")
    local.mkdir(parents=True, exist_ok=True)
    temporary = local / (output.name + ".building")
    # Keep previous local/Drive partials intact; only final verified bytes are reused.
    candidates = [output.with_suffix(output.suffix + ".partial"), temporary]
    reusable = []
    for i, source in enumerate(candidates):
        if source.is_file() and zipfile.is_zipfile(source):
            copied = local / f"recovery-{i}-{output.name}"
            shutil.copyfile(source, copied)
            reusable.append(copied)
    complete, reused = set(), 0
    tick = time.monotonic()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for source in reusable:
            with zipfile.ZipFile(source) as previous:
                for info in previous.infolist():
                    if info.filename not in files or info.filename in complete:
                        continue
                    try:
                        data = previous.read(info)
                    except (zipfile.BadZipFile, RuntimeError, EOFError):
                        continue
                    if hashlib.sha256(data).hexdigest() != hashes[info.filename]:
                        continue
                    z.writestr(info.filename, data)
                    complete.add(info.filename)
                    reused += 1
        print(f"Reused {reused} verified files; CPU read workers={read_workers}. Building ZIP locally.", flush=True)

        def read_verified(item):
            name, path = item
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != hashes[name]:
                raise RuntimeError(f"Prepared bytes changed while exporting: {name}")
            return name, data

        remaining = iter((name, path) for name, path in sorted(files.items()) if name not in complete)
        pending = collections.deque()
        with concurrent.futures.ThreadPoolExecutor(max_workers=read_workers) as pool:
            for _ in range(read_workers * 2):
                item = next(remaining, None)
                if item is not None:
                    pending.append(pool.submit(read_verified, item))
            while pending:
                name, data = pending.popleft().result()
                z.writestr(name, data)
                complete.add(name)
                if len(complete) % 500 == 0:
                    print(f"Packed {len(complete)}/{len(files)} files; elapsed={time.monotonic()-tick:.1f}s", flush=True)
                item = next(remaining, None)
                if item is not None:
                    pending.append(pool.submit(read_verified, item))
        for name, data in generated_bytes.items():
            z.writestr(name, data)
        z.writestr("kaggle_bundle.json", json.dumps(payload, indent=2))
        z.writestr("cpu_verified.json", json.dumps({"schema": SCHEMA, "status": "cpu_verified",
            "bundle_sha256": avh.digest(payload), "media_index_sha256": avh.digest(pindex),
            "verification": "Prepared bytes and evidence checked on CPU during private archive export"}, indent=2))
        z.writestr("run_control.json", json.dumps({"mode": "auto", "model": "gemma", "n": 2000,
            "n_workers": 6, "reference": "fresh_unbatched_gpu", "sample_policy": "calibration_addon",
            "gemma_complete": False, "qwen_attempt": 0}, indent=2))
        z.writestr("CHECKSUMS.sha256", "".join(f"{h}  {name}\n" for name, h in sorted(hashes.items())))
    local_sha = avh.file_hash(temporary)
    print("Local ZIP complete. Copying one archive to Drive...", flush=True)
    transfer = output.with_suffix(output.suffix + ".transferring")
    shutil.copyfile(temporary, transfer)
    transfer.replace(output)
    result = {"archive": str(output), "bytes": output.stat().st_size,
              "sha256": local_sha, "media_clips": len(index), "private_only": True,
              "cpu_read_workers": read_workers, "reused_files": reused,
              "export_method": "bounded parallel CPU reads, local ZIP, one Drive copy"}
    avh.atomic_json(output.with_suffix(".checksum.json"), result)
    return result


def verify_bundle(root):
    """CPU-only integrity pass before allocating GPUs. No Drive traversal at runtime."""
    root = Path(root)
    bundle = avh.read_json(root / "kaggle_bundle.json")
    if bundle.get("schema") != SCHEMA or bundle.get("private_only") is not True:
        raise ValueError("Wrong study bundle or public-media policy")
    for i, (name, expected) in enumerate(sorted(bundle["files"].items()), 1):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or avh.file_hash(path) != expected:
            raise RuntimeError(f"Bundle integrity failure: {name}")
        if i % 500 == 0:
            print(f"Verified {i}/{len(bundle['files'])} private files", flush=True)
    index = resolved_index(root)
    avh.verify_media(index)
    return {"schema": SCHEMA, "bundle_sha256": avh.digest(bundle),
            "media_index_sha256": avh.digest(avh.read_json(root / "kaggle_media_index.json")),
            "status": "cpu_verified", "private_only": True}


def safe_pairs(manifest, splits=None):
    return {(r["id"], p) for r in manifest["rows"] if splits is None or r["split"] in splits
            for p in passes_for(r)}


class PairJournal:
    """Per-worker pair checkpoints; full-only fp16 NPY shards of 250 questions.

    Buffers are fsynced every 25 completed questions and on normal/soft-stop exit.
    An abrupt VM loss can repeat up to 25 uncommitted questions. A saved Kaggle
    output version must be reattached on restart; /working is not a shared drive.
    """
    def __init__(self, directory, identity, worker_id, account_index, n_workers, manifest):
        self.directory = Path(directory)
        self.identity, self.worker_id, self.account_index = identity, worker_id, account_index
        self.n_workers, self.manifest = n_workers, manifest
        if worker_id not in account_workers(account_index, n_workers):
            raise ValueError("Worker ID is not assigned to this account")

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = self.directory / "worker.lock"
        fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        try:
            immutable_json(self.directory / "identity.json", self.identity)
            self.path = self.directory / "predictions.jsonl"
            self.rows, valid = [], 0
            lines = self.path.read_bytes().splitlines(keepends=True) if self.path.exists() else []
            for i, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if i != len(lines) - 1 or line.endswith(b"\n"):
                        raise RuntimeError("Corrupt journal interior")
                    break
                if row["identity_sha256"] != avh.digest(self.identity):
                    raise RuntimeError("Checkpoint identity changed")
                self.rows.append(row)
                valid += len(line)
            self.done = {(r["question_id"], r["pass"]) for r in self.rows}
            if len(self.done) != len(self.rows):
                raise RuntimeError("Duplicate question/pass pair in checkpoint")
            wanted = safe_pairs(self.manifest)
            if not self.done <= wanted or any(worker_for(q, self.n_workers) != self.worker_id for q, _ in self.done):
                raise RuntimeError("Checkpoint contains unassigned work")
            if self.path.exists():
                with self.path.open("r+b") as f:
                    f.truncate(valid)
                if valid and not self.path.read_bytes().endswith(b"\n"):
                    with self.path.open("ab") as f:
                        f.write(b"\n")
            self.full_count = sum(r["pass"] == "full" for r in self.rows)
            self.maps, self.dirty_questions = {}, 0
            self.handle = self.path.open("a", encoding="utf-8")
            return self
        except BaseException:
            self.lock.unlink(missing_ok=True)
            raise

    def append(self, row, pass_name, result, hidden=None):
        import numpy as np
        key = (row["id"], pass_name)
        if key in self.done:
            return
        if pass_name not in passes_for(row) or worker_for(row["id"], self.n_workers) != self.worker_id:
            raise ValueError("Unassigned question/pass")
        record = dict(result, question_id=row["id"], id=row["id"], split=row["split"],
            **{"pass": pass_name}, worker_id=self.worker_id, account_index=self.account_index,
            model=self.identity["model"], model_revision=self.identity["model_revision"],
            code_git_hash=self.identity["code_git_hash"], code_sha256=self.identity["code_sha256"],
            code_hash=self.identity["code_sha256"],
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            identity_sha256=avh.digest(self.identity))
        if pass_name == "full":
            h = np.asarray(hidden, dtype=np.float16)
            if h.ndim != 2 or not np.isfinite(h).all():
                raise ValueError("Full pass requires finite all-layer hidden states")
            number, slot = divmod(self.full_count, 250)
            name = f"full-hidden-{number:05d}.npy"
            if name not in self.maps:
                path = self.directory / name
                self.maps[name] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+",
                    dtype=np.float16, shape=(250, *h.shape))
            self.maps[name][slot] = h
            record.update(feature=name, feature_slot=slot,
                feature_value_sha256=hashlib.sha256(h.tobytes()).hexdigest())
            self.full_count += 1
        elif hidden is not None:
            raise ValueError("Hidden states may be saved only for the full pass")
        self.handle.write(json.dumps(record, allow_nan=False) + "\n")
        self.rows.append(record)
        self.done.add(key)

    def question_complete(self):
        self.dirty_questions += 1
        if self.dirty_questions >= 25:
            self.flush()

    def flush(self):
        for array in self.maps.values():
            array.flush()
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.dirty_questions = 0

    def __exit__(self, *args):
        try:
            self.flush()
            self.handle.close()
        finally:
            self.maps.clear()
            self.lock.unlink(missing_ok=True)


def validate_outputs(directories, manifest, identity, n_workers, splits=None):
    """Coverage-only merge gate. Never computes test outcomes or chooses settings."""
    import numpy as np
    wanted = safe_pairs(manifest, splits)
    found, records, locations = set(), [], set()
    lookup = {r["id"]: r for r in manifest["rows"]}
    hidden_shape = None
    for directory in map(Path, directories):
        if avh.read_json(directory / "identity.json") != identity:
            raise RuntimeError("Mixed runtime/study identity")
        for line in (directory / "predictions.jsonl").read_text().splitlines():
            r = json.loads(line)
            pair = (r["question_id"], r["pass"])
            if pair not in wanted or pair in found:
                raise RuntimeError("Duplicate or unexpected question/pass pair")
            if (r["identity_sha256"] != avh.digest(identity)
                    or worker_for(pair[0], n_workers) != r["worker_id"]
                    or r["worker_id"] not in account_workers(r["account_index"], n_workers)):
                raise RuntimeError("Worker assignment or checkpoint identity mismatch")
            expected = {"model": identity["model"], "model_revision": identity["model_revision"],
                        "code_git_hash": identity["code_git_hash"], "code_sha256": identity["code_sha256"],
                        "code_hash": identity["code_sha256"], "split": lookup[pair[0]]["split"]}
            if any(r.get(k) != value for k, value in expected.items()):
                raise RuntimeError("Per-row provenance/split differs from the frozen assignment")
            if not np.isfinite([r["margin"], r["yes_logsumexp"], r["no_logsumexp"]]).all():
                raise FloatingPointError("Nonfinite merged answer scores")
            if r["answer"] != int(r["margin"] > 0) or abs(r["margin"] - (r["yes_logsumexp"] - r["no_logsumexp"])) > 1e-5:
                raise RuntimeError("Saved answer/margin/logit disagreement")
            if datetime.datetime.fromisoformat(r["timestamp"]).tzinfo is None:
                raise RuntimeError("Timestamp requires a timezone")
            if r["pass"] == "full":
                feature = (directory / r["feature"]).resolve()
                if not feature.is_relative_to(directory.resolve()):
                    raise ValueError("Unsafe feature path")
                location = (str(feature), r["feature_slot"])
                if not isinstance(r["feature_slot"], int) or not 0 <= r["feature_slot"] < 250 or location in locations:
                    raise RuntimeError("Invalid or aliased feature slot")
                locations.add(location)
                h = np.load(feature, mmap_mode="r", allow_pickle=False)[r["feature_slot"]]
                if h.dtype != np.float16 or not np.isfinite(h).all() or hashlib.sha256(h.tobytes()).hexdigest() != r["feature_value_sha256"]:
                    raise RuntimeError("Feature data changed")
                if h.ndim != 2 or (hidden_shape is not None and h.shape != hidden_shape):
                    raise RuntimeError("Inconsistent all-layer hidden-state shape")
                hidden_shape = h.shape
            found.add(pair)
            records.append((directory, r))
    if found != wanted:
        raise RuntimeError(f"Incomplete coverage: {len(wanted - found)} missing pairs")
    return records


def timing_summary(reports, manifest, n_workers):
    """Observed timings + explicitly labelled planning forecasts, never metrics.json.

    Per-pass allocation divides a measured batch duration by its pass count; it is
    not an isolated-pass benchmark. Forecast assumes equal workers/no failures;
    compute-hours and end-to-end wall time remain distinct.
    """
    import numpy as np
    rows = [r for report in reports for r in report["rows"]]
    if len({r["question_id"] for r in rows}) != 50:
        raise RuntimeError("Timing check must cover 50 distinct calibration questions")
    identities = {avh.digest(r["identity"]) for r in reports}
    if len(identities) != 1:
        raise RuntimeError("Timing workers use different numerical environments")
    observed = {}
    for task in avh.TASKS:
        selected = [r for r in rows if r["task"] == task]
        if not selected:
            raise RuntimeError("Timing check missing a task stratum")
        values = [r["question_seconds"] for r in selected]
        groups = [r["source_group"] for r in selected]
        observed[task] = {"question_seconds_mean": avh.cluster_ci(values, groups, lambda x: float(x.mean())),
            "question_seconds_p50": avh.cluster_ci(values, groups, lambda x: float(np.quantile(x, .5))),
            "question_seconds_p90": avh.cluster_ci(values, groups, lambda x: float(np.quantile(x, .9)))}
    counts = collections.Counter(r["task"] for r in manifest["rows"])
    compute_seconds = sum(counts[t] * observed[t]["question_seconds_mean"]["estimate"] for t in counts)
    batch_seconds = [b["seconds"] for r in rows for b in r["batches"]]
    allocations = collections.defaultdict(list)
    for r in rows:
        for b in r["batches"]:
            for p in b["passes"]:
                allocations[p].append(b["seconds"] / len(b["passes"]))
    return {"kind": "planning_forecast_not_research_result", "n_questions": 50,
        "n_source_groups": len({r["source_group"] for r in rows}), "observed": observed,
        "batch_seconds": {"p50": float(np.quantile(batch_seconds, .5)), "p90": float(np.quantile(batch_seconds, .9)),
                          "n_batches": len(batch_seconds)},
        "allocated_pass_seconds": {p: {"p50": float(np.quantile(v, .5)), "p90": float(np.quantile(v, .9)),
                                              "n": len(v)} for p, v in allocations.items()},
        "batch_sizes": [r["batch_size"] for r in reports],
        "model_load_seconds_observed": [r["load_seconds"] for r in reports],
        "forecast": {"gpu_hours_scoring_only": compute_seconds / 3600,
                     "ideal_wall_hours_scoring_only": compute_seconds / (3600 * n_workers),
                     "n_workers": n_workers, "question_count": len(manifest["rows"]),
                     "assumptions": "representative task timing, all workers equally fast; excludes startup, retries, transfers, audit, analysis and site"},
        "warnings": ["Per-pass timings are allocations of batch wall time, not isolated-pass measurements",
                     "Task-specific timing CIs are empirical; the wall-time forecast has no validated end-to-end CI",
                     "No predicted accuracy, AUROC, false-yes rate or site result"]}


def restore_checkpoint(source, destination, worker_id, bundle_sha256):
    """Attach the participant's own previous private output version before resuming."""
    source, destination = Path(source), Path(destination)
    identity = avh.read_json(source / "identity.json")
    if identity["bundle_sha256"] != bundle_sha256:
        raise RuntimeError("Resume checkpoint belongs to another private data version")
    for line in (source / "predictions.jsonl").read_text().splitlines():
        if json.loads(line)["worker_id"] != worker_id:
            raise RuntimeError("Resume checkpoint belongs to another worker")
    if destination.exists():
        raise FileExistsError("Do not overwrite a live or existing checkpoint")
    destination.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.name != "worker.lock":
            shutil.copyfile(path, destination / path.name)


def calibration_power_decision(manifest, records, expanded, policy):
    """Predeclared one-time planning decision; never uses validation/test predictions."""
    if policy == "fixed_2000":
        return {"policy": policy, "chosen_n": 2000, "adaptation": "disabled by owner decision"}
    if policy not in ("adaptive_calibration_first", "calibration_addon"):
        raise RuntimeError("Sample policy pending owner decision")
    if manifest.get("n") != 2000:
        raise RuntimeError("The add-on decision is permitted once, from the initial n=2000 calibration")
    lookup = {r["id"]: r for r in manifest["rows"]}
    full = {}
    for _, result in records:
        if result["pass"] != "full":
            continue
        row = lookup[result["question_id"]]
        if row["split"] != "calibration":
            raise RuntimeError("Sample expansion decision cannot inspect validation/test predictions")
        if row["id"] in full:
            raise RuntimeError("Duplicate calibration question")
        full[row["id"]] = result
    required = {r["id"] for r in manifest["rows"] if r["split"] == "calibration"}
    if set(full) != required:
        raise RuntimeError("Complete calibration before one-time expansion decision")
    planning = {}
    for task in avh.TASKS[:2]:
        negatives = [r for r in manifest["rows"] if r["split"] == "calibration"
                     and r["task"] == task and r["label"] == 0]
        if not negatives:
            raise RuntimeError("Missing negative calibration stratum")
        rate = avh.cluster_ci([full[r["id"]]["answer"] for r in negatives],
            [r["source_group"] for r in negatives], lambda a: float(a.mean()))
        test_negatives = sum(r["split"] == "test" and r["task"] == task and r["label"] == 0
                             for r in manifest["rows"])
        planning[task] = {"calibration_false_yes_rate": rate,
            "test_negative_inventory": test_negatives,
            "expected_false_yes_planning_only": rate["estimate"] * test_negatives}
    enlarge = any(r["expected_false_yes_planning_only"] < 40 for r in planning.values())
    if enlarge and (expanded["n"] != 3000 or expanded["group_assignment_sha256"] != manifest["group_assignment_sha256"]):
        raise RuntimeError("Expanded sample must preserve the group assignment and n<=3000")
    return {"kind": "calibration_based_sample_planning_not_test_results", "policy": policy,
        "chosen_n": 3000 if enlarge else 2000, "max_adaptations": 1, "planning": planning,
        "rule": "Increase once if either primary task expects fewer than 40 test false-yes cases; never inspect test predictions"}


def calibration_receipt(journal, manifest, worker_id, n_workers, launch_id):
    """Small calibration-only receipt, safe to copy from a running private notebook."""
    import numpy as np
    items = sorted((r for r in manifest["rows"] if r["split"] == "calibration"
                    and worker_for(r["id"], n_workers) == worker_id), key=lambda r: r["id"])
    required = {(r["id"], p) for r in items for p in passes_for(r)}
    if not required <= journal.done:
        return None
    journal.flush()
    lookup = {r["question_id"]: r["answer"] for r in journal.rows if r["pass"] == "full"}
    packed = np.packbits(np.asarray([lookup[r["id"]] for r in items], dtype=np.uint8)).tobytes().hex()
    return {"worker_id": worker_id, "n_workers": n_workers, "launch_id": launch_id,
        "manifest_sha256": avh.digest(manifest), "identity_sha256": avh.digest(journal.identity),
        "cal_ids_sha256": avh.digest([r["id"] for r in items]), "n_cal": len(items),
        "cal_pair_count": len(required), "answers_packed_hex": packed, "scope": "calibration_only"}


def records_from_calibration_receipts(accounts, manifest, n_workers):
    import numpy as np
    receipts = [r for account in accounts for r in account]
    if len(receipts) != n_workers or {r["worker_id"] for r in receipts} != set(range(n_workers)):
        raise RuntimeError("Need exactly one calibration receipt per worker")
    if len({r["identity_sha256"] for r in receipts}) != 1:
        raise RuntimeError("Calibration receipts mix code/data/runtime identities")
    records = []
    for receipt in receipts:
        rows = sorted((r for r in manifest["rows"] if r["split"] == "calibration"
                       and worker_for(r["id"], n_workers) == receipt["worker_id"]), key=lambda r: r["id"])
        if (receipt["scope"] != "calibration_only" or receipt["manifest_sha256"] != avh.digest(manifest)
                or receipt["cal_ids_sha256"] != avh.digest([r["id"] for r in rows])
                or receipt["n_cal"] != len(rows)
                or receipt["cal_pair_count"] != sum(len(passes_for(r)) for r in rows)):
            raise RuntimeError("Calibration receipt coverage does not match the frozen sample")
        bits = np.unpackbits(np.frombuffer(bytes.fromhex(receipt["answers_packed_hex"]), dtype=np.uint8))
        if len(bits) < len(rows):
            raise RuntimeError("Truncated calibration answer receipt")
        records.extend((None, {"question_id": r["id"], "pass": "full", "answer": int(a)})
                       for r, a in zip(rows, bits))
    return records


def calibration_records_from_outputs(directories, manifest, n_workers):
    """Inspect calibration pairs only; ignore sealed validation/test values."""
    wanted = safe_pairs(manifest, ("calibration",))
    seen, records, identities = set(), [], set()
    family = {k: manifest[k] for k in ("parent_manifest_sha256", "group_assignment_sha256", "seed", "k")}
    for directory in map(Path, directories):
        identity = avh.read_json(directory / "identity.json")
        if identity["selection_family"] != family or identity["model"] != "gemma" or identity["n_workers"] != n_workers:
            raise RuntimeError("Calibration outputs belong to another study")
        identities.add(avh.digest(identity))
        for line in (directory / "predictions.jsonl").read_text().splitlines():
            row = json.loads(line)
            pair = (row["question_id"], row["pass"])
            if pair not in wanted:
                continue
            if pair in seen or row["identity_sha256"] != avh.digest(identity) or worker_for(pair[0], n_workers) != row["worker_id"]:
                raise RuntimeError("Duplicate or misassigned calibration pair")
            seen.add(pair)
            if pair[1] == "full":
                records.append((directory, row))
    if len(identities) != 1 or seen != wanted:
        raise RuntimeError("Complete matching calibration outputs are required")
    return records


def select_output_versions(directories):
    """Select a cumulative resume version; retain disjoint base/add-on journals.

    Overlap must be byte-equivalent JSON records from a genuine checkpoint resume.
    A fresh rerun with new values/timestamps is not silently deduplicated.
    """
    selected = []
    for source in map(Path, directories):
        current = {(r["question_id"], r["pass"]): r for r in
                   (json.loads(line) for line in (source / "predictions.jsonl").read_text().splitlines())}
        keep = True
        for old in list(selected):
            previous = {(r["question_id"], r["pass"]): r for r in
                        (json.loads(line) for line in (old / "predictions.jsonl").read_text().splitlines())}
            overlap = current.keys() & previous.keys()
            if not overlap:
                continue
            if any(current[k] != previous[k] for k in overlap):
                raise RuntimeError("Overlapping runs differ; cannot silently deduplicate")
            if current.keys() <= previous.keys():
                keep = False
                break
            if previous.keys() <= current.keys():
                selected.remove(old)
            else:
                raise RuntimeError("Partially overlapping output versions")
        if keep:
            selected.append(source)
    return selected


def import_for_analysis(dataset, manifest, directories, destination, n_workers):
    """Strict complete merge into the existing CPU analysis schema; sealed test gate stays active."""
    dataset, destination = Path(dataset), Path(destination)
    directories = list(map(Path, directories))
    if not directories:
        raise ValueError("No worker output directories supplied")
    identity = avh.read_json(directories[0] / "identity.json")
    family = {k: manifest[k] for k in ("parent_manifest_sha256", "group_assignment_sha256", "seed", "k")}
    if identity["selection_family"] != family or identity["n_workers"] != n_workers:
        raise RuntimeError("Wrong sample family or worker count")
    actual_code = avh.digest({name: avh.file_hash(Path(__file__).parent / name)
                            for name in ("avh_common.py", "avh_kaggle.py", "avh_kaggle_worker.py")})
    if identity["code_sha256"] != actual_code:
        raise RuntimeError("Analysis runtime differs from the scored code snapshot")
    records = validate_outputs(directories, manifest, identity, n_workers)
    model = identity["model"]
    cfg = avh.Config(str(destination), model, identity["model_revision"], k=2,
                     workers=n_workers, scoring_mode="full_forward")
    for name in ("terms_review.json", "release_inventory.json", "gemma_LICENSE.txt", "qwen_LICENSE.txt"):
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / name
        if target.exists() and avh.file_hash(target) != avh.file_hash(dataset / name):
            raise RuntimeError("Cannot replace existing analysis evidence")
        shutil.copyfile(dataset / name, target)
    out = destination / model
    out.mkdir(exist_ok=True)
    for name in ("model_metadata.json", "scoring_amendment.json"):
        shutil.copyfile(dataset / model / name, out / name)
    amendment = avh.read_json(out / "scoring_amendment.json")
    shutil.copyfile(dataset / model / amendment["prefix_evidence_file"], out / amendment["prefix_evidence_file"])
    prereg = dataset / "runtime/PREREGISTRATION_KAGGLE.md"
    prereg_bytes = prereg.read_bytes()
    if "serial_amendment_sha256" in identity:
        amendment_path = Path(__file__).with_name("GEMMA_SERIAL_AMENDMENT.md")
        if avh.file_hash(amendment_path) != identity["serial_amendment_sha256"]:
            raise RuntimeError("Analysis amendment differs from the scored protocol")
        prereg_bytes += b"\n\n" + amendment_path.read_bytes()
    destination_prereg = destination / "PREREGISTRATION.md"
    if destination_prereg.exists() and destination_prereg.read_bytes() != prereg_bytes:
        raise RuntimeError("Do not overwrite existing analysis preregistration")
    destination_prereg.write_bytes(prereg_bytes)
    immutable_json(destination / "manifest.json", manifest)
    index = resolved_index(dataset)
    environment = identity["environment"]
    signature = avh.study_signature(cfg, manifest, index, destination, environment)
    lookup = {r["id"]: r for r in manifest["rows"]}
    staged = collections.defaultdict(list)
    copied, feature_hashes = {}, {}
    for source, record in records:
        row = lookup[record["question_id"]]
        result = dict(record, signature=signature, **{k: row[k] for k in
            ("id", "video_id", "source_group", "split", "task", "label", "object")})
        if record["pass"] == "full":
            key = (str(source), record["feature"])
            if key not in feature_hashes:
                feature_hashes[key] = avh.file_hash(source / record["feature"])
            target = out / "f8/full/shards" / f"worker-{record['worker_id']:02d}-{feature_hashes[key][:16]}-{record['feature']}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if key not in copied:
                if target.exists() and avh.file_hash(target) != avh.file_hash(source / record["feature"]):
                    raise RuntimeError("Existing merged feature shard changed")
                shutil.copyfile(source / record["feature"], target)
                copied[key] = target
                if avh.file_hash(target) != feature_hashes[key]:
                    raise RuntimeError("Merged feature copy changed")
            result.update(feature=str(target.relative_to(out / "f8/full")),
                          feature_index=record["feature_slot"], feature_sha256=feature_hashes[key])
        staged[(record["pass"], record["worker_id"])].append(result)
    for (name, worker), rows in staged.items():
        target = out / "f8" / name / f"partition-{worker}-of-{n_workers}.jsonl"
        data = "".join(json.dumps(r, allow_nan=False) + "\n" for r in sorted(rows, key=lambda r: r["id"]))
        if target.exists() and target.read_text() != data:
            raise RuntimeError("Cannot replace an existing merged journal")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data)
    avh.create_audit(destination, manifest)
    immutable_json(out / "environment.json", environment)
    immutable_json(out / "merge_receipt.json", {"status": "complete_coverage_only", "signature": signature,
        "question_count": len(manifest["rows"]), "pair_count": len(records),
        "test_analysis": "blocked until FROZEN.json and owner HALT 2 approval"})
    return cfg, index, environment


def evaluate_sealed_once(root, cfg, manifest, index, environment, owner_ok_halt2=False):
    if owner_ok_halt2 is not True:
        raise RuntimeError("HALT 2: owner must approve frozen choices before test evaluation")
    out = Path(root) / cfg.model
    receipt = out / "test_evaluation.json"
    if receipt.exists():
        saved = avh.read_json(receipt)
        if (saved["metrics_sha256"] != avh.file_hash(out / "metrics.json")
                or saved["freeze_sha256"] != avh.file_hash(out / "FROZEN.json")):
            raise RuntimeError("Frozen test result changed")
        return avh.read_json(out / "metrics.json")
    if (out / "metrics.json").exists():
        raise RuntimeError("Unreceipted metrics exist; inspect before evaluating again")
    result = avh.evaluate_test(root, cfg, manifest, index, environment, "GO")
    mde_path = out / "calibration_mde.json"
    if mde_path.exists():
        mde = avh.read_json(mde_path)
        result["calibration_mde_planning"] = mde
        if any(r.get("mde_delta_auroc_approx") is not None and r["mde_delta_auroc_approx"] > .05
               for r in mde["tasks"].values()):
            result["limitations"] = [*result.get("limitations", []),
                "Calibration-based approximate MDE exceeds 0.05 for at least one primary task"]
        avh.atomic_json(out / "metrics.json", result)
    immutable_json(receipt, {"status": "evaluated_once", "metrics_sha256": avh.file_hash(out / "metrics.json"),
                           "freeze_sha256": avh.file_hash(out / "FROZEN.json")})
    return result


def calibration_mde(calibration):
    """Planning approximation from honest group-out-of-fold calibration predictions.

    Fixed last layer/C=.1 avoids selecting this planning number on validation.
    CI width is not a power simulation; the assumed normal approximation and
    alpha/power are explicit. No validation or test rows are accepted.
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold
    if any(r["split"] != "calibration" for r in calibration):
        raise RuntimeError("MDE planning must use calibration only")
    output = {"kind": "calibration_planning_approximation_not_test_result", "alpha": .05,
        "assumed_power": .8, "layer": "last", "C": .1, "tasks": {},
        "method": "(1.96+0.842)*CI_width/(2*1.96), paired group-bootstrap OOF Delta AUROC; approximate, not simulated power"}
    for task in avh.TASKS[:2]:
        rows = [r for r in calibration if r["task"] == task]
        yes = [r for r in rows if r["margin"] > 0]
        groups = np.array([r["source_group"] for r in rows])
        false_count = sum(not r["label"] for r in yes)
        summary = {"n_calibration": len(rows), "n_yes": len(yes), "n_false_yes": false_count,
                   "power_flag": "underpowered" if false_count < 40 else "at_least_40_false_yes",
                   "mde_delta_auroc_approx": None}
        output["tasks"][task] = summary
        if len(set(groups)) < 3 or len({r["label"] for r in yes}) < 2:
            summary["status"] = "undefined_missing_groups_or_class"
            continue
        predictions = {}
        failed = False
        for train_i, hold_i in GroupKFold(n_splits=min(5, len(set(groups)))).split(rows, groups=groups):
            train = [rows[i] for i in train_i]
            train_yes = [r for r in train if r["margin"] > 0]
            hold_yes = [rows[i] for i in hold_i if rows[i]["margin"] > 0]
            if not hold_yes:
                continue
            if len({r["label"] for r in train_yes}) < 2:
                failed = True
                break
            y = np.array([1-r["label"] for r in train_yes])
            bx = avh.baseline_matrix(train, train_yes, training=True)
            hx = avh.baseline_matrix(train, hold_yes)
            transform = avh.Residualizer().fit(train, train[0]["hidden"].shape[0] - 1)
            base = avh.logistic(bx, y, .1)
            aug = avh.logistic(np.column_stack([bx, transform.transform(train_yes)]), y, .1)
            bp = base.predict_proba(hx)[:, 1]
            ap = aug.predict_proba(np.column_stack([hx, transform.transform(hold_yes)]))[:, 1]
            predictions.update({r["id"]: (float(a), float(b)) for r, a, b in zip(hold_yes, ap, bp)})
        if failed or len(predictions) != len(yes):
            summary["status"] = "undefined_oof_training_class"
            continue
        paired = np.array([[1-r["label"], *predictions[r["id"]]] for r in yes])
        def delta(a):
            first, second = avh.safe_auc(a[:, 0], a[:, 1]), avh.safe_auc(a[:, 0], a[:, 2])
            return first-second if first is not None and second is not None else None
        ci = avh.cluster_ci(paired, [r["source_group"] for r in yes], delta)
        summary["oof_delta_auroc"] = ci
        if ci["ci95"] is None:
            summary["status"] = "undefined_bootstrap_width"
            continue
        summary["mde_delta_auroc_approx"] = (1.96+.842) * (ci["ci95"][1]-ci["ci95"][0]) / (2*1.96)
        summary["status"] = "planning_approximation"
        summary["sensitivity_limitation_gt_005"] = summary["mde_delta_auroc_approx"] > .05
    return output


def validation_report(root, cfg, manifest, index, environment):
    """Descriptive validation numbers after choices have been frozen; no test rows."""
    import joblib
    import numpy as np
    out = Path(root) / cfg.model
    freeze = avh.read_json(out / "FROZEN.json")
    if avh.file_hash(out / "analysis_bundle.joblib") != freeze["bundle_sha256"]:
        raise RuntimeError("Frozen analysis state changed")
    saved = joblib.load(out / "analysis_bundle.joblib")  # Locally created trusted state only.
    rows = avh.load_completed(root, cfg, manifest, index, environment, "validation")
    result = {"split": "validation", "choices_frozen": True, "tasks": {}}
    for task in avh.TASKS[:2]:
        selected = [r for r in rows if r["task"] == task]
        y, g = [r["label"] for r in selected], [r["source_group"] for r in selected]
        entry = {"full": avh.rates(y, [r["margin"] > 0 for r in selected], g)}
        bundle = saved["bundles"][task]
        yes = [r for r in selected if r["margin"] > 0]
        if bundle["status"] == "fitted" and len({r["label"] for r in yes}) == 2:
            scores = avh.probe_scores(bundle, saved["calibration"], yes)
            labels = np.array([1-r["label"] for r in yes])
            entry["baseline_auc"] = avh.auc_ci(labels, scores["baseline"], [r["source_group"] for r in yes])
            entry["augmented_auc"] = avh.auc_ci(labels, scores["augmented"], [r["source_group"] for r in yes])
        else:
            entry["probe_status"] = "underpowered_missing_class"
        result["tasks"][task] = entry
    avh.atomic_json(out / "validation_results.json", result)
    return result


def launch_account(runtime, job, output, hf_token=None):
    """Two new Python processes; CUDA is never initialized in the notebook parent."""
    workers = account_workers(job["account_index"], job["n_workers"])
    if job.get("mode") not in ("timing", "auto", "run", "addon"):
        raise ValueError("Unknown worker mode")
    if job.get("reference") != "fresh_unbatched_gpu":
        raise RuntimeError("Reference decision pending; no GPU load started")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    import uuid
    launch_id = uuid.uuid4().hex
    gate_path = output / f"gate-{launch_id}.json"
    worker_outputs = [output.parent / f"out_{job['account_index']}_{worker}" for worker in workers]
    children, handles = [], []
    gate_written, calibration_printed = False, False
    try:
        for gpu, worker in enumerate(workers):
            cfg = dict(job, worker_id=worker, local_gpu=gpu, output=str(worker_outputs[gpu]),
                       launch_id=launch_id, gate_file=str(gate_path))
            job_path = output / f"worker-{worker:02d}-job.json"
            avh.atomic_json(job_path, cfg)  # No credentials in this record.
            log = (output / f"worker-{worker:02d}.log").open("w", encoding="utf-8")
            handles.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
            if hf_token:
                env["HF_TOKEN"] = hf_token
            children.append(subprocess.Popen([sys.executable, str(Path(runtime) / "avh_kaggle_worker.py"),
                "--job", str(job_path)], env=env, stdout=log, stderr=subprocess.STDOUT))
        last = 0
        while any(p.poll() is None for p in children):
            if not calibration_printed:
                receipts = [avh.read_json(p / "calibration_receipt.json") for p in worker_outputs
                            if (p / "calibration_receipt.json").exists()]
                receipts = [r for r in receipts if r["launch_id"] == launch_id]
                if len(receipts) == 2:
                    print("CALIBRATION_ACCOUNT_JSON=" + json.dumps(receipts), flush=True)
                    avh.atomic_json(output.parent / "calibration_account.json", receipts)
                    calibration_printed = True
            if job["mode"] == "auto" and not gate_written:
                reports = [avh.read_json(p / "timing.json") for p in worker_outputs if (p / "timing.json").exists()]
                reports = [r for r in reports if r.get("launch_id") == launch_id]
                if len(reports) == 2:
                    summary = timing_summary(reports, avh.read_json(job["manifest"]), job["n_workers"])
                    startup = max(r["load_seconds"] + r["smoke_seconds"] for r in reports)
                    elapsed = time.monotonic() - job["session_started"]
                    # Friends start after receiving this approval. Include their observed startup allowance.
                    total = summary["forecast"]["ideal_wall_hours_scoring_only"] + (elapsed + startup) / 3600
                    summary["forecast"]["wall_hours_including_measured_owner_elapsed_and_friend_startup"] = total
                    approved = total <= 2.5
                    avh.atomic_json(output.parent / "timing_summary.json", summary)
                    gate = {"status": "approved" if approved else "halt_forecast",
                        "bundle_sha256": reports[0]["identity"]["bundle_sha256"],
                        "code_sha256": reports[0]["identity"]["code_sha256"],
                        "code_git_hash": reports[0]["identity"]["code_git_hash"],
                        "code_dirty": reports[0]["identity"]["code_dirty"],
                        "identity_sha256": avh.digest(reports[0]["identity"]),
                        "n_workers": job["n_workers"], "n": 2000,
                        "numerical_smoke_passed": True,
                        "comparison_kind": reports[0]["identity"].get("comparison_kind", "batch_parity"),
                        "forecast_wall_hours_planning_only": total,
                        "timing_summary_sha256": avh.digest(summary)}
                    avh.atomic_json(gate_path, gate)
                    avh.atomic_json(output.parent / "RUN_APPROVAL.json", gate)
                    print("RUN_APPROVAL_JSON=" + json.dumps(gate), flush=True)
                    print(json.dumps(summary, indent=2), flush=True)
                    print((Path(job["dataset"]) / "runtime/PREREGISTRATION_KAGGLE.md").read_text(), flush=True)
                    if "serial_amendment_sha256" in reports[0]["identity"]:
                        print(Path(__file__).with_name("GEMMA_SERIAL_AMENDMENT.md").read_text(), flush=True)
                    if not approved:
                        print("HALT: forecast exceeds 2.5 h. Proposed cut for owner review: test 1200 -> 900; keep cal/val and every pass.", flush=True)
                    gate_written = True
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError("A worker failed; inspect private worker logs")
            if time.monotonic() - last > 30:
                print("Workers running:", [w for w, p in zip(workers, children) if p.poll() is None], flush=True)
                last = time.monotonic()
            time.sleep(2)
        if any(p.returncode for p in children):
            raise RuntimeError("Worker failed; inspect private logs")
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for handle in handles:
            handle.close()
