"""CounterPrior v2, independent continuation, November 2026.

Import is CPU-only. Model dependencies are imported only at the inference boundary.
Bulk inference requires GO, terms, a manifest, and formal smoke. Audit gates analysis.
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import faulthandler
import hashlib
import json
import os
import platform
import re
import subprocess
import threading
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
TASKS = ("Video-driven Audio Hallucination", "Audio-driven Video Hallucination", "AV Matching")
PASSES = ("full", "cf0", "cf1", "cf2", "cf3", "blank", "no_media")
MODEL_IDS = {"gemma": "google/gemma-4-E2B-it", "qwen": "Qwen/Qwen2.5-Omni-3B"}
SEED = 20261103
LINEAGE_LIMITATION = "lineage_unrecoverable: grouping by released clip; swapped clips may share sources across splits"
OBJECT_RULE = "object-v1: casefold/normalize spaces; extract anchored Is there a/an, Is the ... making sound/visible, Can you hear/see; otherwise exact normalized question"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_go(owner_release):
    if owner_release != "GO":
        raise RuntimeError("HALT: notebook review required; owner must type GO in chat first.")


@dataclasses.dataclass(frozen=True)
class Config:
    root: str
    model: str
    revision: str  # resolved immutable Hub SHA, never main
    frames: int = 8
    max_side: int = 224
    seed: int = SEED
    k: int = 4
    worker_index: int = 0
    workers: int = 1
    stage: str = "full"
    scoring_mode: str = "verified_prefix"

    def __post_init__(self):
        if self.model not in MODEL_IDS or self.frames not in (1, 8) or self.k != 4:
            raise ValueError("Only the two approved models, 8/1 frames, and K=4 are permitted")
        if self.stage not in PASSES or not 0 <= self.worker_index < self.workers:
            raise ValueError("Invalid stage or work partition")
        if self.scoring_mode not in ("verified_prefix", "full_forward"):
            raise ValueError("Unknown scoring mode")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("Freeze the model to a 40-character Hub commit SHA")


def resolve_model(model):
    """Save actual license evidence; some repositories link it instead of shipping LICENSE."""
    from html.parser import HTMLParser
    from urllib.request import urlopen
    from huggingface_hub import HfApi, hf_hub_download
    repo = MODEL_IDS[model]
    info = HfApi().model_info(repo)
    files = {s.rfilename for s in info.siblings}
    candidates = sorted(f for f in files if f.casefold() in ("license", "license.txt", "license.md"))
    extra = {}
    if candidates:
        name = candidates[0]
        text = Path(hf_hub_download(repo, name, revision=info.sha)).read_text(encoding="utf-8")
        source_url = f"https://huggingface.co/{repo}/resolve/{info.sha}/{name}"
    else:
        card_path = hf_hub_download(repo, "README.md", revision=info.sha)
        card = Path(card_path).read_text(encoding="utf-8")
        # This fallback is specific to the publisher's explicit Gemma 4 model-card link.
        source_url = "https://ai.google.dev/gemma/docs/gemma_4_license"
        if (model != "gemma" or not re.search(r"(?m)^license:\s*apache-2\.0\s*$", card)
                or not re.search(r"(?m)^license_link:\s*" + re.escape(source_url) + r"\s*$", card)):
            raise RuntimeError("No bundled license or verified publisher license link; inspect model card")
        class LicenseParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.depth, self.parts = 0, []
            def handle_starttag(self, tag, attrs):
                if tag == "pre":
                    self.depth += 1
            def handle_endtag(self, tag):
                if tag == "pre":
                    self.depth -= 1
            def handle_data(self, value):
                if self.depth > 0:
                    self.parts.append(value)
        with urlopen(source_url, timeout=30) as response:
            resolved = response.geturl()
            if resolved not in (source_url, "https://ai.google.dev/gemma/apache_2"):
                raise RuntimeError("Unexpected redirect for publisher license evidence")
            parser = LicenseParser()
            parser.feed(response.read().decode("utf-8"))
        text = "".join(parser.parts).strip()
        if not all(term in text for term in ("Apache License", "Version 2.0", "Grant of Copyright License")):
            raise RuntimeError("Publisher page did not contain the expected full license text")
        extra = {"model_card_text": card, "model_card_sha256": file_hash(card_path),
                 "model_card_source_url": f"https://huggingface.co/{repo}/resolve/{info.sha}/README.md",
                 "license_resolved_url": resolved}
    if not text.strip():
        raise RuntimeError("License evidence was empty")
    return {"repo": repo, "revision": info.sha, "license_text": text,
            "license_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "license_source_url": source_url, **extra}


def download_release(root, owner_release):
    """Post-GO only. Archives stay private. No automatic extraction of untrusted paths."""
    require_go(owner_release)
    import gdown
    root = Path(root) / "release"
    root.mkdir(parents=True, exist_ok=True)
    ids = {"videos": "10-Qp8zxA3ITT-ileEnCgJkf5Nzx1wry7",
           "qa": "1KcYDAv9lLy3hsx5rWdfRqMFV2NYcZ94W"}
    for name, drive_id in ids.items():
        target = root / (name + ".download")
        if not target.exists():
            tmp = target.with_suffix(".partial")
            result = gdown.download(id=drive_id, output=str(tmp), quiet=False)
            if not result:
                raise RuntimeError("Dataset download failed; STOP rather than substitute data")
            os.replace(tmp, target)
    print("Downloaded private release files; inspect archive members and README/terms before extraction/use.")
    return {p.name: file_hash(p) for p in root.glob("*.download")}


def extract_release(root):
    """Extract only regular files/directories in zip/tar; reject traversal and symlinks."""
    import tarfile
    import zipfile
    root = Path(root) / "release"
    out = (root / "unpacked").resolve()
    out.mkdir(exist_ok=True)
    for archive in root.glob("*.download"):
        def safe(name):
            dest = (out / name).resolve()
            if not dest.is_relative_to(out):
                raise ValueError("Unsafe archive member: " + name)
            return dest
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    safe(member.filename)
                    if (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("Archive symlink rejected")
                z.extractall(out)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as t:
                for member in t.getmembers():
                    safe(member.name)
                    if not (member.isfile() or member.isdir()):
                        raise ValueError("Archive links/devices rejected")
                t.extractall(out, filter="data")
        else:
            # QA may be a JSON file rather than an archive.
            value = read_json(archive)
            atomic_json(out / (archive.stem + ".json"), value)
    return out


def inventory_release(unpacked, schema_mapping=None, records_key=None, task_mapping=None):
    """Print unfiltered records first. Noncanonical schemas require explicit mappings."""
    root = Path(unpacked)
    media = sorted(root.rglob("*.mp4"))
    raw, containers = [], {}
    for path in sorted(root.rglob("*.json")):
        value = read_json(path)
        if isinstance(value, dict):
            containers[str(path.relative_to(root))] = sorted(value)
            if records_key is not None:
                value = value.get(records_key)
            else:
                # Show the container verbatim; do not silently choose a list-valued key.
                raw.append(value)
                continue
        if isinstance(value, list):
            raw.extend(value)
    print("FIRST THREE RAW RECORDS (unmodified field values):")
    for record in raw[:3]:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    mapping = schema_mapping or {k: k for k in ("video_id", "task", "text", "label")}
    task_key = mapping.get("task", "task")
    names = sorted({str(x[task_key]) for x in raw if isinstance(x, dict) and task_key in x})
    terms = [str(p.relative_to(root)) for p in root.rglob("*")
             if p.is_file() and any(s in p.name.lower() for s in ("readme", "licen", "terms"))]
    result = {"media_files": len(media), "raw_record_count": len(raw), "terms_files": terms,
              "all_key_names": sorted({k for x in raw if isinstance(x, dict) for k in x}),
              "container_key_names": containers, "all_task_names": names,
              "raw_task_counts": {t: sum(isinstance(x, dict) and str(x.get(task_key)) == t for x in raw) for t in names},
              "schema_mapping": mapping, "records_key": records_key, "task_mapping": task_mapping or {}}
    print(json.dumps(result, indent=2))
    if (set(mapping) != {"video_id", "task", "text", "label"} or not raw or
            any(not isinstance(x, dict) or any(k not in x for k in mapping.values()) for x in raw)):
        raise ValueError("QA schema differs: set explicit schema_mapping / records_key from the printed records; no guessing")
    qa = []
    for record in raw:
        row = dict(record)  # preserve release metadata for lineage recovery
        row.update({canonical: record[actual] for canonical, actual in mapping.items()})
        row["video_id"] = str(row["video_id"])
        row["task"] = (task_mapping or {}).get(row["task"], row["task"])
        qa.append(row)
    result.update(qa_rows=len(qa), task_counts={t: sum(x["task"] == t for x in qa) for t in sorted({x["task"] for x in qa})})
    atomic_json(root.parent / "qa_schema_mapping.json", {k: result[k] for k in ("schema_mapping", "records_key", "task_mapping")})
    print("EXPLICIT QA MAPPING:", json.dumps({k: result[k] for k in ("schema_mapping", "records_key", "task_mapping")}))
    return qa, result


def check_terms(root, model):
    """Evidence must be a saved local document + exact applicable passage, not a boolean."""
    review = read_json(Path(root) / "terms_review.json")
    for key in ("dataset", model):
        entry = review[key]
        private = key == "dataset" and entry.get("decision") == "no_licence_stated_private_research_eval"
        if entry.get("decision") != "research_use_permitted" and not private:
            raise RuntimeError(f"STOP: {key} terms not established as usable")
        evidence = Path(root) / entry["evidence_file"]
        if file_hash(evidence) != entry["sha256"]:
            raise RuntimeError("Terms evidence changed")
        quote = entry.get("applicable_passage", "")
        if not private and (not quote.strip() or quote not in evidence.read_text(encoding="utf-8")):
            raise RuntimeError("Record an exact applicable terms passage and source URL")
        if not entry.get("source_url") or not entry.get("reviewer"):
            raise RuntimeError("Terms require provenance and a named reviewer")
    return review


def object_key(question):
    text = " ".join(question.casefold().strip().rstrip("?.!").split())
    patterns = (r"^is there (?:an? |the )?(.+?)(?: (?:in|on) (?:the )?(?:audio|video|image|clip))?$",
                r"^is (?:the |an? )?(.+?) (?:making (?:a )?sound|visible|audible)(?: .*)?$",
                r"^(?:can|do) you (?:hear|see) (?:the |an? )?(.+?)(?: (?:in|on) .*)?$")
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            return match[1].strip()
    return "question:" + text


def auto_lineage(root, qa, media_dir):
    """Recover only explicit metadata/marked filename origins; otherwise disclose clip fallback."""
    by_video = defaultdict(list)
    for row in qa:
        by_video[str(row["video_id"])].append(row)
    output = []
    for vid, rows in sorted(by_video.items()):
        matches = list(Path(media_dir).rglob(vid + ".mp4"))
        name = str(matches[0].relative_to(media_dir)) if len(matches) == 1 else vid
        metadata_rows = list(rows)
        metadata_rows.extend(r["metadata"] for r in rows if isinstance(r.get("metadata"), dict))
        for sidecar in Path(media_dir).rglob(vid + ".json"):
            value = read_json(sidecar)
            if isinstance(value, dict):
                metadata_rows.append(value)
        origins, evidence = set(), []
        swapped = any(r.get("is_swapped") is True for r in metadata_rows) or bool(re.search(r"(?:^|[/_\-])(swapped|swap|synthetic)(?:[/_\-]|$)", name, re.I))
        marked = swapped or any(isinstance(r.get("is_swapped"), bool) for r in metadata_rows)
        for row in metadata_rows:
            source_ids = row.get("source_ids", [])
            if isinstance(source_ids, list):
                origins.update(str(s) for s in source_ids if s is not None and str(s).strip())
                if source_ids:
                    evidence.append("metadata:source_ids")
            for field in ("audio_source_id", "video_source_id", "source_video_id", "valor_id", "audiocaps_id"):
                if row.get(field) is not None and str(row[field]).strip():
                    origins.add(str(row[field]))
                    evidence.append("metadata:" + field)
        # Explicitly tagged names only, e.g. audio_dog1__video_clip2; never split opaque IDs.
        tagged = re.fullmatch(r"audio_(.+?)__video_(.+)", vid, re.I)
        if tagged:
            origins.update(tagged.groups())
            evidence.append("filename:audio_<id>__video_<id>")
            swapped, marked = True, True
        unrecoverable = not origins or (swapped and len(origins) < 2)
        output.append({"video_id": vid, "source_ids": sorted(origins) if origins else [vid],
            "is_swapped": swapped, "swap_status_known": marked,
            "lineage_unrecoverable": unrecoverable, "grouping_fallback": unrecoverable,
            "provenance": "; ".join(sorted(set(evidence))) if evidence else "released clip ID fallback",
            "object_rule": OBJECT_RULE,
            "object_by_question": {r["text"]: object_key(r["text"]) for r in rows if r["task"] in TASKS}})
    path = Path(root) / "lineage.json"
    if path.exists():
        return read_json(path)  # retain any prior explicit/manual corrections
    atomic_json(path, output)
    atomic_json(Path(root) / "lineage_rules.json", {"object_rule": OBJECT_RULE,
        "source_fields": ["source_ids", "audio_source_id", "video_source_id", "source_video_id", "valor_id", "audiocaps_id"],
        "filename_pattern": "audio_<id>__video_<id>",
        "unknown_swap_status": "not marked swapped, not proof of real origin", "limitation": LINEAGE_LIMITATION})
    return output


def connected_groups(lineage):
    """Union all audio/video origins, including swapped clips, before assigning splits."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    for row in lineage:
        origins = [row["video_id"]] if row.get("grouping_fallback") else row["source_ids"]
        if not origins or any(not str(x).strip() for x in origins):
            raise ValueError("Unknown source lineage: STOP; numeric release IDs are not source IDs")
        for source in origins:
            union(str(origins[0]), str(source))
    return {r["video_id"]: find(str(r["video_id"] if r.get("grouping_fallback") else r["source_ids"][0])) for r in lineage}


def make_manifest(root, qa, lineage, seed=SEED):
    """lineage: video_id, source_ids (both origins), is_swapped, provenance, object map.

    Auto object keys and explicitly disclosed released-clip fallback are supported.
    """
    import numpy as np
    root = Path(root)
    groups = connected_groups(lineage)
    by_video = {r["video_id"]: r for r in lineage}
    if len(by_video) != len(lineage):
        raise ValueError("Duplicate lineage IDs")
    unique = sorted(set(groups.values()))
    if len(unique) < 10:
        raise ValueError("Fewer than ten independent connected source groups: STOP for protocol review")
    order = np.random.default_rng(seed).permutation(unique).tolist()
    ncal, nval = max(1, int(.6 * len(order))), max(1, int(.2 * len(order)))
    splits = {g: ("calibration" if i < ncal else "validation" if i < ncal+nval else "test")
              for i, g in enumerate(order)}
    rows = []
    for q in qa:
        if q["task"] not in TASKS:
            continue
        vid = str(q["video_id"])
        meta = by_video[vid]
        if not meta.get("provenance") or not isinstance(meta.get("is_swapped"), bool):
            raise ValueError("Record lineage provenance and real/swapped status")
        label = str(q["label"]).strip().lower()
        if label not in ("yes", "no"):
            raise ValueError("Unexpected yes/no label")
        obj = meta["object_by_question"][q["text"]]
        if not obj:
            raise ValueError("Object/event key must be derived before scoring")
        row = {"id": digest([vid, q["task"], q["text"]])[:24], "video_id": vid,
               "source_group": groups[vid], "source_ids": meta["source_ids"],
               "split": splits[groups[vid]], "task": q["task"], "question": q["text"],
               "label": int(label == "yes"), "object": obj, "is_swapped": meta["is_swapped"],
               "swap_status_known": meta.get("swap_status_known", True)}
        rows.append(row)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate QA rows; resolve explicitly, do not silently deduplicate")
    # Donor selection is clip-level, unique, split-local, and source-disjoint from target.
    for row in rows:
        candidates = sorted({r["video_id"] for r in rows if r["split"] == row["split"]
                             and r["source_group"] != row["source_group"] and not r["is_swapped"]})
        if len(candidates) < 4:
            raise ValueError("Need four real, other-source donor clips in every split")
        rng = np.random.default_rng(int(digest([seed, row["video_id"], row["task"]])[:16], 16))
        row["donors"] = rng.choice(candidates, 4, replace=False).tolist()
    path = root / "manifest.json"
    limitations = [LINEAGE_LIMITATION] if any(r.get("lineage_unrecoverable") for r in lineage) else []
    if any(r.get("swap_status_known") is False for r in lineage):
        limitations.append("Swap status absent for some clips: donor pool uses clips not marked swapped; real origin is unverified for those clips")
    payload = {"seed": seed, "rows": rows, "lineage_sha256": digest(lineage),
               "qa_sha256": digest(qa), "split_rule": "connected recovered origins or disclosed released-clip fallback, 60/20/20",
               "limitations": limitations, "object_rule": OBJECT_RULE}
    if path.exists() and read_json(path) != payload:
        raise RuntimeError("Manifest already frozen: start a new study directory to change it")
    atomic_json(path, payload)
    return payload


def audit_targets(manifest, limit=100):
    selected = {}
    label_items = sorted((r for r in manifest["rows"] if r["is_swapped"] or r.get("swap_status_known") is False),
                         key=lambda r: digest([SEED, "audit", r["id"]]))[:limit]
    objects = sorted(manifest["rows"], key=lambda r: digest([SEED, "object-check", r["id"]]))[:50]
    for row in label_items + objects:
        selected[row["id"]] = dict(row, label_audit=row in label_items, object_spotcheck=row in objects)
    return list(selected.values())


def create_audit(root, manifest, limit=100):
    """Blind media/label review; no model answers or hidden states in this worksheet."""
    root = Path(root)
    path = root / "audit.csv"
    if path.exists():
        return path
    items = audit_targets(manifest, limit)
    fields = ["id", "video_id", "task", "question", "label", "object", "label_audit", "object_spotcheck",
              "decision", "object_check", "reviewer", "note"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in fields} for r in items)
    return path


def check_audit(root, manifest, require_complete=False):
    path = Path(root) / "audit.csv"
    if not path.exists() and not require_complete:
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    expected = audit_targets(manifest)
    if {r["id"] for r in rows} != {r["id"] for r in expected} or len(rows) != len(expected):
        raise RuntimeError("Audit selection changed or incomplete")
    targets = {r["id"]: r for r in expected}
    complete = [r for r in rows if targets[r["id"]]["label_audit"] and
                r["decision"] in ("valid", "exclude", "uncertain") and r["reviewer"]]
    object_complete = [r for r in rows if targets[r["id"]]["object_spotcheck"] and
                       r.get("object_check") in ("correct", "flagged") and r["reviewer"]]
    n_labels = sum(r["label_audit"] for r in expected)
    n_objects = sum(r["object_spotcheck"] for r in expected)
    if require_complete and (len(complete) != n_labels or len(object_complete) != n_objects):
        raise RuntimeError("Complete label audit and 50-item object spot-check before analysis freeze")
    excluded = {r["id"] for r in complete if r["decision"] != "valid"}
    # Audit error CI clusters by source, using the shared bootstrap implementation.
    groups = {r["id"]: r["source_group"] for r in manifest["rows"]}
    import numpy as np
    error = np.array([r["decision"] == "exclude" for r in complete], dtype=float)
    result = cluster_ci(error, [groups[r["id"]] for r in complete], lambda a: float(a.mean()))
    result.update({"n_audited": len(complete), "n_label_pending": n_labels-len(complete),
                   "n_object_checked": len(object_complete), "n_object_pending": n_objects-len(object_complete),
                   "n_object_flagged": sum(r["object_check"] == "flagged" for r in object_complete),
                   "n_uncertain": sum(r["decision"] == "uncertain" for r in complete),
                   "excluded_ids": sorted(excluded), "audit_sha256": file_hash(path)})
    atomic_json(Path(root) / "audit_summary.json", result)
    return excluded


def prepare_media(root, video_dir, manifest, frames=8, max_side=224):
    """Same deterministic frame indices/audio bytes for both models; full track, no truncation."""
    import numpy as np
    from PIL import Image
    import av
    import soundfile as sf
    root = Path(root)
    video_dir = Path(video_dir)
    index = {}
    for vid in sorted({r["video_id"] for r in manifest["rows"]}):
        matches = list(video_dir.rglob(vid + ".mp4"))
        if len(matches) != 1:
            raise ValueError(f"Need exactly one media file for {vid}, found {len(matches)}")
        source = matches[0]
        target = root / "media" / f"f{frames}-s{max_side}" / vid
        target.mkdir(parents=True, exist_ok=True)
        ready = target / "ready.json"
        if ready.exists():
            saved = read_json(ready)
            if saved["source_sha256"] != file_hash(source):
                raise RuntimeError("Source video changed during preprocessing")
            if any(file_hash(target / name) != sha for name, sha in saved["file_hashes"].items()):
                raise RuntimeError("Preprocessed media changed")
            index[vid] = saved
            continue
        with av.open(str(source)) as container:
            count = sum(1 for _ in container.decode(video=0))
        if count < 1:
            raise ValueError("Empty video")
        indices = (np.linspace(0, count - 1, frames).round().astype(int).tolist()
                   if frames == 8 else [count // 2])
        with av.open(str(source)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                for j in [j for j, wanted in enumerate(indices) if wanted == i]:
                    image = frame.to_image().convert("RGB")
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                    image.save(target / f"{j:02d}.png")
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source),
                        "-vn", "-ac", "1", "-ar", "16000", str(target / "audio.wav")], check=True)
        audio, sr = sf.read(target / "audio.wav")
        if sr != 16000 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("Invalid extracted audio")
        index[vid] = {"directory": str(target), "source_sha256": file_hash(source),
                      "frame_indices": indices, "audio_samples": len(audio),
                      "file_hashes": {p.name: file_hash(p) for p in sorted(target.iterdir()) if p.name != "ready.json"}}
        atomic_json(ready, index[vid])
    atomic_json(root / f"media_index_f{frames}.json", index)
    return index


def verify_media(index):
    for entry in index.values():
        directory = Path(entry["directory"])
        for name, sha in entry["file_hashes"].items():
            if file_hash(directory / name) != sha:
                raise RuntimeError("Media bytes changed since preparation")


class Watchdog:
    """A truly wedged CUDA call cannot be interrupted safely: dump stacks, end kernel.

    Only active inside an inference stage; no background idle-kernel termination.
    """
    def __init__(self, seconds=600, deadline=None):
        self.seconds = seconds
        self.deadline = deadline
        self.last = time.monotonic()
        self.stop = threading.Event()

    def touch(self):
        self.last = time.monotonic()

    def __enter__(self):
        def monitor():
            while not self.stop.wait(5):
                now = time.monotonic()
                if now - self.last > self.seconds or (self.deadline is not None and now > self.deadline):
                    print("STALLED: no durable row within time limit; terminating kernel", flush=True)
                    faulthandler.dump_traceback(all_threads=True)
                    os._exit(124)
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)


class Journal:
    """Single writer, dedicated partition. Per-question features durable before row commit.

    Unique stage/partition paths avoid cross-account shared-file appends. Drive is not
    a transactional filesystem: never run the same partition twice concurrently.
    """
    def __init__(self, directory, worker, signature):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{worker}.jsonl"
        self.lock = self.directory / f"{worker}.lock"
        self.signature = signature

    def __enter__(self):
        fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "started": time.time(), "signature": self.signature}, f)
        self.rows = []
        if self.path.exists():
            raw = self.path.read_bytes()
            lines = raw.splitlines(keepends=True)
            valid_bytes = 0
            for i, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if i != len(lines)-1 or line.endswith(b"\n"):
                        self.lock.unlink()
                        raise RuntimeError("Corrupt journal interior; preserve and investigate")
                    break  # only an incomplete final write can be recovered
                if row["signature"] != self.signature:
                    self.lock.unlink()
                    raise RuntimeError("Journal configuration mismatch")
                self.rows.append(row)
                valid_bytes += len(line)
            with open(self.path, "r+b") as f:
                f.truncate(valid_bytes)
            if self.rows and not self.path.read_bytes().endswith(b"\n"):
                with open(self.path, "ab") as f:
                    f.write(b"\n")
        self.done = {r["id"] for r in self.rows}
        if len(self.done) != len(self.rows):
            self.lock.unlink()
            raise RuntimeError("Duplicate journal IDs")
        return self

    def append(self, row, hidden):
        import numpy as np
        if row["id"] in self.done:
            return
        if hidden is not None:
            target = self.directory / "pending_features" / (row["id"] + ".npz")
            target.parent.mkdir(exist_ok=True)
            tmp = target.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                np.savez(f, hidden=np.asarray(hidden, dtype=np.float16))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
            row["feature"] = str(target.relative_to(self.directory))
            row["feature_sha256"] = file_hash(target)
        row["signature"] = self.signature
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.rows.append(row)
        self.done.add(row["id"])

    def compact(self):
        import numpy as np
        rows = [r for r in self.rows if "feature" in r]
        for start in range(0, len(rows), 200):
            batch = rows[start:start+200]
            path = self.directory / f"{self.path.stem}-features-{start//200:05d}.npz"
            # Pending per-item files remain the recovery source. Shards are derived.
            states = [np.load(self.directory / r["feature"])["hidden"] for r in batch]
            with open(path.with_suffix(".tmp"), "wb") as f:
                np.savez(f, ids=np.asarray([r["id"] for r in batch]), hidden=np.stack(states).astype(np.float16))
            os.replace(path.with_suffix(".tmp"), path)

    def __exit__(self, *args):
        self.lock.unlink(missing_ok=True)


def attention_backend(model):
    # Gemma's chunked audio attention only implements eager in Transformers.
    return ({"": "sdpa", "text_config": "sdpa", "vision_config": "sdpa", "audio_config": "eager"}
            if model == "gemma" else "sdpa")


class OmniScorer:
    def __init__(self, cfg):
        import torch
        import transformers
        self.torch, self.cfg = torch, cfg
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Protocol requires a CUDA T4; do not claim fit from CPU/offload")
        if torch.cuda.get_device_properties(0).total_memory < 14 * 1024**3:
            raise RuntimeError("Insufficient GPU memory")
        torch.manual_seed(cfg.seed)
        torch.cuda.reset_peak_memory_stats()
        self.processor = transformers.AutoProcessor.from_pretrained(MODEL_IDS[cfg.model], revision=cfg.revision)
        cls = (transformers.Gemma4ForConditionalGeneration if cfg.model == "gemma"
               else transformers.Qwen2_5OmniThinkerForConditionalGeneration)
        self.model = cls.from_pretrained(MODEL_IDS[cfg.model], revision=cfg.revision,
                    torch_dtype=torch.float16, device_map={"": 0},
                    attn_implementation=attention_backend(cfg.model)).eval()
        if any(p.device.type != "cuda" or (p.is_floating_point() and p.dtype != torch.float16)
               for p in self.model.parameters()):
            raise RuntimeError("All parameters must be fp16 on GPU; no silent offload/quantization")
        # v4 Qwen projects every sequence position otherwise. Last-token logits only;
        # decoder hidden-state outputs remain unchanged. Also used in generation parity.
        self.head_hook = self.model.get_output_embeddings().register_forward_pre_hook(
            lambda module, args: (args[0][:, -1:, :],))
        self.tokens = self.answer_tokens()
        self.prefix = None
        self.prefix_key = None
        self.env = {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
                    "model_revision": cfg.revision, "cuda": torch.version.cuda,
                    "attention_backend": attention_backend(cfg.model),
                    "scoring_mode": cfg.scoring_mode,
                    "packages": subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True).splitlines()}
        print(json.dumps({"environment": self.env, "answer_tokens": self.tokens}, indent=2))

    def answer_tokens(self):
        tok = self.processor.tokenizer
        result = {"yes": [], "no": []}
        # Enumerate every vocabulary token decoding to exactly a Yes/No case/space variant.
        for token in range(len(tok)):
            if token in tok.all_special_ids:
                continue
            decoded = tok.decode([token], skip_special_tokens=False)
            if decoded.strip().casefold() in result:
                result[decoded.strip().casefold()].append(token)
        if not all(result.values()) or set(result["yes"]) & set(result["no"]):
            raise RuntimeError("Invalid Yes/No token mapping")
        return result

    def media(self, row, pass_name, index):
        import numpy as np
        import soundfile as sf
        from PIL import Image
        if pass_name == "no_media":
            return None, None, "no_media"
        vid = row["video_id"]
        image_vid = audio_vid = vid
        if pass_name.startswith("cf"):
            donor = row["donors"][int(pass_name[-1])]
            if row["task"] == TASKS[0]:
                audio_vid = donor
            elif row["task"] == TASKS[1]:
                image_vid = donor
            else:
                raise ValueError("AV Matching has full/no-media only; no privileged queried sense")
        images = [Image.open(p).convert("RGB") for p in sorted(Path(index[image_vid]["directory"]).glob("*.png"))]
        audio, sr = sf.read(Path(index[audio_vid]["directory"]) / "audio.wav", dtype="float32")
        if sr != 16000 or len(images) != self.cfg.frames:
            raise ValueError("Media format mismatch")
        if pass_name == "blank":
            if row["task"] == TASKS[0]:
                audio = np.zeros_like(audio)
            elif row["task"] == TASKS[1]:
                images = [Image.new("RGB", im.size) for im in images]
            else:
                raise ValueError("AV Matching blank undefined")
        return images, audio, digest([image_vid, audio_vid,
                    row["task"] if pass_name == "blank" else None, self.cfg.frames])

    def inputs(self, row, pass_name, index):
        torch = self.torch
        images, audio, key = self.media(row, pass_name, index)
        media = ([{"type": "image"} for _ in images] + [{"type": "audio"}]) if images else []
        def encode(question):
            content = media + [{"type": "text", "text": "\nQuestion: " + question + "\nAnswer Yes or No only."}]
            text = self.processor.apply_chat_template([{"role": "user", "content": content}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
            args = {"text": text, "return_tensors": "pt"}
            if images:
                args.update(images=images, audio=[audio], sampling_rate=16000)
            if self.cfg.model == "qwen":
                args["use_audio_in_video"] = False
            encoded = self.processor(**args)
            return {k: v.to(device="cuda", dtype=torch.float16 if v.is_floating_point() else v.dtype)
                    for k, v in encoded.items() if isinstance(v, torch.Tensor)}
        full, empty = encode(row["question"]), encode("")
        a, b = full["input_ids"][0], empty["input_ids"][0]
        n = min(len(a), len(b))
        unequal = (a[:n] != b[:n]).nonzero()
        cut = int(unequal[0]) if len(unequal) else n
        if cut <= 0 or cut >= len(a):
            raise RuntimeError("Cannot isolate question-independent media prefix")
        if self.cfg.model == "qwen":
            audio_lengths = full.get("feature_attention_mask")
            audio_lengths = audio_lengths.sum(-1) if audio_lengths is not None else None
            positions, _ = self.model.get_rope_index(
                full["input_ids"], full.get("image_grid_thw"), None, full["attention_mask"],
                False, audio_lengths, None)
            full["position_ids"] = positions
        else:
            full["position_ids"] = torch.arange(len(a), device="cuda").unsqueeze(0)
        key = digest([key, a[:cut].tolist()])
        return full, cut, key

    def split_inputs(self, full, cut, suffix=False):
        length = full["input_ids"].shape[-1]
        if not suffix:
            result = dict(full)
            for k in result:
                if k not in ("input_ids", "attention_mask", "position_ids") and "token_type" not in k and "mm_type" not in k:
                    continue
                if k in result:
                    result[k] = result[k][..., :cut]
            return result
        result = {"input_ids": full["input_ids"][:, cut:], "attention_mask": full["attention_mask"],
                  "position_ids": full["position_ids"][..., cut:]}
        for key, value in full.items():
            if "token_type" in key or "mm_type" in key:
                result[key] = value[..., cut:]
        if self.cfg.model == "qwen":
            result["cache_position"] = self.torch.arange(cut, length, device="cuda")
        return result

    def _forward(self, inputs, **kwargs):
        with self.torch.inference_mode():
            return self.model(**inputs, return_dict=True, output_hidden_states=True, **kwargs)

    def extract(self, output):
        import numpy as np
        torch = self.torch
        logits = output.logits[0, -1].float()
        hidden = torch.stack([h[0, -1].float() for h in output.hidden_states])
        if not torch.isfinite(logits).all() or not torch.isfinite(hidden).all():
            raise FloatingPointError("fp16 overflow/nonfinite logits or hidden states; STOP")
        yes = torch.logsumexp(logits[self.tokens["yes"]], 0)
        no = torch.logsumexp(logits[self.tokens["no"]], 0)
        margin = float(yes-no)
        array = hidden.cpu().numpy()
        if not np.isfinite(array.astype(np.float16)).all():
            raise FloatingPointError("Feature shard would overflow fp16")
        return {"margin": margin, "answer": int(margin > 0), "yes_logsumexp": float(yes),
                "no_logsumexp": float(no), "greedy_token": int(logits.argmax())}, array, logits.cpu().numpy()

    def score(self, row, pass_name, index, cached=None):
        tick = time.monotonic()
        mode = getattr(self.cfg, "scoring_mode", "verified_prefix")
        if cached is None:
            cached = mode == "verified_prefix"
        if mode == "full_forward" and cached:
            raise RuntimeError("Prefix reuse is disabled in full_forward mode")
        full, cut, key = self.inputs(row, pass_name, index)
        if not cached:
            output = self._forward(full, use_cache=False)
        else:
            if self.prefix_key != key:
                self.prefix = None
                self.torch.cuda.empty_cache()
                output = self._forward(self.split_inputs(full, cut), use_cache=True)
                self.prefix = output.past_key_values
                self.prefix_key = key
                del output
            # DynamicCache mutates; deepcopy must isolate both tensor storage and metadata.
            cache = copy.deepcopy(self.prefix)
            before = self.prefix.get_seq_length()
            output = self._forward(self.split_inputs(full, cut, suffix=True), past_key_values=cache, use_cache=True)
            if self.prefix.get_seq_length() != before:
                raise RuntimeError("Prefix cache mutated across questions")
        result, hidden, logits = self.extract(output)
        del output
        self.torch.cuda.synchronize()
        result["peak_vram_bytes"] = self.torch.cuda.max_memory_allocated()
        result["seconds"] = time.monotonic() - tick
        result["scoring_mode"] = mode
        return result, hidden, logits

    def generation_check(self, row, index):
        full, _, _ = self.inputs(row, "full", index)
        # Reset model-level Qwen RoPE state; generate must compute its own media positions.
        full.pop("position_ids", None)
        if hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = None
        with self.torch.inference_mode():
            output = self.model.generate(**full, max_new_tokens=1, do_sample=False,
                                        return_dict_in_generate=True, output_scores=True)
        scores = output.scores[0][0].float()
        generated = int(output.sequences[0, -1])
        return {"generated_token": generated, "generated_text": self.processor.tokenizer.decode([generated]),
                "generated_answer": 1 if generated in self.tokens["yes"] else 0 if generated in self.tokens["no"] else None,
                "margin": float(self.torch.logsumexp(scores[self.tokens["yes"]], 0) -
                                self.torch.logsumexp(scores[self.tokens["no"]], 0))}


def record_scoring_amendment(cfg):
    """Explicit pre-test fallback record; never relabel failed prefix parity as passed."""
    if cfg.scoring_mode != "full_forward":
        return
    folder = Path(cfg.root) / cfg.model
    evidence_path = folder / "cache_diagnostic_v3.json"
    evidence = read_json(evidence_path)
    if evidence.get("status") != "comparison_complete" or not (
            evidence["logits_max_abs_diff"] > .02 or evidence["hidden_max_abs_diff"] > .02
            or len(set(evidence["answers"])) != 1):
        raise RuntimeError("Full-forward fallback requires the saved failed prefix diagnostic")
    payload = {"scoring_mode": "full_forward", "prefix_cache_status": "failed_gate_disabled",
        "model": MODEL_IDS[cfg.model], "revision": cfg.revision, "frames": cfg.frames,
        "reason": "Use fresh full forward as reference after prefix parity failed; no tolerance relaxation",
        "prefix_evidence_file": evidence_path.name, "prefix_evidence_sha256": file_hash(evidence_path),
        "prefix_evidence": evidence, "code_sha256": file_hash(__file__),
        "formal_checks": "full-forward repeatability, generation agreement, finiteness and VRAM",
        "statistics": "unchanged; separate formal smoke and pre-test analysis freeze still required"}
    target = folder / "scoring_amendment.json"
    if target.exists() and read_json(target) != payload:
        raise RuntimeError("Scoring amendment changed; do not reuse an existing run identity")
    atomic_json(target, payload)


def use_full_forward(scorer, cfg):
    """Upgrade a retained scorer's dispatch without allocating/reloading model weights."""
    from types import MethodType
    updated = Config(**{**dataclasses.asdict(cfg), "scoring_mode": "full_forward"})
    for key in ("model", "revision", "frames", "max_side"):
        if getattr(scorer.cfg, key) != getattr(updated, key):
            raise RuntimeError("Retained model/config mismatch: " + key)
    record_scoring_amendment(updated)
    scorer.cfg = updated
    scorer.score = MethodType(OmniScorer.score, scorer)
    scorer.env = dict(scorer.env, scoring_mode="full_forward")
    scorer.prefix = None
    scorer.prefix_key = None
    return updated


def study_signature(cfg, manifest, media_index, root, environment):
    return digest({"model": cfg.model, "revision": cfg.revision, "frames": cfg.frames,
                   "max_side": cfg.max_side, "seed": cfg.seed, "k": cfg.k,
                   "workers": cfg.workers, "manifest": digest(manifest), "media": digest(media_index),
                   "scoring_mode": cfg.scoring_mode,
                   "scoring_amendment": file_hash(Path(root) / cfg.model / "scoring_amendment.json")
                       if cfg.scoring_mode == "full_forward" else None,
                   "terms": file_hash(Path(root) / "terms_review.json"),
                   "preregistration": file_hash(Path(root) / "PREREGISTRATION.md"),
                   "code": file_hash(__file__), "environment": environment})


def smoke_test(scorer, cfg, manifest, index, owner_release):
    require_go(owner_release)
    if getattr(getattr(scorer, "cfg", None), "scoring_mode", "verified_prefix") != cfg.scoring_mode:
        raise RuntimeError("Scorer mode differs from smoke configuration")
    check_terms(cfg.root, cfg.model)
    verify_media(index)
    # Calibration only. Test data remain unseen during all engineering choices.
    items = sorted((r for r in manifest["rows"] if r["split"] == "calibration" and r["task"] != TASKS[2]),
                   key=lambda r: digest(["smoke", r["id"]]))[:20]
    if len(items) != 20 or len({r["video_id"] for r in items}) < 2:
        raise RuntimeError("Need 20 calibration questions spanning multiple media items")
    report = parity_checks(scorer, items, index, all_passes=True)
    result = {"passed": True, "configuration": dataclasses.asdict(cfg), "environment": scorer.env,
              "comparison_kind": "prefix_parity" if cfg.scoring_mode == "verified_prefix" else "full_forward_repeatability",
              "prefix_cache_verified": cfg.scoring_mode == "verified_prefix",
              "manifest_sha256": digest(manifest), "media_sha256": digest(index),
              "code_sha256": file_hash(__file__), "tokens": scorer.tokens, "checks": report}
    atomic_json(Path(cfg.root) / cfg.model / f"smoke_f{cfg.frames}.json", result)
    return result


def parity_checks(scorer, items, index, all_passes=False):
    """Shared diagnostics: identical parity, generation, finiteness and VRAM checks."""
    report = []
    use_prefix = getattr(getattr(scorer, "cfg", None), "scoring_mode", "verified_prefix") == "verified_prefix"
    with Watchdog(600) as dog:
        for i, row in enumerate(items):
            # Every pass can change media length/positioning. Check all pass paths.
            for pass_name in (PASSES if all_passes and i == 0 else ("full",)):
                fresh, fh, fl = scorer.score(row, pass_name, index, cached=False)
                cached, ch, cl = scorer.score(row, pass_name, index, cached=use_prefix)
                import numpy as np
                ld, hd = float(np.max(np.abs(fl-cl))), float(np.max(np.abs(fh-ch)))
                entry = {"id": row["id"], "pass": pass_name, "logits_max_abs_diff": ld,
                         "hidden_max_abs_diff": hd, "fresh_answer": fresh["answer"],
                         "cached_answer" if use_prefix else "repeat_answer": cached["answer"],
                         "comparison_kind": "prefix_parity" if use_prefix else "full_forward_repeatability",
                         "fresh_seconds": fresh.get("seconds"),
                         "peak_vram_bytes": cached["peak_vram_bytes"]}
                if i < 10 and pass_name == "full":
                    gen = scorer.generation_check(row, index)
                    entry["generation"] = gen
                    if gen["generated_answer"] != fresh["answer"]:
                        raise RuntimeError("generate/Yes-No scoring disagreement; inspect prompt/token mapping")
                report.append(entry)
                if ld > .02 or hd > .02 or fresh["answer"] != cached["answer"]:
                    name = "Cache parity" if use_prefix else "Full-forward repeatability"
                    raise RuntimeError(name + " failed: " + json.dumps(entry))
                dog.touch()
    return report


def pre_smoke(cfg, raw_qa, media_dir, scorer=None):
    """Early private engineering diagnostic only; no terms, lineage, manifest or result gate.

    Caller notebook still requires owner GO. These raw items can later be in test;
    do not use their answers to tune the research protocol or claim a result.
    """
    import numpy as np
    record_scoring_amendment(cfg)
    pool = [dict(r) for r in raw_qa if r["task"] in TASKS[:2]]
    videos = sorted({str(r["video_id"]) for r in pool})
    if len(pool) < 20 or len(videos) < 3:
        raise RuntimeError("Early smoke needs 20 raw QA rows from at least three videos")
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(pool)).tolist()
    selected, covered = [], set()
    for i in order:
        if str(pool[i]["video_id"]) not in covered:
            selected.append(i)
            covered.add(str(pool[i]["video_id"]))
        if len(covered) == 3:
            break
    selected.extend(i for i in order if i not in selected)
    items = [pool[i] for i in selected[:20]]
    for row in items:
        row.update(video_id=str(row["video_id"]), question=row["text"],
                   id=digest(["pre-smoke", row["video_id"], row["task"], row["text"]])[:24])
    diagnostic_root = Path(cfg.root) / cfg.model / "pre_smoke_media"
    index = prepare_media(diagnostic_root, media_dir, {"rows": items}, cfg.frames, cfg.max_side)
    verify_media(index)
    started = time.monotonic()
    timeout = 1800 if cfg.model == "qwen" else 3600
    with Watchdog(timeout, deadline=started+1800 if cfg.model == "qwen" else None) as dog:
        reused_model = scorer is not None
        if scorer is None:
            scorer = OmniScorer(cfg)
        else:
            for key in ("model", "revision", "frames", "max_side"):
                if getattr(scorer.cfg, key) != getattr(cfg, key):
                    raise RuntimeError("Retained model/config mismatch: " + key)
            scorer.cfg = cfg
            scorer.env = dict(scorer.env, scoring_mode=cfg.scoring_mode)
            scorer.prefix = None
            scorer.prefix_key = None
            if hasattr(scorer, "torch"):
                scorer.torch.cuda.reset_peak_memory_stats()
        dog.touch()
        checks = parity_checks(scorer, items, index)
    result = {"kind": "engineering_diagnostic_only_not_a_result", "passed": True,
        "eligible_for_bulk_gate": False, "configuration": dataclasses.asdict(cfg),
        "reused_loaded_model": reused_model,
        "comparison_kind": "prefix_parity" if cfg.scoring_mode == "verified_prefix" else "full_forward_repeatability",
        "prefix_cache_verified": cfg.scoring_mode == "verified_prefix",
        "seconds_load_and_checks": time.monotonic()-started, "n_qa": len(items),
        "n_videos": len({r["video_id"] for r in items}), "selection_seed": cfg.seed,
        "raw_selection_sha256": digest(items), "environment": scorer.env,
        "tokens": scorer.tokens, "checks": checks, "media_sha256": digest(index)}
    times = [r["fresh_seconds"] for r in checks if r["pass"] == "full" and r["fresh_seconds"] is not None]
    if len(times) == len(items):
        result["observed_full_pass_seconds"] = cluster_ci(
            times, [r["video_id"] for r in items], lambda a: float(a.mean()))
    atomic_json(Path(cfg.root) / cfg.model / "pre_smoke.json", result)
    return scorer, result


def run_stage(scorer, cfg, manifest, index, owner_release, splits=("calibration", "validation")):
    require_go(owner_release)
    if getattr(getattr(scorer, "cfg", None), "scoring_mode", "verified_prefix") != cfg.scoring_mode:
        raise RuntimeError("Scorer mode differs from run configuration")
    check_terms(cfg.root, cfg.model)
    verify_media(index)
    smoke = read_json(Path(cfg.root) / cfg.model / f"smoke_f{cfg.frames}.json")
    if (not smoke.get("passed") or smoke["manifest_sha256"] != digest(manifest)
            or smoke["media_sha256"] != digest(index) or smoke["code_sha256"] != file_hash(__file__)
            or smoke["environment"] != scorer.env):
        raise RuntimeError("Matching smoke gate missing/stale")
    for key in ("revision", "frames", "max_side", "seed", "model", "scoring_mode"):
        if smoke["configuration"][key] != getattr(cfg, key):
            raise RuntimeError("Smoke configuration mismatch")
    if "test" in splits:
        freeze = read_json(Path(cfg.root) / cfg.model / "analysis_freeze.json")
        if (freeze.get("status") != "frozen_before_test" or freeze["manifest_sha256"] != digest(manifest)
                or freeze["signature"] != study_signature(cfg, manifest, index, cfg.root, scorer.env)):
            raise RuntimeError("Freeze CPU validation choices before opening test inference")
    signature = study_signature(cfg, manifest, index, cfg.root, scorer.env)
    directory = Path(cfg.root) / cfg.model / f"f{cfg.frames}" / cfg.stage
    items = [r for r in manifest["rows"] if r["split"] in splits
             and (r["task"] != TASKS[2] or cfg.stage in ("full", "no_media"))
             and int(digest(r["id"])[:16], 16) % cfg.workers == cfg.worker_index]
    # Group same media/donor prefixes for encode-once reuse across questions.
    items.sort(key=lambda r: (r["video_id"], r["task"], tuple(r["donors"]), r["id"]))
    worker = f"partition-{cfg.worker_index}-of-{cfg.workers}"
    with Journal(directory, worker, signature) as journal, Watchdog() as dog:
        start, added = time.monotonic(), 0
        for row in items:
            if row["id"] in journal.done:
                continue
            try:
                tick = time.monotonic()
                result, hidden, _ = scorer.score(row, cfg.stage, index)
                result.update({k: row[k] for k in ("id", "video_id", "source_group", "split", "task", "label", "object")})
                result.update({"pass": cfg.stage, "seconds": time.monotonic()-tick})
                journal.append(result, hidden if cfg.stage == "full" else None)
                dog.touch()
                added += 1
                if added % 100 == 0:
                    print(json.dumps({"new_rows": added, "seconds_per_question": (time.monotonic()-start)/added,
                                      "peak_vram_bytes": result["peak_vram_bytes"]}), flush=True)
                if added % 200 == 0:
                    journal.compact()
            except BaseException:
                traceback.print_exc()
                raise
        journal.compact()
    return {"new_rows": added, "eligible": len(items), "stage": cfg.stage, "signature": signature}


def cluster_ci(values, groups, statistic, resamples=2000, seed=SEED):
    """Paired resampling unit is connected source component; undefined draws discarded explicitly."""
    import numpy as np
    values = np.asarray(values)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if not len(values):
        return {"estimate": None, "ci95": None, "n": 0, "n_clusters": 0, "valid_resamples": 0}
    estimate = statistic(values)
    rng = np.random.default_rng(seed)
    buckets = {g: np.flatnonzero(groups == g) for g in unique}
    draws = []
    if len(unique) >= 2:
        for _ in range(resamples):
            indices = np.concatenate([buckets[g] for g in rng.choice(unique, len(unique), replace=True)])
            value = statistic(values[indices])
            if value is not None and np.isfinite(value):
                draws.append(value)
    return {"estimate": float(estimate) if estimate is not None and np.isfinite(estimate) else None,
            "ci95": np.quantile(draws, [.025, .975]).tolist() if len(draws) >= .9 * resamples else None,
            "n": len(values), "n_clusters": len(unique), "valid_resamples": len(draws),
            "requested_resamples": resamples, "seed": seed}


def rates(labels, answers, groups):
    import numpy as np
    y, p, g = np.asarray(labels), np.asarray(answers), np.asarray(groups)
    return {"accuracy": cluster_ci((y == p).astype(float), g, lambda x: float(x.mean())),
            "false_yes": cluster_ci(p[y == 0].astype(float), g[y == 0], lambda x: float(x.mean())),
            "recall": cluster_ci(p[y == 1].astype(float), g[y == 1], lambda x: float(x.mean())),
            "n_false_yes": int(((y == 0) & (p == 1)).sum()), "n": len(y),
            "n_negative": int((y == 0).sum()), "n_positive": int((y == 1).sum()),
            "n_correct": int((y == p).sum()), "n_true_yes": int(((y == 1) & (p == 1)).sum()),
            "power": "underpowered" if ((y == 0) & (p == 1)).sum() < 40 else "at_least_40_false_yes"}


def load_completed(root, cfg, manifest, index, environment, split):
    """Strict completeness: partial runs never become a metrics result."""
    import numpy as np
    signature = study_signature(cfg, manifest, index, root, environment)
    excluded = check_audit(root, manifest)
    freeze_path = Path(root) / cfg.model / "analysis_freeze.json"
    if freeze_path.exists() and read_json(freeze_path)["audit_sha256"] != file_hash(Path(root) / "audit.csv"):
        raise RuntimeError("Audit changed after analysis freeze")
    wanted = {r["id"]: r for r in manifest["rows"] if r["split"] == split and r["id"] not in excluded}
    stores = {}
    for stage in PASSES:
        directory = Path(root) / cfg.model / f"f{cfg.frames}" / stage
        rows = {}
        for path in directory.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["signature"] != signature:
                    raise RuntimeError("Mixed run/environment signature")
                if row["id"] in rows:
                    raise RuntimeError("Duplicate work across partitions")
                rows[row["id"]] = row
        required = {k for k, r in wanted.items() if r["task"] != TASKS[2] or stage in ("full", "no_media")}
        missing = required - rows.keys()
        if missing:
            raise RuntimeError(f"Incomplete {stage}/{split}: {len(missing)} missing questions; no extrapolation")
        stores[stage] = rows
    ids = sorted(wanted)
    data = []
    for key in ids:
        row = dict(wanted[key])
        row["margin"] = stores["full"][key]["margin"]
        row["no_media_margin"] = stores["no_media"][key]["margin"]
        row["contrast"] = (row["margin"] - np.mean([stores[f"cf{k}"][key]["margin"] for k in range(4)])
                           if row["task"] != TASKS[2] else None)
        row["blank_contrast"] = (row["margin"] - stores["blank"][key]["margin"] if row["task"] != TASKS[2] else None)
        frow = stores["full"][key]
        directory = Path(root) / cfg.model / f"f{cfg.frames}" / "full"
        feature = directory / frow["feature"]
        if file_hash(feature) != frow["feature_sha256"]:
            raise RuntimeError("Feature hash mismatch")
        row["hidden"] = np.load(feature)["hidden"].astype(np.float32)
        data.append(row)
    return data


def safe_auc(y, score):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def auc_ci(y, score, groups):
    import numpy as np
    return cluster_ci(np.column_stack([y, score]), groups, lambda a: safe_auc(a[:, 0], a[:, 1]))


def object_rates(calibration, rows, leave_group_out=False):
    """Smoothed model-Yes propensity, never held-out ground-truth object prevalence."""
    import numpy as np
    result = []
    for row in rows:
        pool = [r for r in calibration if r["task"] == row["task"] and r["object"] == row["object"]
                and (not leave_group_out or r["source_group"] != row["source_group"])]
        result.append((1 + sum(r["margin"] > 0 for r in pool)) / (2 + len(pool)))
    return np.asarray(result)


def baseline_matrix(calibration, rows, training=False):
    import numpy as np
    return np.column_stack([[r["margin"] for r in rows], [r["contrast"] for r in rows],
                            [r["no_media_margin"] for r in rows],
                            object_rates(calibration, rows, leave_group_out=training)])


class Residualizer:
    """Unlabeled calibration centering + ridge regression on answer margin; frozen transforms."""
    def fit(self, rows, layer, direction=None):
        import numpy as np
        self.layer, self.direction = layer, direction
        self.centers = {t: np.stack([r["hidden"][layer] for r in rows if r["task"] == t]).mean(0)
                        for t in {r["task"] for r in rows}}
        x = np.stack([r["hidden"][layer] - self.centers[r["task"]] for r in rows])
        m = np.column_stack([np.ones(len(rows)), [r["margin"] for r in rows]])
        self.beta = np.linalg.solve(m.T @ m + np.diag([1e-8, 1e-4]), m.T @ x)
        return self

    def transform(self, rows):
        import numpy as np
        x = np.stack([r["hidden"][self.layer] - self.centers[r["task"]] for r in rows])
        m = np.column_stack([np.ones(len(rows)), [r["margin"] for r in rows]])
        x = x - m @ self.beta
        if self.direction is not None:
            d = self.direction / max(np.linalg.norm(self.direction), 1e-12)
            if len(d) != x.shape[1]:
                raise ValueError("Unembedding and residual dimension mismatch")
            x = x - np.outer(x @ d, d)
        return x


def logistic(x, y, c):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="liblinear", max_iter=2000,
                                                              random_state=SEED)).fit(x, y)


def select_logistic(x, y, vx, vy):
    options = []
    for c in (.01, .1, 1.):
        model = logistic(x, y, c)
        auc = safe_auc(vy, model.predict_proba(vx)[:, 1])
        if auc is not None:
            options.append((auc, -c, model))
    if not options:
        raise ValueError("Both classes required in calibration and validation")
    best = max(options, key=lambda a: a[:2])
    return best[2], best[0]


def fit_task_probe(calibration, validation, task, unembedding):
    import numpy as np
    cal = [r for r in calibration if r["task"] == task and r["margin"] > 0]
    val = [r for r in validation if r["task"] == task and r["margin"] > 0]
    counts = {"calibration_yes": len(cal), "validation_yes": len(val),
              "calibration_false_yes": sum(not r["label"] for r in cal),
              "validation_false_yes": sum(not r["label"] for r in val)}
    if not cal or not val or len({r["label"] for r in cal}) < 2 or len({r["label"] for r in val}) < 2:
        return {"status": "underpowered_missing_class", "counts": counts}
    y, vy = np.array([1-r["label"] for r in cal]), np.array([1-r["label"] for r in val])
    bx, bv = baseline_matrix(calibration, cal, True), baseline_matrix(calibration, val)
    base, _ = select_logistic(bx, y, bv, vy)
    # Use all unlabeled calibration rows for task centering/margin residualization.
    choices = []
    for layer in range(cal[0]["hidden"].shape[0]):
        transform = Residualizer().fit(calibration, layer)
        x, vx = transform.transform(cal), transform.transform(val)
        model, score = select_logistic(np.column_stack([bx, x]), y, np.column_stack([bv, vx]), vy)
        choices.append((score, -layer, transform, model))
    score, neg_layer, transform, augmented = max(choices, key=lambda a: a[:2])
    layer = -neg_layer
    x, vx = transform.transform(cal), transform.transform(val)
    standalone, _ = select_logistic(x, y, vx, vy)
    # Same selected layer/C and same preprocessing for the permutation control.
    shuffled_y = np.random.default_rng(SEED).permutation(y)
    shuffled = logistic(np.column_stack([bx, x]), shuffled_y,
                        augmented.named_steps["logisticregression"].C)
    projected_transform = Residualizer().fit(calibration, layer, unembedding)
    px, pv = projected_transform.transform(cal), projected_transform.transform(val)
    projected, _ = select_logistic(np.column_stack([bx, px]), y, np.column_stack([bv, pv]), vy)
    return {"status": "fitted", "counts": counts, "layer": layer,
            "baseline": base, "augmented": augmented, "standalone": standalone,
            "shuffled": shuffled, "transform": transform, "projected": projected,
            "projected_transform": projected_transform,
            "validation_margin_edges": np.quantile([r["margin"] for r in val], [.2, .4, .6, .8]).tolist()}


def threshold_select(rows, field, risk=False):
    """Max validation false-yes reduction at <=.02 point recall cost, ties favor coverage."""
    import numpy as np
    y = np.array([r["label"] for r in rows])
    original = np.array([r["margin"] > 0 for r in rows])
    score = np.asarray([r[field] for r in rows], dtype=float)
    if not (np.any(y == 0) and np.any(y == 1)) or not np.isfinite(score).all():
        raise ValueError("Threshold selection needs finite scores and positive/negative labels")
    values = np.unique(score)
    thresholds = np.r_[values[0]-1, values, values[-1]+1]
    original_recall = original[y == 1].mean()
    candidates = []
    for threshold in thresholds:
        accepted = original & ((score <= threshold) if risk else (score >= threshold))
        if original_recall - accepted[y == 1].mean() <= .02 + 1e-12:
            candidates.append((accepted[y == 0].mean(), -accepted.mean(), float(threshold)))
    return min(candidates) [2]


def probe_scores(bundle, calibration, rows):
    import numpy as np
    if bundle["status"] != "fitted":
        return None
    b = baseline_matrix(calibration, rows)
    x = bundle["transform"].transform(rows)
    p = bundle["projected_transform"].transform(rows)
    return {"baseline": bundle["baseline"].predict_proba(b)[:, 1],
            "augmented": bundle["augmented"].predict_proba(np.column_stack([b, x]))[:, 1],
            "probe_only": bundle["standalone"].predict_proba(x)[:, 1],
            "shuffled": bundle["shuffled"].predict_proba(np.column_stack([b, x]))[:, 1],
            "projected": bundle["projected"].predict_proba(np.column_stack([b, p]))[:, 1]}


def freeze_analysis(root, cfg, manifest, index, environment, unembedding):
    """CPU only. Fits calibration, selects validation, saves immutable choices before test."""
    import numpy as np
    import joblib
    check_audit(root, manifest, require_complete=True)
    calibration = load_completed(root, cfg, manifest, index, environment, "calibration")
    validation = load_completed(root, cfg, manifest, index, environment, "validation")
    signature = study_signature(cfg, manifest, index, root, environment)
    out = Path(root) / cfg.model
    freeze_path = out / "analysis_freeze.json"
    if freeze_path.exists():
        existing = read_json(freeze_path)
        if existing["signature"] != signature:
            raise RuntimeError("Frozen analysis signature mismatch")
        if file_hash(out / "analysis_bundle.joblib") != existing["bundle_sha256"]:
            raise RuntimeError("Analysis bundle changed")
        return existing
    bundles, thresholds, summary = {}, {}, {}
    for task in TASKS[:2]:
        val = [dict(r) for r in validation if r["task"] == task]
        if not val:
            raise RuntimeError("No validation rows for a primary task")
        bundle = fit_task_probe(calibration, validation, task, np.asarray(unembedding))
        bundles[task] = bundle
        thresholds[task] = {field: threshold_select(val, field) for field in ("contrast", "blank_contrast")}
        summary[task] = {k: bundle[k] for k in ("status", "counts")}
        if bundle["status"] == "fitted":
            yes = [r for r in val if r["margin"] > 0]
            scores = probe_scores(bundle, calibration, yes)
            mapping = {r["id"]: float(s) for r, s in zip(yes, scores["augmented"])}
            for row in val:
                row["probe_risk"] = mapping.get(row["id"], 0.)
            thresholds[task]["probe_risk"] = threshold_select(val, "probe_risk", risk=True)
            summary[task]["selected_layer"] = bundle["layer"]
    path = out / "analysis_bundle.joblib"
    # Bundle is local trusted analysis state; never load a third-party pickle/joblib file.
    joblib.dump({"bundles": bundles, "calibration": calibration, "thresholds": thresholds}, path)
    freeze = {"status": "frozen_before_test", "signature": signature, "manifest_sha256": digest(manifest),
              "audit_sha256": file_hash(Path(root) / "audit.csv"),
              "bundle_sha256": file_hash(path), "thresholds": thresholds, "probe_choices": summary,
              "environment": environment, "unembedding_sha256": digest(np.asarray(unembedding).tolist())}
    atomic_json(freeze_path, freeze)
    return freeze


def evaluate_policy(rows, accepted, resamples=2000):
    import numpy as np
    y = np.array([r["label"] for r in rows])
    original = np.array([r["margin"] > 0 for r in rows])
    g = np.array([r["source_group"] for r in rows])
    accepted = np.asarray(accepted, dtype=bool)
    false_delta = cluster_ci((original[y == 0].astype(float)-accepted[y == 0]).astype(float), g[y == 0],
                             lambda a: float(a.mean()), resamples)
    recall_drop = cluster_ci((original[y == 1].astype(float)-accepted[y == 1]).astype(float), g[y == 1],
                             lambda a: float(a.mean()), resamples)
    supported = (false_delta["ci95"] is not None and false_delta["ci95"][0] > 0
                 and recall_drop["estimate"] is not None and recall_drop["estimate"] <= .02 + 1e-12)
    return {"false_yes_before": cluster_ci(original[y == 0].astype(float), g[y == 0], lambda a: float(a.mean()), resamples),
            "false_yes_after": cluster_ci(accepted[y == 0].astype(float), g[y == 0], lambda a: float(a.mean()), resamples),
            "false_yes_reduction": false_delta, "recall_drop": recall_drop,
            "recall_before": cluster_ci(original[y == 1].astype(float), g[y == 1], lambda a: float(a.mean()), resamples),
            "recall_after": cluster_ci(accepted[y == 1].astype(float), g[y == 1], lambda a: float(a.mean()), resamples),
            "share_answered": cluster_ci((~original | accepted).astype(float), g, lambda a: float(a.mean()), resamples),
            "verdict": "ship_demo_flag" if supported else "did_not_ship",
            "n": len(rows), "n_false_yes_before": int(((y == 0) & original).sum()),
            "power": "underpowered" if ((y == 0) & original).sum() < 40 else "at_least_40_false_yes"}


def evaluate_test(root, cfg, manifest, index, environment, owner_release):
    require_go(owner_release)
    import numpy as np
    import joblib
    freeze = read_json(Path(root) / cfg.model / "analysis_freeze.json")
    if freeze["signature"] != study_signature(cfg, manifest, index, root, environment):
        raise RuntimeError("Freeze does not match run")
    path = Path(root) / cfg.model / "analysis_bundle.joblib"
    if file_hash(path) != freeze["bundle_sha256"]:
        raise RuntimeError("Bundle hash mismatch")
    bundle = joblib.load(path)
    test = load_completed(root, cfg, manifest, index, environment, "test")
    terms = check_terms(root, cfg.model)
    metrics = {"study": "CounterPrior v2", "date": "2026-11", "model": MODEL_IDS[cfg.model],
               "scoring_mode": cfg.scoring_mode,
               "prefix_cache_verified": cfg.scoring_mode == "verified_prefix",
               "revision": cfg.revision, "signature": freeze["signature"], "tasks": {},
               "method": "AVCD-style queried-modality contrast; not published attentive-mask AVCD",
               "uncertainty": "2000 paired grouping-unit bootstrap draws; conditional on fixed donor pool; see lineage limitations",
               "limitations": manifest.get("limitations", []), "object_rule": manifest.get("object_rule", OBJECT_RULE),
               "dataset_terms_decision": terms["dataset"]["decision"],
               "public_avhbench_media_allowed": False,
               "publication_policy": "Aggregate numbers and our own text only; no AVHBench audio, video or frames" if
                   terms["dataset"]["decision"] == "no_licence_stated_private_research_eval" else
                   "No public media authorization granted by this evaluation; separate permission required",
               "audit_sha256": freeze["audit_sha256"],
               "preregistration_sha256": file_hash(Path(root) / "PREREGISTRATION.md")}
    for task in TASKS:
        rows = [r for r in test if r["task"] == task]
        if not rows:
            metrics["tasks"][task] = {"status": "not_available"}
            continue
        y, g = np.array([r["label"] for r in rows]), [r["source_group"] for r in rows]
        original = np.array([r["margin"] > 0 for r in rows])
        out = {"full": rates(y, original, g),
               "no_media": rates(y, [r["no_media_margin"] > 0 for r in rows], g)}
        no_media = np.array([r["no_media_margin"] > 0 for r in rows])
        out["full_minus_no_media"] = {
            "accuracy": cluster_ci((original == y).astype(float)-(no_media == y), g, lambda a: float(a.mean())),
            "false_yes": cluster_ci(original[y == 0].astype(float)-no_media[y == 0], np.array(g)[y == 0], lambda a: float(a.mean())),
            "recall": cluster_ci(original[y == 1].astype(float)-no_media[y == 1], np.array(g)[y == 1], lambda a: float(a.mean()))}
        if task == TASKS[2]:
            metrics["tasks"][task] = out
            continue
        thresholds = bundle["thresholds"][task]
        for field in ("contrast", "blank_contrast"):
            accepted = original & (np.array([r[field] for r in rows]) >= thresholds[field])
            out[field] = evaluate_policy(rows, accepted)
            out[field]["validation_threshold"] = thresholds[field]
        pb = bundle["bundles"][task]
        yes = [r for r in rows if r["margin"] > 0]
        if pb["status"] != "fitted" or not yes:
            out["probe"] = {"status": "underpowered_missing_class", "calibration_validation": pb["counts"]}
        else:
            scores = probe_scores(pb, bundle["calibration"], yes)
            target = np.array([1-r["label"] for r in yes])
            group = [r["source_group"] for r in yes]
            probe = {"selected_layer": pb["layer"], "n_yes": len(yes), "n_false_yes": int(target.sum()),
                     "power": "underpowered" if target.sum() < 40 else "at_least_40_false_yes",
                     "auroc": {k: auc_ci(target, v, group) for k, v in scores.items()}}
            paired = np.column_stack([target, scores["augmented"], scores["baseline"]])
            def delta(a):
                a1, a0 = safe_auc(a[:, 0], a[:, 1]), safe_auc(a[:, 0], a[:, 2])
                return a1-a0 if a1 is not None and a0 is not None else None
            probe["delta_auroc_augmented_over_baseline"] = cluster_ci(paired, group, delta)
            bins = np.digitize([r["margin"] for r in yes], pb["validation_margin_edges"])
            probe["within_margin_quintiles"] = {}
            for q in range(5):
                mask = bins == q
                probe["within_margin_quintiles"][str(q+1)] = {
                    "delta_auroc": cluster_ci(paired[mask], np.array(group)[mask], delta),
                    "n_false_yes": int(target[mask].sum()),
                    "power": "underpowered" if target[mask].sum() < 40 else "at_least_40_false_yes"}
            risk = {r["id"]: s for r, s in zip(yes, scores["augmented"])}
            accepted = np.array([r["margin"] > 0 and risk[r["id"]] <= thresholds["probe_risk"] for r in rows])
            probe["policy"] = evaluate_policy(rows, accepted)
            # Paired, measured difference versus contrast alone, not just each versus no fix.
            contrast_accepted = original & (np.array([r["contrast"] for r in rows]) >= thresholds["contrast"])
            probe["false_yes_reduction_vs_contrast"] = cluster_ci(
                (contrast_accepted[y == 0].astype(float)-accepted[y == 0]), np.array(g)[y == 0], lambda a: float(a.mean()))
            # Descriptive test risk-coverage points; threshold selection remains validation-only.
            probe["risk_coverage"] = []
            for threshold in np.linspace(0., 1., 21):
                keep = np.array([not (r["margin"] > 0) or risk[r["id"]] <= threshold for r in rows])
                errors = np.array([int((r["margin"] > 0) != bool(r["label"])) for r in rows])
                probe["risk_coverage"].append({"threshold": float(threshold), "n_answered": int(keep.sum()),
                    "coverage": cluster_ci(keep.astype(float), g, lambda a: float(a.mean())),
                    "risk": cluster_ci(errors[keep].astype(float), np.array(g)[keep], lambda a: float(a.mean()))})
            out["probe"] = probe
        metrics["tasks"][task] = out
    atomic_json(Path(root) / cfg.model / "metrics.json", metrics)
    return metrics


def geometry_and_transfer(root, cfg, manifest, index, environment, unembedding):
    """Secondary analysis only, frozen-layer transfer. No test choice feeds main analysis."""
    import numpy as np
    import joblib
    freeze = read_json(Path(root) / cfg.model / "analysis_freeze.json")
    path = Path(root) / cfg.model / "analysis_bundle.joblib"
    if file_hash(path) != freeze["bundle_sha256"]:
        raise RuntimeError("Bundle hash mismatch")
    bundle = joblib.load(path)
    cal = bundle["calibration"]
    test = load_completed(root, cfg, manifest, index, environment, "test")
    result = {"geometry_split": "calibration", "transfer_split": "test", "transfer": {}, "geometry": []}
    for source, target in ((TASKS[0], TASKS[1]), (TASKS[1], TASKS[0])):
        train_bundle = bundle["bundles"][source]
        ceiling_bundle = bundle["bundles"][target]
        rows = [r for r in test if r["task"] == target and r["margin"] > 0]
        if not rows or train_bundle["status"] != "fitted" or ceiling_bundle["status"] != "fitted":
            result["transfer"][source] = {"status": "underpowered_missing_class"}
            continue
        y, g = np.array([1-r["label"] for r in rows]), [r["source_group"] for r in rows]
        scores = probe_scores(train_bundle, cal, rows)["probe_only"]
        ceiling = probe_scores(ceiling_bundle, cal, rows)["probe_only"]
        data = np.column_stack([y, scores, ceiling])
        def ratio(a):
            ta, ca = safe_auc(a[:, 0], a[:, 1]), safe_auc(a[:, 0], a[:, 2])
            return (ta-.5)/(ca-.5) if ta is not None and ca is not None and ca > .5 else None
        result["transfer"][source] = {"target": target, "auroc": auc_ci(y, scores, g),
            "in_task_ceiling_auroc": auc_ci(y, ceiling, g), "above_chance_ratio": cluster_ci(data, g, ratio),
            "n_false_yes": int(y.sum()), "power": "underpowered" if y.sum() < 40 else "at_least_40_false_yes"}
    def cosine(a, b):
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        return float(a@b/denominator) if denominator > 0 else None
    pool = [[r for r in cal if r["task"] == task and r["margin"] > 0] for task in TASKS[:2]]
    if all(rows and len({r["label"] for r in rows}) == 2 for rows in pool):
        for layer in range(pool[0][0]["hidden"].shape[0]):
            xs = [np.stack([r["hidden"][layer] for r in rows]) for rows in pool]
            ys = [np.array([1-r["label"] for r in rows]) for rows in pool]
            directions = [x[y == 1].mean(0)-x[y == 0].mean(0) for x, y in zip(xs, ys)]
            rng = np.random.default_rng(SEED + layer)
            null = []
            for _ in range(2000):
                shuffled = [rng.permutation(y) for y in ys]
                ds = [x[y == 1].mean(0)-x[y == 0].mean(0) for x, y in zip(xs, shuffled)]
                value = cosine(*ds)
                if value is not None:
                    null.append(value)
            # Bootstrap both task directions with the same source-component draw.
            joined = [dict(r, task_index=i) for i, rows in enumerate(pool) for r in rows]
            values = np.arange(len(joined))
            def geometry_stat(indices, reference=None):
                ds = []
                for i in range(2):
                    selected = [joined[int(j)] for j in indices if joined[int(j)]["task_index"] == i]
                    if len({r["label"] for r in selected}) < 2:
                        return None
                    a = np.stack([r["hidden"][layer] for r in selected if r["label"] == 0]).mean(0)
                    b = np.stack([r["hidden"][layer] for r in selected if r["label"] == 1]).mean(0)
                    ds.append(a-b)
                return cosine(*ds) if reference is None else cosine(ds[reference], np.asarray(unembedding))
            groups = [r["source_group"] for r in joined]
            result["geometry"].append({"layer": layer,
                "cross_task_cosine": cluster_ci(values, groups, geometry_stat),
                "null_interval95": np.quantile(null, [.025, .975]).tolist() if null else None,
                "unembedding_cosines": [cluster_ci(values, groups, lambda a, i=i: geometry_stat(a, i)) for i in range(2)],
                "n_yes": [len(r) for r in pool], "n_false_yes": [int(y.sum()) for y in ys],
                "power": ["underpowered" if y.sum() < 40 else "at_least_40_false_yes" for y in ys]})
    atomic_json(Path(root) / cfg.model / "secondary_metrics.json", result)
    return result
