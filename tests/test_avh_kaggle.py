"""Synthetic CPU contracts; no Kaggle/GPU results are represented by these tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import avh_common as avh
import avh_kaggle as kg


def parent(n=6000):
    rows = []
    for i in range(n):
        rows.append({"id": f"q{i:05d}", "video_id": f"v{i:05d}", "source_group": f"g{i//2:05d}",
            "source_ids": [f"s{i//2}"], "task": avh.TASKS[i % 3], "label": (i // 3) % 2,
            "question": f"Is there object {i}?", "object": f"object {i}", "is_swapped": False,
            "swap_status_known": False, "split": ("calibration", "validation", "test")[i % 3],
            "donors": ["unused"] * 4})
    return {"rows": rows, "limitations": [avh.LINEAGE_LIMITATION], "object_rule": avh.OBJECT_RULE}


def identity():
    return {"model": "gemma", "model_revision": "a" * 40, "code_git_hash": "b" * 40,
            "code_sha256": "c" * 64, "bundle_sha256": "d" * 64}


@pytest.mark.parametrize("n_workers", [6, 8])
def test_workers_cover_every_question_exactly_once_and_two_per_account(n_workers):
    questions = [f"q{i}" for i in range(5293)]
    parts = [{q for q in questions if kg.worker_for(q, n_workers) == worker} for worker in range(n_workers)]
    assert set.union(*parts) == set(questions)
    assert sum(map(len, parts)) == len(questions)
    assert all(parts[i].isdisjoint(parts[j]) for i in range(n_workers) for j in range(i))
    assert [kg.account_workers(a, n_workers) for a in range(n_workers//2)] == [
        (a*2, a*2+1) for a in range(n_workers//2)]
    assert [kg.worker_for(q, n_workers) for q in questions] == [kg.worker_for(q, n_workers) for q in questions]
    with pytest.raises(ValueError):
        kg.account_workers(4, 8)


def test_lean_sample_has_exact_counts_source_disjointness_and_local_donors():
    original = parent()
    before = avh.digest(original)
    sample = kg.lean_manifest(original)
    assert len(sample["rows"]) == 2000
    assert sample["split_counts"] == {"calibration": 400, "validation": 400, "test": 1200}
    assert avh.digest(original) == before
    assert sample == kg.lean_manifest(original)
    sources = {s: {r["source_group"] for r in sample["rows"] if r["split"] == s}
               for s in sample["split_counts"]}
    assert all(sources[a].isdisjoint(sources[b]) for a in sources for b in sources if a != b)
    by_video = {r["video_id"]: r for r in original["rows"]}
    # All donors from selected rows must be disjoint from targets and from other selected splits.
    for r in sample["rows"]:
        assert len(r["donors"]) == len(set(r["donors"])) == 2
        for d in r["donors"]:
            group = by_video[d]["source_group"]
            assert group != r["source_group"]
            assert not any(group in groups for s, groups in sources.items() if s != r["split"])
    for split in sample["split_counts"]:
        assert {(r["task"], r["label"]) for r in sample["rows"] if r["split"] == split} == {
            (t, label) for t in avh.TASKS for label in (0, 1)}


def test_expansion_keeps_old_questions_splits_and_donors():
    original = parent()
    small, large = kg.lean_manifest(original), kg.lean_manifest(original, n=3000)
    lookup = {r["id"]: r for r in large["rows"]}
    assert small["group_assignment_sha256"] == large["group_assignment_sha256"]
    assert all(r == lookup[r["id"]] for r in small["rows"])
    assert kg.lean_manifest(original, n=1000, model="qwen")["split_counts"]["test"] == 600


def test_pairs_keep_blank_and_no_media_and_matching_has_only_two():
    rows = parent(12)
    pairs = kg.safe_pairs(rows)
    for r in rows["rows"]:
        names = {p for q, p in pairs if q == r["id"]}
        assert names == ({"full", "no_media"} if r["task"] == avh.TASKS[2] else set(kg.PASSES))


def test_relocatable_index_preserves_hashes_without_editing_parent(tmp_path):
    media = tmp_path / "media/f8-s224/00001"
    media.mkdir(parents=True)
    (media / "audio.wav").write_bytes(b"synthetic")
    original = {"00001": {"directory": "/old/drive/media/f8-s224/00001",
        "file_hashes": {"audio.wav": avh.file_hash(media / "audio.wav")}}}
    before = avh.digest(original)
    companion = kg.portable_index(original, tmp_path)
    avh.atomic_json(tmp_path / "kaggle_media_index.json", companion)
    assert avh.digest(original) == before
    assert companion["00001"]["directory"] == "media/f8-s224/00001"
    resolved = kg.resolved_index(tmp_path)
    assert resolved["00001"]["directory"] == str(media.resolve())
    avh.verify_media(resolved)
    companion["00001"]["directory"] = "../outside"
    avh.atomic_json(tmp_path / "kaggle_media_index.json", companion)
    with pytest.raises(ValueError, match="escapes"):
        kg.resolved_index(tmp_path)


def test_pair_journal_resumes_by_pair_and_saves_only_full_features(tmp_path):
    manifest = parent(12)
    worker = kg.worker_for(manifest["rows"][0]["id"], 8)
    row = manifest["rows"][0]
    folder = tmp_path / "pairs"
    with kg.PairJournal(folder, identity(), worker, worker // 2, 8, manifest) as journal:
        journal.append(row, "full", {"margin": 1., "answer": 1}, np.ones((3, 4)))
        journal.question_complete()
        with pytest.raises(ValueError, match="only"):
            journal.append(row, "blank", {"margin": 0.}, np.ones((3, 4)))
    with (folder / "predictions.jsonl").open("ab") as f:
        f.write(b'{"torn":')
    with kg.PairJournal(folder, identity(), worker, worker // 2, 8, manifest) as journal:
        assert journal.done == {(row["id"], "full")}
        journal.append(row, "full", {"margin": 999.}, np.zeros((3, 4)))
        journal.append(row, "blank", {"margin": 0., "answer": 0})
    records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    assert len(records) == 2 and records[0]["margin"] == 1.
    h = np.load(folder / records[0]["feature"], allow_pickle=False)
    assert h.shape == (250, 3, 4) and h.dtype == np.float16
    assert "feature" not in records[1]
    assert all({"question_id", "pass", "worker_id", "account_index", "model", "model_revision",
                "code_git_hash", "code_sha256", "timestamp"} <= r.keys() for r in records)


def test_merge_checks_coverage_duplicates_and_feature_corruption(tmp_path):
    manifest = parent(12)
    folders = []
    for worker in range(8):
        folder = tmp_path / str(worker)
        with kg.PairJournal(folder, identity(), worker, worker // 2, 8, manifest) as journal:
            for row in manifest["rows"]:
                if kg.worker_for(row["id"], 8) == worker:
                    for name in kg.passes_for(row):
                        journal.append(row, name, {"margin": 1., "answer": 1, "yes_logsumexp": 1., "no_logsumexp": 0.},
                            np.ones((2, 3)) if name == "full" else None)
        folders.append(folder)
    assert len(kg.validate_outputs(folders, manifest, identity(), 8)) == len(kg.safe_pairs(manifest))
    with pytest.raises(RuntimeError, match="Duplicate"):
        kg.validate_outputs(folders + folders, manifest, identity(), 8)
    with pytest.raises(RuntimeError, match="Incomplete"):
        kg.validate_outputs([], manifest, identity(), 8)
    feature = next(tmp_path.rglob("full-hidden-00000.npy"))
    array = np.load(feature, mmap_mode="r+", allow_pickle=False)
    array[0, 0, 0] = 42
    array.flush()
    with pytest.raises(RuntimeError, match="changed"):
        kg.validate_outputs(folders, manifest, identity(), 8)


def test_sealed_test_analysis_rejected_without_valid_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(avh, "study_signature", lambda *args: "synthetic")
    monkeypatch.setattr(avh, "check_audit", lambda *args: set())
    cfg = avh.Config(str(tmp_path), "gemma", "a" * 40, k=2)
    manifest = dict(parent(12), test_scoring_policy="sealed_precompute")
    with pytest.raises(RuntimeError, match="Sealed"):
        avh.load_completed(tmp_path, cfg, manifest, {}, {}, "test")
    avh.atomic_json(tmp_path / "gemma/analysis_freeze.json", {"status": "pending"})
    with pytest.raises(RuntimeError, match="freeze"):
        avh.load_completed(tmp_path, cfg, manifest, {}, {}, "test")


def test_launcher_will_not_load_gpu_with_unresolved_reference(tmp_path):
    with pytest.raises(RuntimeError, match="pending"):
        kg.launch_account(tmp_path, {"account_index": 0, "n_workers": 8,
            "mode": "timing", "reference": "pending"}, tmp_path / "out")


def test_checkpoint_restore_refuses_other_bundle_or_worker(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    avh.atomic_json(source / "identity.json", identity())
    (source / "predictions.jsonl").write_text(json.dumps({"worker_id": 2}) + "\n")
    with pytest.raises(RuntimeError, match="version"):
        kg.restore_checkpoint(source, tmp_path / "dest", 2, "wrong")
    with pytest.raises(RuntimeError, match="worker"):
        kg.restore_checkpoint(source, tmp_path / "dest", 0, identity()["bundle_sha256"])


def test_cpu_export_preserves_sources_and_produces_checked_private_bundle(tmp_path):
    import zipfile
    root, support = tmp_path / "drive", tmp_path / "support"
    root.mkdir(); support.mkdir()
    original = parent()
    for i, r in enumerate(original["rows"]):
        r.update(video_id=f"v{i % 120:03d}", source_group=f"g{i % 120:03d}")
    avh.atomic_json(root / "manifest.json", original)
    for name in ("lineage", "released_qa", "qa_resolution", "release_inventory"):
        avh.atomic_json(root / f"{name}.json", {})
    (root / "audit.csv").write_text("id,decision\n")
    (support / "PREREGISTRATION_KAGGLE.md").write_text("Synthetic policy fixture, not research")
    terms = {}
    for key in ("dataset", "gemma", "qwen"):
        evidence = root / f"{key}_LICENSE.txt"
        evidence.write_text("Synthetic research permission")
        terms[key] = {"evidence_file": evidence.name, "sha256": avh.file_hash(evidence)}
    avh.atomic_json(root / "terms_review.json", terms)
    for model in avh.MODEL_IDS:
        avh.atomic_json(root / model / "model_metadata.json", {"synthetic": True})
        avh.atomic_json(root / model / "scoring_amendment.json", {"prefix_evidence_file": "failure.json"})
        avh.atomic_json(root / model / "failure.json", {"synthetic": True})
    index = {}
    for vid in sorted({r["video_id"] for r in original["rows"]}):
        directory = root / "media/f8-s224" / vid
        directory.mkdir(parents=True)
        (directory / "audio.wav").write_bytes(b"synthetic bytes, not an audio recording")
        index[vid] = {"directory": str(directory), "file_hashes": {"audio.wav": avh.file_hash(directory / "audio.wav")}}
    avh.atomic_json(root / "media_index_f8.json", index)
    before = avh.digest(original)
    archive = root / "exports/data.zip"
    archive.parent.mkdir()
    # A stopped serial export is a valid prefix ZIP. Reuse verified entries, reject a bad one.
    partial = archive.with_suffix(".zip.partial")
    vids = sorted(index)
    with zipfile.ZipFile(partial, "w") as previous:
        for vid in vids[:2]:
            previous.write(root / "media/f8-s224" / vid / "audio.wav", f"media/f8-s224/{vid}/audio.wav")
        previous.writestr(f"media/f8-s224/{vids[2]}/audio.wav", b"invalid partial bytes")
    summary = kg.export_private_study(root, archive, support, "a" * 40, read_workers=4)
    assert summary["reused_files"] == 2
    assert partial.is_file()
    assert summary["media_clips"] == 120
    assert summary["sha256"] == avh.file_hash(archive)
    assert avh.digest(avh.read_json(root / "manifest.json")) == before
    target = tmp_path / "kaggle_input"
    with zipfile.ZipFile(archive) as z:
        assert "CHECKSUMS.sha256" in z.namelist()
        z.extractall(target)
    assert kg.verify_bundle(target)["status"] == "cpu_verified"
    assert avh.read_json(target / "lean_manifest_n2000.json")["split_counts"]["test"] == 1200
    assert avh.read_json(target / "run_control.json")["n_workers"] == 6
    with pytest.raises(FileExistsError):
        kg.export_private_study(root, archive, support, "a" * 40)


def test_two_process_launcher_pins_gpus_and_keeps_secret_out_of_files(tmp_path, monkeypatch):
    import types
    calls = []
    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(kg.subprocess, "Popen", popen)
    token = "SYNTHETIC_TOKEN_NOT_A_CREDENTIAL"
    kg.launch_account(tmp_path, {"account_index": 1, "n_workers": 6,
        "mode": "run", "reference": "fresh_unbatched_gpu"}, tmp_path / "account_1", hf_token=token)
    assert [call[1]["env"]["CUDA_VISIBLE_DEVICES"] for call in calls] == ["0", "1"]
    jobs = [avh.read_json(call[0][-1]) for call in calls]
    assert [j["worker_id"] for j in jobs] == [2, 3]
    assert [Path(j["output"]).name for j in jobs] == ["out_1_2", "out_1_3"]
    assert all(token not in p.read_text() for p in tmp_path.rglob("*.json"))


def test_calibration_receipts_support_addon_without_reading_test(tmp_path):
    manifest = dict(parent(36), n=2000, group_assignment_sha256="synthetic-groups")
    for i, row in enumerate(manifest["rows"]):
        row["split"] = ("calibration", "validation", "test")[(i // 6) % 3]
    receipts = []
    for worker in range(6):
        with kg.PairJournal(tmp_path / str(worker), identity(), worker, worker//2, 6, manifest) as journal:
            for row in manifest["rows"]:
                if row["split"] == "calibration" and kg.worker_for(row["id"], 6) == worker:
                    for name in kg.passes_for(row):
                        journal.append(row, name, {"answer": 1, "margin": 1.},
                            np.ones((2, 3)) if name == "full" else None)
            receipt = kg.calibration_receipt(journal, manifest, worker, 6, "synthetic-launch")
            assert receipt is not None and receipt["scope"] == "calibration_only"
            receipts.append(receipt)
    accounts = [receipts[i:i+2] for i in (0, 2, 4)]
    records = kg.records_from_calibration_receipts(accounts, manifest, 6)
    assert len(records) == sum(r["split"] == "calibration" for r in manifest["rows"])
    decision = kg.calibration_power_decision(manifest, records,
        dict(manifest, n=3000), "calibration_addon")
    assert decision["chosen_n"] == 3000
    with pytest.raises(RuntimeError, match="exactly one"):
        kg.records_from_calibration_receipts(accounts + accounts, manifest, 6)
    wrong = [*records, (None, {"question_id": next(r["id"] for r in manifest["rows"] if r["split"] == "test"),
                              "pass": "full", "answer": 1})]
    with pytest.raises(RuntimeError, match="cannot inspect"):
        kg.calibration_power_decision(manifest, wrong, dict(manifest, n=3000), "calibration_addon")


def test_native_batch_padding_matches_unbatched_reference_on_synthetic_cpu(monkeypatch):
    import types
    import torch
    import avh_kaggle_worker as wk
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    class Processor:
        tokenizer = types.SimpleNamespace(padding_side="right")
        def apply_chat_template(self, messages, **kwargs):
            return "media" if messages[0]["content"][0]["type"] == "image" else "text"
        def __call__(self, text, **kwargs):
            images = kwargs.get("images")
            if any(t == "media" for t in text):
                assert len(images) == len(text), "Gemma requires one image group per prompt"
                assert all(isinstance(group, list) for group in images), "Do not flatten images across samples"
                assert [len(group) for group in images] == [8 if t == "media" else 0 for t in text]
                assert len(kwargs["audio"]) == sum(t == "media" for t in text)
            else:
                assert images is None and "audio" not in kwargs, "No-media must remain truly text-only"
            seqs = [[2, 3, 4, 5] if t == "media" else [5] for t in text]
            width = max(map(len, seqs))
            assert self.tokenizer.padding_side == "left"
            ids = torch.tensor([[0]*(width-len(s)) + s for s in seqs])
            return {"input_ids": ids, "attention_mask": (ids != 0).long(),
                    "mm_token_type_ids": (ids == 2).long()}
    class Model:
        def __call__(self, input_ids, attention_mask, position_ids, **kwargs):
            assert "mm_token_type_ids" in kwargs
            for p, m in zip(position_ids, attention_mask):
                assert p[m.bool()].tolist() == list(range(int(m.sum())))
            h = (input_ids.float() + position_ids).unsqueeze(-1).repeat(1, 1, 3)
            logits = torch.stack([h[:, :, 0], h[:, :, 0] + 1, h[:, :, 0] - 2], -1)
            return types.SimpleNamespace(logits=logits, hidden_states=(h, h*2))
    fake_torch = types.SimpleNamespace(**{name: getattr(torch, name) for name in
        ("Tensor", "float16", "inference_mode", "stack", "isfinite", "logsumexp")},
        cuda=types.SimpleNamespace(synchronize=lambda: None, max_memory_allocated=lambda: 0))
    scorer = types.SimpleNamespace(torch=fake_torch, cfg=types.SimpleNamespace(model="gemma"),
        processor=Processor(), model=Model(), tokens={"yes": [1], "no": [0]},
        media=lambda row, name, index: (None, None, "text") if name == "no_media" else ([object() for _ in range(8)], np.zeros(4), "media"))
    batch = wk.BatchScorer(scorer)
    row = {"question": "Synthetic fixture?"}
    actual, _ = batch.score(row, ["full", "no_media"], {}, diagnostic=True)
    for name, output in zip(("full", "no_media"), actual):
        reference = batch.score(row, [name], {}, diagnostic=True)[0][0]
        assert wk.check_pair(reference, output) == {"logits_max_abs_diff": 0., "hidden_max_abs_diff": 0.}
    production, _ = batch.score(row, ["full", "no_media"], {}, diagnostic=False)
    assert production[0]["hidden"] is not None and production[1]["hidden"] is None
    for names in (["full", "cf0"], ["full", "cf0", "cf1", "blank", "no_media"],
                  ["no_media", "full"], ["no_media"]):
        outputs, _ = batch.score(row, names, {}, diagnostic=True)
        for name, output in zip(names, outputs):
            ref = batch.score(row, [name], {}, diagnostic=True)[0][0]
            assert wk.check_pair(ref, output)["logits_max_abs_diff"] == 0.


def test_gemma_smoke_checks_each_pass_serially():
    from types import SimpleNamespace
    import torch
    from avh_kaggle_worker import startup_smoke

    class SerialBatch:
        scorer = SimpleNamespace(cfg=SimpleNamespace(model="gemma"),
                                 generation_check=lambda row, index: {"generated_answer": 1})

        def score(self, row, names, index, diagnostic=False):
            assert len(names) == 1
            return [{"pass": names[0], "logits": np.array([1.0]),
                     "hidden": np.array([1.0]), "answer": 1}], 0.0

    rows = [{"id": str(i), "task": avh.TASKS[0]} for i in range(10)]
    batch = SerialBatch()
    batch.torch = torch
    smoke = startup_smoke(batch, rows, {}, 0)
    assert smoke["comparison_kind"] == "serial_full_forward_repeatability"
    assert smoke["validated_batch_sizes"] == [1]
    assert smoke["candidates"]["1"]["comparisons"] == 25


def test_sdpa_guard_only_changes_fully_masked_boolean_queries():
    import torch
    from avh_kaggle_worker import guard_empty_sdpa_queries

    original = torch.tensor([[[[False, False, False], [False, True, False], [False, True, True]]]])
    guarded = guard_empty_sdpa_queries(lambda mask: mask, torch.bool)
    actual = guarded(original)
    assert actual[0, 0, 0].all()
    assert torch.equal(actual[..., 1:, :], original[..., 1:, :])
    assert not original[0, 0, 0].any()
    additive_mask = original.float()
    assert guarded(additive_mask) is additive_mask
    assert guarded(None) is None


def test_mde_rejects_noncalibration_and_reports_undefined_without_inventing_numbers():
    with pytest.raises(RuntimeError, match="calibration only"):
        kg.calibration_mde([{"split": "test"}])
    result = kg.calibration_mde([])
    assert all(v["mde_delta_auroc_approx"] is None for v in result["tasks"].values())


def test_kaggle_notebooks_are_unexecuted_and_first_worker_cell_is_only_assignment():
    import ast
    for name in ("01_export_cpu", "02_worker_kaggle", "03_merge_analyse_cpu"):
        notebook = avh.read_json(ROOT / "notebooks" / f"{name}.ipynb")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                assert cell["execution_count"] is None and cell["outputs"] == []
        if name == "02_worker_kaggle":
            body = ast.parse("".join(notebook["cells"][0]["source"])).body
            assert len(body) == 2 and all(isinstance(node, ast.Assign) for node in body)
            assert [node.targets[0].id for node in body] == ["N_WORKERS", "ACCOUNT_INDEX"]


def test_complete_sealed_k2_import_loads_npy_shards_and_keeps_test_closed(tmp_path, monkeypatch):
    import shutil

    dataset, output, study = tmp_path / "dataset", tmp_path / "raw", tmp_path / "analysis"
    dataset.mkdir(); output.mkdir()
    manifest = dict(parent(36), model="gemma", n=2000, seed=avh.SEED, k=2,
        parent_manifest_sha256="synthetic-parent", group_assignment_sha256="synthetic-groups",
        test_scoring_policy="sealed_precompute")
    for i, r in enumerate(manifest["rows"]):
        r["split"] = ("calibration", "validation", "test")[(i // 6) % 3]
        r["donors"] = r["donors"][:2]
    for name in ("release_inventory.json", "terms_review.json"):
        avh.atomic_json(dataset / name, {})
    for name in ("gemma_LICENSE.txt", "qwen_LICENSE.txt"):
        (dataset / name).write_text("Synthetic fixture, no model run")
    avh.atomic_json(dataset / "gemma/model_metadata.json", {"synthetic": True})
    avh.atomic_json(dataset / "gemma/scoring_amendment.json", {"prefix_evidence_file": "failure.json"})
    avh.atomic_json(dataset / "gemma/failure.json", {"synthetic": True})
    (dataset / "runtime").mkdir()
    (dataset / "runtime/PREREGISTRATION_KAGGLE.md").write_text("Synthetic policy fixture")
    avh.atomic_json(dataset / "kaggle_media_index.json", {})
    meta = dict(identity(), n_workers=6, selection_family={k: manifest[k] for k in
        ("parent_manifest_sha256", "group_assignment_sha256", "seed", "k")}, environment={"kind": "synthetic_cpu"})
    meta["code_sha256"] = avh.digest({name: avh.file_hash(ROOT / "notebooks" / name)
        for name in ("avh_common.py", "avh_kaggle.py", "avh_kaggle_worker.py")})
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ("avh_common.py", "avh_kaggle.py", "avh_kaggle_worker.py"):
        shutil.copyfile(ROOT / "notebooks" / name, runtime / name)
    amendment = runtime / "GEMMA_SERIAL_AMENDMENT.md"
    shutil.copyfile(ROOT / "docs" / amendment.name, amendment)
    meta["serial_amendment_sha256"] = avh.file_hash(amendment)
    monkeypatch.setattr(kg, "__file__", str(runtime / "avh_kaggle.py"))
    folders = []
    for worker in range(6):
        folder = output / str(worker)
        with kg.PairJournal(folder, meta, worker, worker//2, 6, manifest) as journal:
            for row in manifest["rows"]:
                if kg.worker_for(row["id"], 6) != worker:
                    continue
                for name in kg.passes_for(row):
                    margin = 2. if name == "full" else .5
                    journal.append(row, name, {"answer": 1, "margin": margin,
                        "yes_logsumexp": margin, "no_logsumexp": 0.}, np.ones((3, 4)) if name == "full" else None)
        folders.append(folder)
    cfg, index, env = kg.import_for_analysis(dataset, manifest, folders, study, 6)
    assert (study / "PREREGISTRATION.md").read_bytes().endswith(amendment.read_bytes())
    rows = avh.load_completed(study, cfg, manifest, index, env, "calibration")
    assert len(rows) == 12 and all(r["hidden"].shape == (3, 4) for r in rows)
    assert all(r["contrast"] == 1.5 for r in rows if r["task"] != avh.TASKS[2])
    assert not (study / "gemma/f8/cf2").exists()
    with pytest.raises(RuntimeError, match="Sealed"):
        avh.load_completed(study, cfg, manifest, index, env, "test")
    with pytest.raises(RuntimeError, match="HALT 2"):
        kg.evaluate_sealed_once(study, cfg, manifest, index, env)
