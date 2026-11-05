"""CPU contract tests only. Synthetic fixtures are never research measurements."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "notebooks" / "avh_common.py"
spec = importlib.util.spec_from_file_location("avh_common", SOURCE)
avh = importlib.util.module_from_spec(spec)
sys.modules["avh_common"] = avh
spec.loader.exec_module(avh)


@pytest.mark.parametrize("dump_fails", [False, True])
def test_watchdog_terminates_even_if_notebook_diagnostic_output_fails(monkeypatch, dump_fails):
    import types
    captured = {}
    class Terminated(BaseException):
        pass
    def fake_thread(target, daemon):
        captured["monitor"] = target
        return types.SimpleNamespace(start=lambda: None, join=lambda **kw: None)
    def dump(**kwargs):
        assert kwargs == {"file": 2, "all_threads": True}
        if dump_fails:
            raise OSError("notebook diagnostic stream unavailable")
    def terminate(code):
        captured["exit_code"] = code
        raise Terminated()
    monkeypatch.setattr(avh.threading, "Thread", fake_thread)
    monkeypatch.setattr(avh.faulthandler, "dump_traceback", dump)
    monkeypatch.setattr(avh.os, "_exit", terminate)
    watchdog = avh.Watchdog(seconds=600)
    watchdog.last = 0
    monkeypatch.setattr(avh.time, "monotonic", lambda: 601)
    monkeypatch.setattr(watchdog.stop, "wait", lambda seconds: False)
    with watchdog:
        with pytest.raises(Terminated):
            captured["monitor"]()
    assert captured["exit_code"] == 124


@pytest.mark.parametrize("bundled", [True, False])
def test_license_resolution_uses_repo_file_or_verified_publisher_link(tmp_path, monkeypatch, bundled):
    import types
    import urllib.request
    card = "---\nlicense: apache-2.0\nlicense_link: https://ai.google.dev/gemma/docs/gemma_4_license\n---\n"
    (tmp_path / "README.md").write_text(card, encoding="utf-8")
    (tmp_path / "LICENSE.txt").write_text("Synthetic bundled research license", encoding="utf-8")
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = lambda: types.SimpleNamespace(model_info=lambda repo: types.SimpleNamespace(
        sha="a"*40, siblings=[types.SimpleNamespace(rfilename="LICENSE.txt" if bundled else "README.md")]))
    downloads = []
    def download(repo, name, revision):
        downloads.append(name)
        assert revision == "a"*40
        return str(tmp_path / name)
    fake.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def geturl(self):
            return "https://ai.google.dev/gemma/apache_2"
        def read(self):
            return b"<html>Navigation<pre><code>Apache License\nVersion 2.0\nGrant of Copyright License</code></pre></html>"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: Response())
    result = avh.resolve_model("gemma")
    assert result["revision"] == "a"*40
    assert result["license_sha256"] == avh.hashlib.sha256(result["license_text"].encode()).hexdigest()
    assert downloads == (["LICENSE.txt"] if bundled else ["README.md"])
    assert ("model_card_sha256" in result) == (not bundled)
    if not bundled:
        assert "Navigation" not in result["license_text"]
        (tmp_path / "README.md").write_text(card.replace("apache-2.0", "unverified"))
        with pytest.raises(RuntimeError, match="verified publisher"):
            avh.resolve_model("gemma")


def release_fixture():
    lineage, qa = [], []
    for i in range(60):
        vid = f"v{i:03d}"
        origins = [f"s{i}"]
        swapped = i >= 50
        if swapped:
            origins.append(f"s{i-50}")
        questions = {f"Question {t} {i}": f"object-{i%3}" for t in avh.TASKS}
        lineage.append({"video_id": vid, "source_ids": origins, "is_swapped": swapped,
                        "provenance": "synthetic test fixture", "object_by_question": questions})
        for task, question in zip(avh.TASKS, questions):
            qa.append({"video_id": vid, "task": task, "text": question, "label": "Yes" if i % 2 else "No"})
    return qa, lineage


def private_terms_fixture(root):
    avh.atomic_json(root / "release_inventory.json", {"terms_files": [], "kind": "synthetic fixture"})
    texts = {
        "gemma": "Apache License\nVersion 2.0\n2. Grant of Copyright License. Synthetic grant.\n3. Grant of Patent License.",
        "qwen": "Qwen RESEARCH LICENSE AGREEMENT\nNon-Commercial means research or evaluation purposes only.\n2. Grant of Rights\nFOR NON-COMMERCIAL PURPOSES ONLY\n3. Redistribution"}
    for model, text in texts.items():
        avh.atomic_json(root / model / "model_metadata.json", {"repo": avh.MODEL_IDS[model], "revision": "a"*40,
            "license_text": text, "license_sha256": avh.hashlib.sha256(text.encode()).hexdigest()})


def test_private_terms_setup_requires_approval_and_preserves_existing_inference_identity(tmp_path):
    private_terms_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="Explicit owner"):
        avh.save_approved_private_terms(tmp_path)
    review = avh.save_approved_private_terms(tmp_path, True, "synthetic owner")
    assert review["dataset"]["decision"] == "no_licence_stated_private_research_eval"
    assert review["qwen"]["applicable_passage"].startswith("2. Grant of Rights")
    assert avh.read_json(tmp_path / "publication_policy.json")["public_avhbench_media_allowed"] is False
    before = (tmp_path / "terms_review.json").read_bytes()
    ledger = tmp_path / "gemma" / "f8" / "full" / "partition.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("synthetic ledger")
    assert avh.save_approved_private_terms(tmp_path, True, "synthetic owner") == review
    with pytest.raises(RuntimeError, match="Terms changed after inference"):
        avh.save_approved_private_terms(tmp_path, True, "different reviewer")
    assert (tmp_path / "terms_review.json").read_bytes() == before


def test_single_media_catalog_preserves_opaque_ids_and_finds_duplicate_names(tmp_path, monkeypatch):
    (tmp_path / "nested").mkdir()
    (tmp_path / "01162.mp4").write_bytes(b"synthetic")
    (tmp_path / "nested" / "01162.mp4").write_bytes(b"duplicate")
    avh.atomic_json(tmp_path / "01162.json", {"source_ids": ["real-origin"], "is_swapped": False})
    catalog = avh.media_catalog(tmp_path)
    assert len(catalog["video"]["01162"]) == 2
    def forbidden(*args):
        raise AssertionError("Provided catalog must prevent a new full directory scan")
    monkeypatch.setattr(avh, "media_catalog", forbidden)
    qa = [{"video_id": "01162", "task": avh.TASKS[0], "text": "Is the dog making sound in the audio?", "label": "Yes"}]
    lineage = avh.auto_lineage(tmp_path, qa, tmp_path, catalog=catalog)
    assert lineage[0]["source_ids"] == ["real-origin"]
    assert lineage[0]["object_by_question"][qa[0]["text"]] == "dog"


def test_shared_preparation_does_not_require_completed_audit_or_load_a_model(tmp_path, monkeypatch):
    private_terms_fixture(tmp_path)
    qa, _ = release_fixture()
    avh.atomic_json(tmp_path / "released_qa.json", qa)
    (tmp_path / "release" / "unpacked").mkdir(parents=True)
    calls = []
    def media(root, video_dir, manifest, frames, max_side, catalog=None):
        calls.append(catalog)
        return {r["video_id"]: {"kind": "synthetic media fixture"} for r in manifest["rows"]}
    monkeypatch.setattr(avh, "prepare_media", media)
    monkeypatch.setattr(avh, "OmniScorer", lambda *args: pytest.fail("Preparation must not load a model"))
    manifest, index, result = avh.prepare_shared_study(tmp_path, True, "synthetic owner")
    assert result["status"] == "shared_data_ready" and len(index) == 60
    assert len(manifest["rows"]) == 180 and len(calls) == 1
    assert avh.check_audit(tmp_path, manifest) == set()
    assert avh.read_json(tmp_path / "audit_summary.json")["n_label_pending"] > 0


def test_release_duplicate_resolution_excludes_conflicts_without_relabeling(tmp_path):
    def row(vid, label):
        return {"video_id": vid, "task": avh.TASKS[0], "text": "Is the dog making sound in the audio?", "label": label}
    raw = [row("same", "Yes"), row("same", "Yes"), row("conflict", "Yes"), row("conflict", "No"), row("unique", "No")]
    original = json.loads(json.dumps(raw))
    resolved, report = avh.resolve_released_qa(tmp_path, raw)
    assert raw == original
    assert resolved == [raw[0], raw[4]]
    assert report["removed_occurrences"] == 3
    assert report["resolved_binary_questions"] == 2
    assert report["decisions"][0]["removed_indices"] == [1]
    assert report["decisions"][1]["removed_indices"] == [2, 3]
    assert avh.resolve_released_qa(tmp_path, raw)[1] == report


def test_go_gate_and_model_constraints(tmp_path):
    with pytest.raises(RuntimeError, match="HALT"):
        avh.require_go("")
    avh.require_go("GO")
    with pytest.raises(ValueError):
        avh.Config(str(tmp_path), "gemma", "main")
    with pytest.raises(ValueError):
        avh.Config(str(tmp_path), "gemma", "a"*40, frames=4)


def test_attention_backend_keeps_gemma_audio_eager_and_qwen_unchanged():
    gemma = avh.attention_backend("gemma")
    assert gemma["audio_config"] == "eager"
    assert gemma["text_config"] == gemma["vision_config"] == gemma[""] == "sdpa"
    assert avh.attention_backend("qwen") == "sdpa"


def failed_prefix_fixture(root):
    avh.atomic_json(root / "gemma" / "cache_diagnostic_v3.json", {
        "status": "comparison_complete", "logits_max_abs_diff": .0625,
        "hidden_max_abs_diff": .125, "answers": [1, 1]})


def test_full_forward_dispatch_never_reuses_prefix_or_reloads_weights(tmp_path):
    import types
    failed_prefix_fixture(tmp_path)
    scorer = object.__new__(avh.OmniScorer)
    cfg = avh.Config(str(tmp_path), "gemma", "a"*40)
    scorer.cfg, scorer.env, scorer.model = cfg, {}, object()
    weights = scorer.model
    scorer.prefix, scorer.prefix_key = object(), "previous"
    updated = avh.use_full_forward(scorer, cfg)
    assert scorer.model is weights
    assert scorer.prefix is None and scorer.prefix_key is None
    assert updated.scoring_mode == "full_forward"
    assert avh.read_json(tmp_path / "gemma" / "scoring_amendment.json")["prefix_cache_status"] == "failed_gate_disabled"
    scorer.inputs = lambda *args: ({"test_input": 1}, 1, "key")
    calls = []
    def forward(inputs, **kwargs):
        calls.append(kwargs)
        assert inputs == {"test_input": 1}
        return object()
    scorer._forward = forward
    scorer.extract = lambda output: ({"answer": 1, "margin": 1.}, np.ones((2, 3)), np.ones(4))
    scorer.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None, max_memory_allocated=lambda: 1))
    result, _, _ = scorer.score({}, "full", {})
    assert calls == [{"use_cache": False}] and result["scoring_mode"] == "full_forward"
    with pytest.raises(RuntimeError, match="Prefix reuse is disabled"):
        scorer.score({}, "full", {}, cached=True)


def test_retained_full_forward_smoke_labels_repeatability_and_keeps_formal_gate_closed(tmp_path, monkeypatch):
    qa, _ = release_fixture()
    failed_prefix_fixture(tmp_path)
    original = avh.Config(str(tmp_path), "gemma", "a"*40)
    cfg = avh.dataclasses.replace(original, scoring_mode="full_forward")
    class Retained:
        env, tokens = {"kind": "synthetic"}, {"yes": [1], "no": [0]}
        def __init__(self):
            self.cfg, self.calls, self.generations = original, 0, 0
        def score(self, row, stage, index, cached=None):
            assert cached is False
            self.calls += 1
            return {"answer": 1, "peak_vram_bytes": 1, "seconds": 1.}, np.ones((2, 3)), np.ones(4)
        def generation_check(self, row, index):
            self.generations += 1
            return {"generated_answer": 1}
    retained = Retained()
    def forbidden_load(*args):
        raise AssertionError("Must reuse the existing model")
    monkeypatch.setattr(avh, "OmniScorer", forbidden_load)
    monkeypatch.setattr(avh, "prepare_media", lambda *args: {})
    monkeypatch.setattr(avh, "verify_media", lambda *args: None)
    returned, result = avh.pre_smoke(cfg, qa, tmp_path, scorer=retained)
    assert returned is retained and retained.calls == 40 and retained.generations == 10
    assert result["comparison_kind"] == "full_forward_repeatability"
    assert result["prefix_cache_verified"] is False and result["reused_loaded_model"] is True
    assert result["observed_full_pass_seconds"]["n"] == 20
    assert all("repeat_answer" in r and "cached_answer" not in r for r in result["checks"])
    assert not (tmp_path / "gemma" / "smoke_f8.json").exists()


def test_bulk_rejects_scorer_with_different_execution_mode(tmp_path):
    import types
    cfg = avh.Config(str(tmp_path), "gemma", "a"*40, scoring_mode="full_forward")
    scorer = types.SimpleNamespace(cfg=avh.dataclasses.replace(cfg, scoring_mode="verified_prefix"))
    with pytest.raises(RuntimeError, match="mode differs"):
        avh.run_stage(scorer, cfg, {}, {}, "GO")


def test_fallback_uses_actual_model_specific_failure_and_preserves_original_code_record(tmp_path):
    import types
    cfg = avh.Config(str(tmp_path), "qwen", "a"*40)
    evidence = {"status": "comparison_complete", "kind": "captured_original_prefix_failure",
                "logits_max_abs_diff": .1, "hidden_max_abs_diff": .2, "answers": [1, 1]}
    folder = tmp_path / "qwen"
    avh.atomic_json(folder / "cache_parity_failure.json", evidence)
    scorer = types.SimpleNamespace(cfg=cfg, env={}, model=object())
    cfg = avh.use_full_forward(scorer, cfg, evidence_file="cache_parity_failure.json")
    path = folder / "scoring_amendment.json"
    saved = avh.read_json(path)
    assert saved["prefix_evidence_file"] == "cache_parity_failure.json"
    assert saved["model"] == avh.MODEL_IDS["qwen"]
    saved["code_sha256"] = "code_at_original_decision"
    avh.atomic_json(path, saved)
    avh.record_scoring_amendment(cfg)
    assert avh.read_json(path) == saved
    avh.atomic_json(folder / "cache_parity_failure.json", dict(evidence, hidden_max_abs_diff=.3))
    with pytest.raises(RuntimeError, match="amendment changed"):
        avh.record_scoring_amendment(cfg)


def test_recovery_reuses_model_from_failed_smoke_without_a_global_scorer(tmp_path, monkeypatch):
    import types
    cfg = avh.Config(str(tmp_path), "qwen", "a"*40)
    retained = types.SimpleNamespace(cfg=cfg, env={}, model=object())
    # Give this synthetic traceback the same code filename/name as the real boundary.
    namespace = {"retained": retained}
    exec(compile('def parity_checks():\n scorer = retained\n entry = {"id":"fixture", "pass":"full", "logits_max_abs_diff":0.1, "hidden_max_abs_diff":0.2, "fresh_answer":1, "cached_answer":1}\n raise RuntimeError("Cache parity failed")\n', str(SOURCE), "exec"), namespace)
    try:
        namespace["parity_checks"]()
    except RuntimeError as error:
        monkeypatch.setattr(sys, "last_traceback", error.__traceback__, raising=False)
        recovered, updated = avh.recover_full_forward()
    assert recovered is retained and updated.scoring_mode == "full_forward"
    evidence = avh.read_json(tmp_path / "qwen" / "cache_parity_failure.json")
    assert evidence["kind"] == "captured_original_prefix_failure"
    assert evidence["failed_check"]["id"] == "fixture"


def test_terms_need_real_unchanged_evidence_and_applicable_passage(tmp_path):
    evidence = tmp_path / "terms.txt"
    evidence.write_text("Synthetic license fixture: research use permitted.")
    entry = {"decision": "research_use_permitted", "evidence_file": "terms.txt",
             "sha256": avh.file_hash(evidence), "applicable_passage": "research use permitted",
             "source_url": "https://example.invalid/synthetic", "reviewer": "test"}
    avh.atomic_json(tmp_path / "terms_review.json", {"dataset": entry, "gemma": entry})
    avh.check_terms(tmp_path, "gemma")
    evidence.write_text("Changed evidence")
    with pytest.raises(RuntimeError, match="changed"):
        avh.check_terms(tmp_path, "gemma")


def test_all_duplicate_questions_and_changed_media_fail(tmp_path):
    qa, lineage = release_fixture()
    with pytest.raises(ValueError, match="Duplicate QA"):
        avh.make_manifest(tmp_path, qa + [qa[0]], lineage)
    evidence = tmp_path / "frame.png"
    evidence.write_bytes(b"synthetic bytes")
    index = {"v": {"directory": str(tmp_path), "file_hashes": {"frame.png": avh.file_hash(evidence)}}}
    avh.verify_media(index)
    evidence.write_bytes(b"modified bytes")
    with pytest.raises(RuntimeError, match="Media bytes"):
        avh.verify_media(index)


def test_both_source_origins_stay_in_one_split_and_donors_are_local(tmp_path):
    qa, lineage = release_fixture()
    manifest = avh.make_manifest(tmp_path, qa, lineage)
    by_video = {r["video_id"]: r for r in manifest["rows"]}
    seen = {}
    for row in manifest["rows"]:
        for origin in row["source_ids"]:
            if origin in seen:
                assert seen[origin] == row["split"]
            seen[origin] = row["split"]
        assert len(row["donors"]) == len(set(row["donors"])) == 4
        for vid in row["donors"]:
            donor = by_video[vid]
            assert donor["split"] == row["split"]
            assert donor["source_group"] != row["source_group"]
            assert not donor["is_swapped"]
    assert avh.make_manifest(tmp_path, qa, lineage) == manifest
    qa[0]["label"] = "No" if qa[0]["label"] == "Yes" else "Yes"
    with pytest.raises(RuntimeError, match="frozen"):
        avh.make_manifest(tmp_path, qa, lineage)


def test_unknown_lineage_cannot_be_replaced_with_release_ids():
    with pytest.raises(ValueError, match="Unknown source"):
        avh.connected_groups([{"video_id": "001", "source_ids": []}])


def test_journal_resume_recovers_only_torn_tail_and_preserves_features(tmp_path):
    with avh.Journal(tmp_path, "p0", "sig") as journal:
        journal.append({"id": "a", "margin": .2}, np.ones((3, 4), dtype=np.float32))
        journal.compact()
    with open(tmp_path / "p0.jsonl", "ab") as f:
        f.write(b'{"id": "incomplete')
    with avh.Journal(tmp_path, "p0", "sig") as journal:
        assert journal.done == {"a"}
        journal.append({"id": "a"}, None)
        journal.append({"id": "b", "margin": -.1}, None)
        feature = tmp_path / journal.rows[0]["feature"]
        assert avh.file_hash(feature) == journal.rows[0]["feature_sha256"]
        assert np.load(feature)["hidden"].dtype == np.float16
    assert len((tmp_path / "p0.jsonl").read_text().splitlines()) == 2
    with pytest.raises(RuntimeError, match="mismatch"):
        with avh.Journal(tmp_path, "p0", "new-signature"):
            pass


def test_journal_rejects_concurrent_partition_and_interior_corruption(tmp_path):
    with avh.Journal(tmp_path, "p0", "sig"):
        with pytest.raises(FileExistsError):
            with avh.Journal(tmp_path, "p0", "sig"):
                pass
    (tmp_path / "p0.jsonl").write_text('{broken}\n{"id":"a","signature":"sig"}\n')
    with pytest.raises(RuntimeError, match="interior"):
        with avh.Journal(tmp_path, "p0", "sig"):
            pass


def test_cluster_bootstrap_is_paired_deterministic_and_handles_missing_class():
    values = np.array([[0, .1, .1], [1, .9, .9], [0, .2, .2], [1, .8, .8]])
    def delta(a):
        b, c = avh.safe_auc(a[:, 0], a[:, 1]), avh.safe_auc(a[:, 0], a[:, 2])
        return b-c if b is not None and c is not None else None
    a = avh.cluster_ci(values, ["one", "one", "two", "two"], delta)
    assert a == avh.cluster_ci(values, ["one", "one", "two", "two"], delta)
    assert a["estimate"] == 0 and a["ci95"] == [0., 0.]
    assert a["n_clusters"] == 2 and a["valid_resamples"] == 2000
    missing = avh.auc_ci([1, 1], [.2, .8], ["a", "b"])
    assert missing["estimate"] is None and missing["ci95"] is None


def test_policy_refuses_false_yes_win_when_recall_loss_exceeds_rule():
    rows = [{"label": i % 2, "margin": 1., "source_group": str(i//2)} for i in range(100)]
    accept = np.array([r["label"] == 1 for r in rows])
    win = avh.evaluate_policy(rows, accept, resamples=100)
    assert win["verdict"] == "ship_demo_flag"
    assert win["recall_drop"]["estimate"] == 0
    assert win["share_answered"]["estimate"] == .5
    accept[1] = False
    accept[3] = False
    loss = avh.evaluate_policy(rows, accept, resamples=100)
    assert loss["verdict"] == "did_not_ship"
    assert loss["recall_drop"]["estimate"] == .04


def test_threshold_select_respects_positive_recall_and_full_no_answers():
    rows = [{"label": 1, "margin": 2., "contrast": 1.},
            {"label": 1, "margin": -1., "contrast": 100.},
            {"label": 0, "margin": 1., "contrast": -.5},
            {"label": 0, "margin": -1., "contrast": 3.}]
    threshold = avh.threshold_select(rows, "contrast")
    assert threshold == 1.
    accepted = np.array([r["margin"] > 0 and r["contrast"] >= threshold for r in rows])
    assert accepted.tolist() == [True, False, False, False]


def test_object_propensity_never_uses_heldout_labels_or_own_source():
    cal = [{"task": "t", "object": "dog", "source_group": "a", "margin": 1.},
           {"task": "t", "object": "dog", "source_group": "b", "margin": -1.}]
    assert avh.object_rates(cal, [cal[0]], leave_group_out=True)[0] == pytest.approx(1/3)
    test = dict(cal[0], source_group="test", label=0)
    before = avh.object_rates(cal, [test])[0]
    test["label"] = 1
    assert avh.object_rates(cal, [test])[0] == before


def test_residualizer_freezes_calibration_and_projects_direction():
    cal = [{"task": t, "margin": float(i), "hidden": np.array([[i, 2*i+1, (-1.)**i]])}
           for t in ("a", "b") for i in range(10)]
    transform = avh.Residualizer().fit(cal, 0, np.array([0., 0., 1.]))
    x = transform.transform(cal)
    assert np.max(np.abs(x[:, -1])) == 0
    assert np.max(np.abs(x[:, :2])) < 1e-3
    centers = {k: v.copy() for k, v in transform.centers.items()}
    transform.transform([dict(cal[0], hidden=np.array([[1000., 2000., 9.]]))])
    for key in centers:
        np.testing.assert_array_equal(centers[key], transform.centers[key])


def test_prefix_slice_keeps_full_attention_for_suffix_and_removes_media():
    scorer = object.__new__(avh.OmniScorer)
    scorer.cfg = avh.Config("unused", "gemma", "a"*40)
    full = {"input_ids": np.arange(8)[None, :], "position_ids": np.arange(8)[None, :],
            "attention_mask": np.ones((1, 8)), "pixel_values": np.ones((2, 3, 4, 4)),
            "mm_token_type_ids": np.array([[1, 1, 1, 1, 0, 0, 0, 0]]),
            "token_type_ids": np.arange(8)[None, :], "custom_mm_type_ids": np.arange(8)[None, :]}
    prefix = scorer.split_inputs(full, 4)
    suffix = scorer.split_inputs(full, 4, suffix=True)
    assert prefix["input_ids"].shape[-1] == 4 and "pixel_values" in prefix
    assert suffix["input_ids"].tolist() == [[4, 5, 6, 7]]
    assert suffix["attention_mask"].shape[-1] == 8
    assert "pixel_values" not in suffix
    for key in ("token_type_ids", "custom_mm_type_ids"):
        assert prefix[key].tolist() == [[0, 1, 2, 3]]
        assert suffix[key].tolist() == [[4, 5, 6, 7]]


def test_inventory_reports_raw_schema_before_requiring_explicit_mapping(tmp_path, capsys):
    rows = [{"clip": "001", "kind": "audio_task", "prompt": "Is there a dog?", "answer": "No"}]
    (tmp_path / "qa.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="explicit schema_mapping"):
        avh.inventory_release(tmp_path)
    output = capsys.readouterr().out
    assert '"clip": "001"' in output and "all_key_names" in output and "raw_record_count" in output
    qa, info = avh.inventory_release(tmp_path, {"video_id": "clip", "task": "kind", "text": "prompt", "label": "answer"},
                                     task_mapping={"audio_task": avh.TASKS[0]})
    assert qa[0]["video_id"] == "001" and qa[0]["task"] == avh.TASKS[0]
    assert info["all_task_names"] == ["audio_task"]
    assert info["schema_mapping"]["text"] == "prompt"


def test_auto_lineage_recovers_explicit_pairs_and_discloses_clip_fallback(tmp_path):
    qa, _ = release_fixture()
    qa[0]["source_ids"] = ["original-a"]
    qa[0]["is_swapped"] = False
    qa[3]["audio_source_id"], qa[3]["video_source_id"] = "original-a", "original-b"
    qa[3]["is_swapped"] = True
    lineage = avh.auto_lineage(tmp_path, qa, tmp_path)
    groups = avh.connected_groups(lineage)
    assert groups["v000"] == groups["v001"]
    assert groups["v010"] == "v010"
    assert next(r for r in lineage if r["video_id"] == "v010")["lineage_unrecoverable"]
    manifest = avh.make_manifest(tmp_path, qa, lineage)
    assert avh.LINEAGE_LIMITATION in manifest["limitations"]
    assert avh.read_json(tmp_path / "lineage_rules.json")["object_rule"] == avh.OBJECT_RULE
    assert avh.object_key("Is the dog making sound in the audio?") == "dog"
    assert avh.object_key("Is there a cat in the video?") == "cat"
    assert avh.object_key("Unmatched wording?") == "question:unmatched wording"


def test_partial_audit_is_analysis_only_and_completion_gates_freeze(tmp_path, monkeypatch):
    qa, lineage = release_fixture()
    manifest = avh.make_manifest(tmp_path, qa, lineage)
    path = avh.create_audit(tmp_path, manifest)
    assert avh.check_audit(tmp_path, manifest) == set()
    cfg = avh.Config(str(tmp_path), "gemma", "a"*40)
    with pytest.raises(RuntimeError, match="before analysis freeze"):
        avh.freeze_analysis(tmp_path, cfg, manifest, {}, {}, np.ones(3))
    import csv
    with path.open(newline="", encoding="utf-8") as f:
        sheet = list(csv.DictReader(f))
    label_row = next(r for r in sheet if r["label_audit"] == "True")
    label_row.update(decision="exclude", reviewer="synthetic fixture")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sheet[0].keys())
        writer.writeheader()
        writer.writerows(sheet)
    assert avh.check_audit(tmp_path, manifest) == {label_row["id"]}


def test_audit_edit_does_not_change_inference_signature(tmp_path):
    cfg = avh.Config(str(tmp_path), "gemma", "a"*40)
    avh.atomic_json(tmp_path / "terms_review.json", {"synthetic": True})
    (tmp_path / "PREREGISTRATION.md").write_text("Synthetic protocol")
    before = avh.study_signature(cfg, {}, {}, tmp_path, {})
    (tmp_path / "audit.csv").write_text("unfinished")
    assert avh.study_signature(cfg, {}, {}, tmp_path, {}) == before
    (tmp_path / "audit.csv").write_text("completed")
    assert avh.study_signature(cfg, {}, {}, tmp_path, {}) == before


def test_private_owner_terms_decision_does_not_waive_model_license(tmp_path):
    evidence = tmp_path / "README.txt"
    evidence.write_text("Synthetic README without a dataset license")
    base = {"decision": "no_licence_stated_private_research_eval", "evidence_file": evidence.name,
            "sha256": avh.file_hash(evidence), "source_url": "https://example.invalid", "reviewer": "owner"}
    model = dict(base, decision="research_use_permitted", applicable_passage="Synthetic README")
    avh.atomic_json(tmp_path / "terms_review.json", {"dataset": base, "gemma": model})
    assert avh.check_terms(tmp_path, "gemma")["dataset"]["decision"] == base["decision"]
    model["decision"] = base["decision"]
    avh.atomic_json(tmp_path / "terms_review.json", {"dataset": base, "gemma": model})
    with pytest.raises(RuntimeError, match="gemma terms"):
        avh.check_terms(tmp_path, "gemma")


@pytest.mark.parametrize("family,timeout", [("gemma", 3600), ("qwen", 1800)])
def test_early_smoke_needs_no_manual_gates_and_never_writes_formal_gate(tmp_path, monkeypatch, family, timeout):
    qa, _ = release_fixture()
    instances = []
    class FakeScorer:
        env, tokens = {"kind": "synthetic"}, {"yes": [1], "no": [0]}
        def __init__(self, cfg):
            self.generations, self.questions = 0, []
            instances.append(self)
        def score(self, row, stage, index, cached=True):
            self.questions.append(row)
            return {"answer": 1, "peak_vram_bytes": 1}, np.ones((2, 3)), np.array([0., 1.])
        def generation_check(self, row, index):
            self.generations += 1
            return {"generated_answer": 1}
    watches = []
    class FakeWatchdog:
        def __init__(self, seconds=600, deadline=None):
            self.seconds, self.deadline, self.touches = seconds, deadline, 0
            watches.append(self)
        def __enter__(self):
            return self
        def touch(self):
            self.touches += 1
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(avh, "OmniScorer", FakeScorer)
    monkeypatch.setattr(avh, "Watchdog", FakeWatchdog)
    monkeypatch.setattr(avh, "prepare_media", lambda *a, **kw: {})
    monkeypatch.setattr(avh, "verify_media", lambda *a: None)
    cfg = avh.Config(str(tmp_path), family, "a"*40)
    scorer, report = avh.pre_smoke(cfg, qa, tmp_path)
    assert report["n_qa"] == 20 and report["n_videos"] >= 3
    assert scorer.generations == 10 and len(scorer.questions) == 40
    assert watches[0].seconds == timeout and watches[0].touches == 1
    assert (watches[0].deadline is not None) == (family == "qwen")
    assert report["eligible_for_bulk_gate"] is False
    assert (tmp_path / family / "pre_smoke.json").exists()
    assert not (tmp_path / family / "smoke_f8.json").exists()


def test_bulk_inference_scores_all_rows_without_reading_audit(tmp_path, monkeypatch):
    qa, lineage = release_fixture()
    manifest = avh.make_manifest(tmp_path, qa, lineage)
    cfg = avh.Config(str(tmp_path), "gemma", "a"*40, stage="no_media")
    env, index = {"kind": "synthetic"}, {}
    avh.atomic_json(tmp_path / "gemma" / "smoke_f8.json", {"passed": True, "manifest_sha256": avh.digest(manifest),
        "media_sha256": avh.digest(index), "code_sha256": avh.file_hash(SOURCE), "environment": env,
        "configuration": avh.dataclasses.asdict(cfg)})
    monkeypatch.setattr(avh, "check_terms", lambda *args: {})
    monkeypatch.setattr(avh, "verify_media", lambda *args: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must not read the audit")
    monkeypatch.setattr(avh, "check_audit", forbidden)
    monkeypatch.setattr(avh, "study_signature", lambda *args: "synthetic-signature")
    class Scorer:
        def score(self, row, stage, index):
            return {"margin": 1., "answer": 1, "peak_vram_bytes": 1}, None, None
    scorer = Scorer()
    scorer.env = env
    result = avh.run_stage(scorer, cfg, manifest, index, "GO")
    expected = sum(r["split"] in ("calibration", "validation") for r in manifest["rows"])
    assert result["new_rows"] == result["eligible"] == expected


def test_authored_notebooks_are_valid_unexecuted_and_compile():
    import nbformat
    for model in ("gemma", "qwen"):
        path = SOURCE.parent / f"avh_{model}.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                assert cell.execution_count is None and not cell.outputs
                compile(cell.source, f"{path.name}:cell{index}", "exec")
        first_code = next(c for c in notebook.cells if c.cell_type == "code")
        with pytest.raises(RuntimeError, match="HALT"):
            exec(first_code.source, {})
