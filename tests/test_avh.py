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


def test_go_gate_and_model_constraints(tmp_path):
    with pytest.raises(RuntimeError, match="HALT"):
        avh.require_go("")
    avh.require_go("GO")
    with pytest.raises(ValueError):
        avh.Config(str(tmp_path), "gemma", "main")
    with pytest.raises(ValueError):
        avh.Config(str(tmp_path), "gemma", "a"*40, frames=4)


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
            "mm_token_type_ids": np.array([[1, 1, 1, 1, 0, 0, 0, 0]])}
    prefix = scorer.split_inputs(full, 4)
    suffix = scorer.split_inputs(full, 4, suffix=True)
    assert prefix["input_ids"].shape[-1] == 4 and "pixel_values" in prefix
    assert suffix["input_ids"].tolist() == [[4, 5, 6, 7]]
    assert suffix["attention_mask"].shape[-1] == 8
    assert "pixel_values" not in suffix


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
