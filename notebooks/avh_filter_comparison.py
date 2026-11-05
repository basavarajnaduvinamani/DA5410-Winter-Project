"""CPU reporting for the preregistered secondary comparison."""
from pathlib import Path

import avh_common as avh


LABEL = "no beneficial operating point within the validation recall budget"
FILTERS = ("contrast", "blank_contrast", "probe_risk")


def filter_masks(rows, saved, task):
    import numpy as np

    original = np.array([row["margin"] > 0 for row in rows])
    thresholds = saved["thresholds"][task]
    masks = {
        field: original & (np.asarray([row[field] for row in rows]) >= thresholds[field])
        for field in FILTERS[:2]
    }
    if "probe_risk" in thresholds:
        accepted = original.copy()
        yes = [row for row, is_yes in zip(rows, original) if is_yes]
        if yes:
            risk = avh.probe_scores(saved["bundles"][task], saved["calibration"], yes)["augmented"]
            accepted[original] = np.asarray(risk) <= thresholds["probe_risk"]
        masks["probe_risk"] = accepted
    return original, masks


def validation_operating_points(rows, saved):
    import numpy as np

    result = {}
    for task in avh.TASKS[:2]:
        selected = [row for row in rows if row["task"] == task]
        negative = np.array([row["label"] == 0 for row in selected])
        original, masks = filter_masks(selected, saved, task)
        result[task] = {
            field: ("unavailable" if field not in masks else
                    "beneficial" if (negative & original & ~masks[field]).any() else LABEL)
            for field in FILTERS
        }
    return result


def test_comparison(rows, saved, operating_points, metrics, resamples=2000):
    import numpy as np

    result = {}
    for task in avh.TASKS[:2]:
        selected = [row for row in rows if row["task"] == task]
        _, masks = filter_masks(selected, saved, task)
        negative = np.array([row["label"] == 0 for row in selected], dtype=bool)
        groups = np.array([row["source_group"] for row in selected])
        policies = metrics["tasks"][task]
        recall = {
            "contrast": policies["contrast"]["recall_drop"],
            "blank_contrast": policies["blank_contrast"]["recall_drop"],
            "probe_risk": policies["probe"].get("policy", {}).get("recall_drop"),
        }
        if "probe_risk" in masks and recall["probe_risk"] is None:
            recall["probe_risk"] = avh.evaluate_policy(selected, masks["probe_risk"], resamples)["recall_drop"]
        entry = {"validation_operating_points": operating_points[task], "test_recall_drop": recall}
        if "probe_risk" in masks:
            paired = masks["blank_contrast"][negative].astype(float) - masks["probe_risk"][negative]
            entry["false_yes_reduction_probe_vs_blank"] = avh.cluster_ci(
                paired, groups[negative], lambda values: float(values.mean()), resamples)
        else:
            entry["false_yes_reduction_probe_vs_blank"] = None
        result[task] = entry
    return result


def load_frozen_bundle(root, cfg):
    import joblib

    out = Path(root) / cfg.model
    frozen = avh.read_json(out / "FROZEN.json")
    path = out / "analysis_bundle.joblib"
    if avh.file_hash(path) != frozen["bundle_sha256"]:
        raise RuntimeError("Frozen analysis bundle changed")
    return frozen, joblib.load(path)
