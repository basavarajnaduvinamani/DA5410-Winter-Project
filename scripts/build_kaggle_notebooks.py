"""Build self-contained private Kaggle artifacts; never execute model code."""
from pathlib import Path
import json
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SECONDARY_PROTOCOL_PREFIX = "Secondary filter comparison (before freeze):"


def md(value):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(value).strip().splitlines(keepends=True)}


def code(value):
    source = textwrap.dedent(value).strip() + "\n"
    compile(source, "generated_cell", "exec")
    return {"cell_type": "code", "metadata": {}, "source": source.splitlines(keepends=True),
            "execution_count": None, "outputs": []}


def write(name, cells):
    for i, cell in enumerate(cells):
        cell["id"] = f"{name[:2]}-{i:02d}"
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks" / name
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def embedded(names):
    return {name: (ROOT / ("docs" if name.endswith(".md") else "notebooks") / name).read_text(encoding="utf-8") for name in names}


def export_notebook():
    payload = embedded(["avh_common.py", "avh_kaggle.py", "PREREGISTRATION_KAGGLE.md"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return write("01_export_cpu.ipynb", [md('''
        # 01 — Preserve and export the private study (CPU only)
        Run in Account A Colab with Hardware accelerator = CPU/None. Never regenerate media.
        The prepared Drive folder stays intact. This creates one private upload archive
        and its checksum file under the study's `exports` folder. No GPU or HF token needed.
        Transfer route: download the archive from Drive, then upload it to a private
        Kaggle Dataset. This uses the existing bytes and avoids regenerating clips.
        The export hashes media while streaming it once from Drive. No runtime Drive access later.
    '''), code('''
        from google.colab import drive
        from pathlib import Path
        import sys, json
        drive.mount("/content/drive")
        ROOT = Path("/content/drive/MyDrive/CounterPrior_v2/2026-11-03")
        assert (ROOT / "manifest.json").is_file(), "Mount Account A's populated shared study folder"
        print("CPU export source:", ROOT)
    '''), code(f'''
        # Self-contained CPU export tools. These do not replace files in Drive.
        payload = {payload!r}
        tools = Path("/content/counterprior_export_tools")
        tools.mkdir(exist_ok=True)
        for name, text in payload.items():
            (tools / name).write_text(text, encoding="utf-8")
        sys.path.insert(0, str(tools))
        for name in ("avh_kaggle", "avh_common"):
            sys.modules.pop(name, None)
        import avh_kaggle as kg
        archive = ROOT / "exports" / "counterprior-v2-data-v1.zip"
        if archive.exists():
            checksum = archive.with_suffix(".checksum.json")
            assert checksum.is_file(), "Existing export has no completion checksum; preserve it and report this"
            summary = json.loads(checksum.read_text())
            assert kg.avh.file_hash(archive) == summary["sha256"], "Existing export checksum mismatch"
            print("Reusing completed export; no media regenerated.")
        else:
            summary = kg.export_private_study(ROOT, archive, tools, {commit!r}, code_dirty=True)
        print(json.dumps(summary, indent=2))
        print("EXPORT READY. Keep private. Send this summary before the next manual step.")
    ''')])


def runtime_cell():
    payload = embedded(["avh_common.py", "avh_kaggle.py", "avh_kaggle_worker.py", "PREREGISTRATION_KAGGLE.md",
                        "GEMMA_SERIAL_AMENDMENT.md"])
    # Workers and the exported bundle use the original protocol bytes.
    protocol = payload["PREREGISTRATION_KAGGLE.md"]
    prefix = "\n\n" + SECONDARY_PROTOCOL_PREFIX
    if prefix in protocol:
        start = protocol.index(prefix)
        end = protocol.index("\n", start + len(prefix))
        payload["PREREGISTRATION_KAGGLE.md"] = protocol[:start] + protocol[end:]
    return code(f'''
        import sys, json, time, subprocess
        from pathlib import Path
        SESSION_STARTED = time.monotonic()
        runtime = Path("/kaggle/working/counterprior_runtime")
        runtime.mkdir(parents=True, exist_ok=True)
        payload = {payload!r}
        for name, text in payload.items():
            (runtime / name).write_text(text, encoding="utf-8")
        sys.path.insert(0, str(runtime))
        for name in ("avh_common", "avh_kaggle"):
            sys.modules.pop(name, None)
        import avh_common as avh
        import avh_kaggle as kg
        markers = list(Path("/kaggle/input").rglob("kaggle_bundle.json"))
        assert len(markers) == 1, "Attach exactly one private counterprior-v2-data dataset"
        DATA = markers[0].parent
        bundle = avh.read_json(markers[0])
        assert bundle["private_only"] is True
        assert (DATA / "runtime/PREREGISTRATION_KAGGLE.md").read_text() == payload["PREREGISTRATION_KAGGLE.md"], "Protocol version mismatch"
        print("Private study found:", DATA)
    ''')


def worker_notebook(approval=None, addon=False, filename="02_worker_kaggle.ipynb", provisional=False):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    # First cell is the only participant-edited code. Approval is owner-issued data,
    # never a friend-edited variable. Changing approval does not change GPU worker code.
    cells = [code('''
        N_WORKERS = 6
        ACCOUNT_INDEX = 0  # Owner=0, friend 1=1, friend 2=2. Change nothing else.
    '''), md('''
        # 02 — Private Gemma workers (T4 x2)
        Attach the private dataset and enable Internet. Add your own HF_TOKEN secret.
        Use Save Version -> Save & Run All. This notebook never prints credentials.
        Use the provisional copy only on accounts 1 and 2. Its local timing gate
        starts its shards, but only account 0's approval can authorize analysis.
        No public dataset/notebook/output versions.
        Each GPU receives one fresh process and a complete fp16 model. No TPU/P100.
    '''), runtime_cell(), code(f'''
        OWNER_APPROVAL = {approval!r}
        ADDON = {addon!r}
        PROVISIONAL = {provisional!r}
        assert N_WORKERS == 6, "Worker count is fixed for this run"
        kg.account_workers(ACCOUNT_INDEX, N_WORKERS)
        if PROVISIONAL:
            assert ACCOUNT_INDEX in (1, 2), "Provisional notebook is for B and C only"
            print("PROVISIONAL: only Account A's RUN_APPROVAL_JSON authorizes analysis.")
        elif OWNER_APPROVAL is None and ACCOUNT_INDEX != 0:
            raise RuntimeError("Wait for the owner's approved notebook link; keep GPU off")
        if ADDON and OWNER_APPROVAL is None:
            raise RuntimeError("Add-on needs the owner's original timing approval")
        version = avh.read_json(DATA / "gemma/model_metadata.json")["revision"]
        saved = DATA / "prior_colab/gemma/pre_smoke.json"
        assert saved.is_file(), "Saved Gemma environment missing"
        transformers_version = avh.read_json(saved)["environment"]["transformers"]
        packages = [f"transformers=={{transformers_version}}", "accelerate==1.10.1", "numpy==2.2.5",
            "scikit-learn==1.7.1", "Pillow==11.3.0", "av==16.0.1", "soundfile==0.13.1",
            "librosa==0.11.0", "joblib==1.5.2"]
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *packages], check=True)
        subprocess.run([sys.executable, "-c", "from transformers import Gemma4ForConditionalGeneration; print('Gemma class verified')"], check=True)
        # Parent never imports torch or initializes CUDA. Verify both devices externally.
        devices = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
        assert len(devices) == 2 and all("T4" in d for d in devices), "Select GPU T4 x2"
        for name, expected in bundle["files"].items():
            if not name.startswith("media/"):
                assert avh.file_hash(DATA / name) == expected, f"Metadata changed: {{name}}"
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        job = {{"n_workers": N_WORKERS, "account_index": ACCOUNT_INDEX, "model": "gemma",
            "mode": "addon" if ADDON else "auto" if OWNER_APPROVAL is None else "run",
            "dataset": str(DATA), "manifest": str(DATA / ("lean_manifest_n3000.json" if ADDON else "lean_manifest_n2000.json")),
            "cpu_verified": str(DATA / "cpu_verified.json"), "reference": "fresh_unbatched_gpu",
            "sample_policy": "calibration_addon", "approval": OWNER_APPROVAL,
            "session_started": SESSION_STARTED,
            "code_git_hash": OWNER_APPROVAL["code_git_hash"] if OWNER_APPROVAL else {commit!r},
            "code_dirty": OWNER_APPROVAL["code_dirty"] if OWNER_APPROVAL else True}}
        destination = Path(f"/kaggle/working/account_{{ACCOUNT_INDEX}}")
        # Attach your own previous private notebook output to resume. Never steal another worker's checkpoint.
        resume_dirs = [p.parent for p in Path("/kaggle/input").rglob("identity.json")
                       if (p.parent / "predictions.jsonl").exists()]
        for worker in kg.account_workers(ACCOUNT_INDEX, N_WORKERS):
            matches = []
            for path in resume_dirs:
                first = next(iter((path / "predictions.jsonl").read_text().splitlines()), None)
                if first and json.loads(first)["worker_id"] == worker:
                    matches.append(path)
            selected = kg.select_output_versions(matches)
            if len(selected) > 1:
                raise RuntimeError("Attach the latest cumulative checkpoint for this worker only")
            target = Path(f"/kaggle/working/out_{{ACCOUNT_INDEX}}_{{worker}}/predictions")
            if selected and not target.exists():
                kg.restore_checkpoint(selected[0], target, worker, avh.digest(bundle))
        try:
            kg.launch_account(runtime, job, destination, hf_token=token)
            status = {{"status": "finished", "account_index": ACCOUNT_INDEX, "provisional": PROVISIONAL}}
        except Exception as exc:
            # Finish normally so Kaggle can commit partial checkpoints and diagnostics.
            status = {{"status": "blocked_checkpoint_preserved", "account_index": ACCOUNT_INDEX,
                      "error": str(exc), "action": "Send private logs to owner; no blind rerun"}}
        finally:
            del token
        avh.atomic_json(Path("/kaggle/working/account_status.json"), status)
        print(json.dumps(status, indent=2))
    ''')]
    return write(filename, cells)


def merge_notebook(approval=None):
    analysis_protocol = embedded(["PREREGISTRATION_KAGGLE.md"])["PREREGISTRATION_KAGGLE.md"]
    secondary_source = embedded(["avh_filter_comparison.py"])["avh_filter_comparison.py"]
    return write("03_merge_analyse_cpu.ipynb", [md('''
        # 03 — Owner CPU merge and sealed analysis
        Hardware accelerator = None. Attach the private dataset and each person's
        private saved output version. For an early add-on decision, paste the three
        CALIBRATION_ACCOUNT_JSON receipts from the running notebooks; they contain
        calibration answers only. No test metrics are computed before HALT 2 approval.
        Run through the audit preparation cell as soon as the study dataset is attached.
        Run the calibration cell once all three receipts arrive.
        For the final merge, attach the worker outputs and the completed private
        audit CSV (keep its audit_n2000.csv or audit_n3000.csv filename).
    '''), code(f'''
        # Owner controls. Leave False until the owner explicitly approves HALT 2.
        OWNER_OK_HALT2 = False
        ACCOUNT_A_APPROVAL = {approval or {}!r}
        CALIBRATION_ACCOUNTS = []  # Optional: paste the three account JSON lists here.
    '''), runtime_cell(), code('''
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy==2.2.5",
                        "scikit-learn==1.7.1", "joblib==1.5.2"], check=True)
        initial = avh.read_json(DATA / "lean_manifest_n2000.json")
        expanded = avh.read_json(DATA / "lean_manifest_n3000.json")
        WORK = Path("/kaggle/working")
        for manifest in (initial, expanded):
            folder = WORK / "audit_templates" / f"n{manifest['n']}"
            folder.mkdir(parents=True, exist_ok=True)
            template = avh.create_audit(folder, manifest)
            named = folder / f"audit_n{manifest['n']}.csv"
            named.write_bytes(template.read_bytes())
            print("Blind audit template:", named)
        initial_ids = {r["id"] for r in avh.audit_targets(initial)}
        expanded_ids = {r["id"] for r in avh.audit_targets(expanded)}
        print("Audit questions shared by both samples:", len(initial_ids & expanded_ids))
        gate = ACCOUNT_A_APPROVAL
        if (gate.get("status") != "approved" or gate.get("numerical_smoke_passed") is not True
                or gate.get("n_workers") != 6 or gate.get("n") != initial["n"]
                or gate.get("bundle_sha256") != avh.digest(bundle)):
            raise RuntimeError("Account A's approved smoke and timing gate is required")
        code_names = ("avh_common.py", "avh_kaggle.py", "avh_kaggle_worker.py")
        code_hashes = {name: avh.file_hash(runtime / name) for name in code_names}
        assert avh.digest(code_hashes) == gate["code_sha256"], "Scoring snapshot differs from Account A"
        from zipfile import ZipFile, ZIP_DEFLATED
        snapshot = WORK / "counterprior_scored_code.zip"
        with ZipFile(snapshot, "w", compression=ZIP_DEFLATED) as archive:
            for name in payload:
                archive.write(runtime / name, name)
            archive.writestr("RUN_APPROVAL.json", json.dumps(gate, indent=2))
            archive.writestr("source_sha256.json", json.dumps(code_hashes, indent=2))
        print("Verified scored code snapshot:", snapshot)
    '''), code('''
        # Receipts are sufficient here, even while GPU scoring is still running.
        paths = [p.parent for p in Path("/kaggle/input").rglob("identity.json")
                 if (p.parent / "predictions.jsonl").exists()]
        paths = kg.select_output_versions(paths)
        for path in paths:
            identity = avh.read_json(path / "identity.json")
            if (identity["code_sha256"] != gate["code_sha256"]
                    or avh.digest(identity) != gate["identity_sha256"]):
                raise RuntimeError(f"Worker checkpoint does not match Account A's approval: {path}")
        if CALIBRATION_ACCOUNTS:
            if any(r["identity_sha256"] != gate["identity_sha256"]
                   for account in CALIBRATION_ACCOUNTS for r in account):
                raise RuntimeError("Calibration receipts do not match Account A's approval")
            cal_records = kg.records_from_calibration_receipts(CALIBRATION_ACCOUNTS, initial, 6)
        else:
            # Complete output files are an alternative to live receipts.
            cal_records = kg.calibration_records_from_outputs(paths, initial, 6)
        decision = kg.calibration_power_decision(initial, cal_records, expanded, "calibration_addon")
        kg.immutable_json(Path("/kaggle/working/ADDON_DECISION.json"), decision)
        print(json.dumps(decision, indent=2))
        print("If chosen_n=3000: owner issues ONE add-on notebook; only extra questions are scored.")
    '''), code('''
        final_manifest = expanded if decision["chosen_n"] == 3000 else initial
        assert paths, "Attach all saved worker output versions before final merge"
        kg.verify_bundle(DATA)
        ANALYSIS = Path("/kaggle/working/analysis")
        cfg, index, environment = kg.import_for_analysis(DATA, final_manifest, paths, ANALYSIS, 6)
        print("Coverage complete. Audit sheet:", ANALYSIS / "audit.csv")
        print("Fill decision/reviewer and object_check fields in that sheet before the next cell.")
    '''), code(f'''
        (runtime / "avh_filter_comparison.py").write_text({secondary_source!r}, encoding="utf-8")
        sys.modules.pop("avh_filter_comparison", None)
        import avh_filter_comparison as comparison
        protocol_path = ANALYSIS / "PREREGISTRATION_KAGGLE.md"
        protocol = {analysis_protocol!r}
        if protocol_path.exists():
            assert protocol_path.read_text(encoding="utf-8") == protocol, "Analysis preregistration changed"
        else:
            protocol_path.write_text(protocol, encoding="utf-8")
    '''), code('''
        import numpy as np
        # Load the selected sample's completed audit from a private Kaggle input.
        import shutil
        audit_name = f"audit_n{final_manifest['n']}.csv"
        audits = list(Path("/kaggle/input").rglob(audit_name))
        assert len(audits) <= 1, f"Attach only one completed {audit_name}"
        if audits:
            shutil.copyfile(audits[0], ANALYSIS / "audit.csv")
        candidates = list(Path("/kaggle/input").rglob("yes_no_unembedding.npy"))
        candidates = [p for p in candidates if "prior_colab" not in str(p)]
        assert candidates, "Saved Gemma worker unembedding direction missing"
        direction = np.load(candidates[0], allow_pickle=False)
        for path in candidates[1:]:
            assert np.array_equal(direction, np.load(path, allow_pickle=False)), "Worker unembedding mismatch"
        avh.check_audit(ANALYSIS, final_manifest, require_complete=True)
        calibration = avh.load_completed(ANALYSIS, cfg, final_manifest, index, environment, "calibration")
        mde = kg.calibration_mde(calibration)
        avh.atomic_json(ANALYSIS / "gemma/calibration_mde.json", mde)
        frozen = avh.freeze_analysis(ANALYSIS, cfg, final_manifest, index, environment, direction)
        _, saved = comparison.load_frozen_bundle(ANALYSIS, cfg)
        validation_rows = avh.load_completed(ANALYSIS, cfg, final_manifest, index, environment, "validation")
        comparison_plan = {
            "primary": "held-out delta AUROC, probe-augmented versus full baseline",
            "secondary": "probe filter versus blank-modality filter",
            "method": "paired source-group bootstrap, 2000 resamples",
            "validation_recall_loss_budget": 0.02,
            "preregistration_sha256": avh.file_hash(protocol_path),
            "analysis_code_sha256": avh.file_hash(runtime / "avh_filter_comparison.py"),
            "validation_operating_points": comparison.validation_operating_points(validation_rows, saved),
        }
        if "filter_comparison" in frozen:
            assert frozen["filter_comparison"] == comparison_plan, "Frozen comparison changed"
        else:
            frozen["filter_comparison"] = comparison_plan
            for marker in ("analysis_freeze.json", "FROZEN.json"):
                avh.atomic_json(ANALYSIS / cfg.model / marker, frozen)
        validation = kg.validation_report(ANALYSIS, cfg, final_manifest, index, environment)
        print(json.dumps({"frozen": frozen, "validation": validation, "calibration_mde_planning": mde}, indent=2))
        print("HALT 2: send frozen choices + validation numbers to the owner. Do not open test yet.")
    '''), code('''
        if not OWNER_OK_HALT2:
            print("HALT 2: awaiting explicit owner approval. Test remains sealed.")
        else:
            frozen, saved = comparison.load_frozen_bundle(ANALYSIS, cfg)
            plan = frozen["filter_comparison"]
            assert plan["preregistration_sha256"] == avh.file_hash(protocol_path)
            assert plan["analysis_code_sha256"] == avh.file_hash(runtime / "avh_filter_comparison.py")
            metrics = kg.evaluate_sealed_once(ANALYSIS, cfg, final_manifest, index, environment, owner_ok_halt2=True)
            report = ANALYSIS / cfg.model / "secondary_filter_comparison.json"
            if report.exists():
                secondary = avh.read_json(report)
                assert secondary["plan"] == plan
            else:
                test_rows = avh.load_completed(ANALYSIS, cfg, final_manifest, index, environment, "test")
                secondary = {"plan": plan, "tasks": comparison.test_comparison(
                    test_rows, saved, plan["validation_operating_points"], metrics)}
                kg.immutable_json(report, secondary)
            print(json.dumps(metrics, indent=2))
            print(json.dumps(secondary, indent=2))
            print("HALT 3: review test CIs, false-yes counts, ship verdicts and underpowered/null flags before site work.")
    ''')])


if __name__ == "__main__":
    import argparse
    import hashlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval-file", type=Path)
    args = parser.parse_args()
    if args.approval_file:
        approval = json.loads(args.approval_file.read_text())
        hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in
                  embedded(["avh_common.py", "avh_kaggle.py", "avh_kaggle_worker.py"]).items()}
        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if approval["status"] != "approved" or approval["n_workers"] != 6 or approval["code_sha256"] != fingerprint:
            raise RuntimeError("Approval is not for this exact tested worker code")
        print(worker_notebook(approval, filename="02_worker_approved_kaggle.ipynb"))
        print(worker_notebook(approval, addon=True, filename="02_addon_kaggle.ipynb"))
        print(merge_notebook(approval))
    else:
        print(export_notebook())
        print(worker_notebook())
        print(merge_notebook())
