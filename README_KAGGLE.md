# CounterPrior: private six-worker run

The Drive copy remains the backup. Initial Kaggle scope is Gemma only, n=2,000,
K=2, three real collaborators, two T4 processes per participant. Qwen replication
is pending. GPU timing and end-to-end compatibility have not yet been measured.
The approximately two-hour target is a target, not an experimental result.

## Files

- `notebooks/01_export_cpu.ipynb`: CPU Colab export of existing prepared bytes.
  The current version reads Drive files concurrently, builds locally, reuses valid
  interrupted ZIP entries, then copies one archive back to Drive. No media regeneration.
- `notebooks/02_worker_kaggle.ipynb`: owner timing and automatic continuation.
  First cell: N_WORKERS=6, ACCOUNT_INDEX=0. Two fresh processes, pinned one per T4.
- `notebooks/03_merge_analyse_cpu.ipynb`: calibration planning, coverage, audit,
  CPU fitting/validation, FROZEN.json and owner-approved sealed test evaluation.
- `FRIEND_GUIDE.md`: participant instructions. Friends receive an approved copy,
  not the initial owner-only timing version.
- `docs/PREREGISTRATION_KAGGLE.md`: fixed scientific protocol and the single add-on rule.
- `tests/test_avh_kaggle.py`: synthetic CPU contracts. Full project validation is
  recorded in `docs/AVH_KAGGLE_VALIDATION.json`; these are not GPU/model results.

## Owner sequence

1. Finish the CPU export. Download its ZIP and checksum JSON from Drive's study
   `exports` folder. Upload to one PRIVATE Kaggle Dataset named
   `counterprior-v2-data`; share with the two participants. A single archive avoids
   a browser upload of thousands of separate files and requires no public Drive link.
2. Upload the worker notebook privately to Kaggle, attach that dataset/version,
   enable Internet and your own HF_TOKEN secret. Verify model terms on your own
   Hugging Face account. Only then select GPU T4 x2 and launch Save & Run All.
3. Two processes run 10-question smokes and five-question batch/fresh-GPU parity.
   They split 50 timing questions. Numerical failure halts. An excessive forecast
   halts with one proposed owner-reviewed cut, test 1,200 to 900. No automatic cut.
4. When the timing auto-gate passes (<=2.5 forecast hours), the owner processes
   stay loaded and immediately score their two Gemma shards. The notebook prints
   `RUN_APPROVAL_JSON=...` and saves `RUN_APPROVAL.json`. Share that approval with
   the coding agent while this run continues; no credentials are in it.
5. The agent validates the approval against the exact source snapshot and builds
   `02_worker_approved_kaggle.ipynb`. In the local source checkout this is:

   `.venv-cpu/Scripts/python.exe scripts/build_kaggle_notebooks.py --approval-file incoming/RUN_APPROVAL.json`

   This inserts approval as notebook configuration without changing worker code.
   Share the private approved notebook, dataset version, guide and account indices
   1 and 2 with the participants. They change only ACCOUNT_INDEX. Their runtimes
   must match the owner's numerical identity. GPU allocation is not a guarantee.
6. Each account prints one compact `CALIBRATION_ACCOUNT_JSON=...` after its two
   calibration portions are committed. The owner can copy those three records from
   the private live logs into the CPU notebook while validation/test scoring continues.
   They contain calibration answers only. Alternatively use completed output files.
7. CPU planning chooses initial n=2,000 or one add-on up to n=3,000. If triggered,
   issue the prebuilt-code add-on notebook with the same timing approval and account
   indices. It scores only IDs absent from the initial sample, calibration first.
   It never makes a second expansion decision. Test predictions remain sealed.
8. Attach the three saved output versions (plus add-on output versions if needed)
   to the CPU notebook. It validates complete pairs, provenance, finite scores and
   feature integrity. Genuine cumulative resume versions are selected; differing
   overlapping reruns are rejected instead of silently deduplicated.
9. Fill the new audit CSV: label `decision` valid/exclude/uncertain plus reviewer;
   object spot-check `object_check` correct/flagged. No completed audit means no
   analysis freeze. Original audit/lineage/evidence are preserved in the private data.
10. Run CPU fits, calibration OOF MDE planning and validation selection. HALT 2:
    review FROZEN.json and validation numbers. OWNER_OK_HALT2 remains False until
    the owner explicitly approves. Then evaluate sealed test once. HALT 3: review
    complete test CIs/counts, ship verdicts and underpowered/null flags before site work.

## Storage and resume

Workers write `/kaggle/working/out_<account>_<worker>/`. Pair journals flush every
25 questions and on normal stop; full states are fp16 NPY shards of 250. Output
versions are private. A normal run or soft stop saves a version; an abruptly lost
unsaved VM is not a remote checkpoint. Worker failures produce a blocked status
while the notebook parent finishes normally so diagnostics/checkpoints can commit.
To resume, attach that participant's own latest private output version to the
approved notebook. Keep worker count, account index, data version and source fixed.
No password/login automation; tokens never enter job files or logs. All dataset
media remain private; future public content is aggregate results plus our own text.

## Qwen replication

Generate the fixed manifest from the private parent manifest with
`avh_kaggle.lean_manifest(parent, n=1000, model="qwen")` using
`notebooks/avh_kaggle.py`. Save it as `lean_manifest_qwen_n1000.json` and
attach it as a private input alongside the study bundle. The public worker
checks its expected digest before running. Use two accounts with T4 ×2.
Set `ACCOUNT_INDEX` to `0` for workers 0–1 and `1` for workers 2–3.
Use each account's own Hugging Face secret. Complete the blind audit and
freeze choices on validation before opening the held-out predictions.
