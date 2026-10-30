#!/usr/bin/env bash
set -euo pipefail
# One-command Colab/GPU runner. Usage: bash scripts/colab_run.sh [setup|timing|run|report|all] [extra run_pope args]
#   COUNTERPRIOR_RUN      run directory (put it on Google Drive so a disconnect can resume)
#   COUNTERPRIOR_PROFILE  quick (160 images) | full (500 images); default quick
#   COUNTERPRIOR_COCO     COCO val2014 image folder; default data/coco/val2014
#   COUNTERPRIOR_CUSTOM=1 also run the secondary 16-row CounterPrior set (pauses for manual review)
# Logs survive failures; re-running the same command resumes from the run directory.
mode="${1:-all}"; shift || true
run="${COUNTERPRIOR_RUN:-results/2026-10-31/pope-$(date -u +%Y%m%dT%H%M%SZ)}"
profile="${COUNTERPRIOR_PROFILE:-quick}"
coco="${COUNTERPRIOR_COCO:-data/coco/val2014}"
py=.venv-colab/bin/python
mkdir -p "$run"

setup() {
  # Colab's Python 3.13 strips ensurepip, so 'python -m venv' fails.
  # Install uv directly into the system pip environment instead.
  if ! command -v uv &>/dev/null && [ ! -f .venv-bootstrap-done ]; then
    pip install --quiet "uv==0.9.7" 2>&1 | tee -a "$run/bootstrap.log"
    touch .venv-bootstrap-done
  fi
  if [ ! -x "$py" ]; then
    uv venv --python 3.11.9 .venv-colab 2>&1 | tee -a "$run/python-setup.log"
  fi
  uv pip sync --python "$py" --require-hashes --only-binary :all: requirements-colab.lock 2>&1 | tee -a "$run/install.log"
  uv pip check --python "$py" 2>&1 | tee -a "$run/pip-check.log"
  uv pip freeze --python "$py" > "$run/environment-freeze.txt"
  "$py" -m unittest discover -s tests -v 2>&1 | tee -a "$run/tests.log"
  "$py" -m scripts.fetch_pope --coco-dir "$coco" 2>&1 | tee -a "$run/fetch.log"
}
timing() { "$py" -m scripts.run_pope --output "$run" --profile "$profile" --coco-dir "$coco" --timing-probe 20 "$@" 2>&1 | tee -a "$run/inference.log"; }
pope() { "$py" -m scripts.run_pope --output "$run" --profile "$profile" --coco-dir "$coco" "$@" 2>&1 | tee -a "$run/inference.log"; }
report() { "$py" -m scripts.make_report --run "$run" --out results/REPORT.md --figures results/figures --readme README.md 2>&1 | tee -a "$run/report.log"; }

case "$mode" in
  setup) setup ;;
  timing) timing "$@" ;;
  run) pope "$@" ;;
  report) report ;;
  all) setup; timing "$@"; pope "$@"; report ;;
  *) echo "unknown mode $mode (setup|timing|run|report|all)"; exit 2 ;;
esac
if [ "${COUNTERPRIOR_CUSTOM:-0}" = 1 ]; then
  "$py" -m scripts.run_experiment --config configs/llava.json --output "$run/custom" 2>&1 | tee -a "$run/custom.log"
fi
echo "Artifacts: $run (results/REPORT.md and README.md are updated by the report step)"
