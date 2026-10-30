#!/usr/bin/env bash
set -euo pipefail
# Execute from a clone of upgrade/2026-10-30. Logs survive failures.
run="${COUNTERPRIOR_RUN:-results/2026-10-30/llava-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$run"
python -m venv .venv-bootstrap
.venv-bootstrap/bin/python -m pip install uv==0.9.7 2>&1 | tee -a "$run/bootstrap.log"
if [ ! -x .venv-colab/bin/python ]; then
  .venv-bootstrap/bin/uv venv --python 3.11.9 .venv-colab 2>&1 | tee -a "$run/python-setup.log"
fi
.venv-bootstrap/bin/uv pip sync --python .venv-colab/bin/python --require-hashes --only-binary :all: requirements-colab.lock 2>&1 | tee -a "$run/install.log"
.venv-bootstrap/bin/uv pip check --python .venv-colab/bin/python 2>&1 | tee -a "$run/pip-check.log"
.venv-bootstrap/bin/uv pip freeze --python .venv-colab/bin/python > "$run/environment-freeze.txt"
.venv-colab/bin/python -m unittest discover -s tests -v 2>&1 | tee -a "$run/tests.log"
.venv-colab/bin/python -m scripts.run_experiment --config configs/llava.json --output "$run" "$@" 2>&1 | tee -a "$run/inference.log"
echo "Artifacts: $run (download before ending the Colab runtime)"
