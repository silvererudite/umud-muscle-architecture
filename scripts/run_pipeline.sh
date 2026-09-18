#!/usr/bin/env bash
# End-to-end pipeline on Kaggle.  Requires KAGGLE_API_TOKEN in the environment.
#
#   ./scripts/run_pipeline.sh                 # everything
#   ./scripts/run_pipeline.sh --from infer    # skip training, reuse its output
#   ./scripts/run_pipeline.sh --no-submit     # stop before spending a submission
#
# Stages: src -> train -> infer -> figures -> submit
set -euo pipefail

COMP=umud-challenge-muscle-architecture-in-ultrasound-data
HERE="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE=${KAGGLE:-"$HERE/.venv/bin/kaggle"}
PYTHON=${PYTHON:-"$HERE/.venv/bin/python"}

START=src
SUBMIT=1
MESSAGE="geometry-aware pipeline"
while [ $# -gt 0 ]; do
  case "$1" in
    --from) START="$2"; shift 2;;
    --no-submit) SUBMIT=0; shift;;
    -m) MESSAGE="$2"; shift 2;;
    *) MESSAGE="$1"; shift;;
  esac
done

stage_index() {
  case "$1" in src) echo 0;; train) echo 1;; infer) echo 2;; figures) echo 3;; submit) echo 4;; *) echo 99;; esac
}
should_run() { [ "$(stage_index "$1")" -ge "$(stage_index "$START")" ]; }

wait_for() {  # wait_for <kernel-ref>
  local ref="$1" status
  while true; do
    status=$("$KAGGLE" kernels status "$ref" 2>&1 | tail -1)
    case "$status" in
      *COMPLETE*) echo "  $ref complete"; return 0;;
      *ERROR*|*CANCEL*) echo "  $ref FAILED: $status" >&2; return 1;;
    esac
    sleep 60
  done
}

if should_run src; then
  echo "==> staging src/umud for the Kaggle dataset"
  mkdir -p "$HERE/kaggle/src/umud/assets"
  cp "$HERE"/src/umud/*.py "$HERE/kaggle/src/umud/"
  cp "$HERE"/src/umud/assets/*.json "$HERE/kaggle/src/umud/assets/"
  "$KAGGLE" datasets version -p "$HERE/kaggle/src" -m "$MESSAGE" -r zip
  sleep 20   # let the new version finish processing before a kernel mounts it
fi

if should_run train; then
  echo "==> training"
  "$KAGGLE" kernels push -p "$HERE/kaggle/train"
  wait_for shamimahossain/umud-train-seg
fi

if should_run infer; then
  echo "==> inference, evaluation, ablations"
  "$KAGGLE" kernels push -p "$HERE/kaggle/infer"
  wait_for shamimahossain/umud-infer
  mkdir -p "$HERE/outputs"
  "$KAGGLE" kernels output shamimahossain/umud-infer -p "$HERE/outputs"
fi

if should_run figures; then
  echo "==> figures"
  "$PYTHON" "$HERE/scripts/make_result_figures.py"
fi

echo "==> pre-flight check"
"$PYTHON" "$HERE/scripts/check_submission.py" "$HERE/outputs/submission.csv"

if should_run submit && [ "$SUBMIT" -eq 1 ]; then
  echo "==> submitting"
  "$KAGGLE" competitions submit -c "$COMP" -f "$HERE/outputs/submission.csv" -m "$MESSAGE"
  sleep 30
  "$KAGGLE" competitions submissions -c "$COMP" | head -5
else
  echo "==> not submitting (use without --no-submit to submit)"
fi
