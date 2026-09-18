#!/usr/bin/env bash
# End-to-end pipeline on Kaggle. Requires KAGGLE_API_TOKEN in the environment.
set -euo pipefail

COMP=umud-challenge-muscle-architecture-in-ultrasound-data
HERE="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE=${KAGGLE:-"$HERE/.venv/bin/kaggle"}
MESSAGE=${1:-"geometry-aware pipeline"}

echo "==> shipping source package"
cp "$HERE"/src/umud/*.py "$HERE"/kaggle/src/umud/
mkdir -p "$HERE"/kaggle/src/umud/assets
cp "$HERE"/src/umud/assets/*.json "$HERE"/kaggle/src/umud/assets/
"$KAGGLE" datasets version -p "$HERE/kaggle/src" -m "$MESSAGE" -r zip

echo "==> training"
"$KAGGLE" kernels push -p "$HERE/kaggle/train"
until "$KAGGLE" kernels status shamimahossain/umud-train-seg | grep -qE 'COMPLETE|ERROR'; do sleep 60; done
"$KAGGLE" kernels status shamimahossain/umud-train-seg

echo "==> inference, evaluation, ablations"
"$KAGGLE" kernels push -p "$HERE/kaggle/infer"
until "$KAGGLE" kernels status shamimahossain/umud-infer | grep -qE 'COMPLETE|ERROR'; do sleep 30; done
mkdir -p "$HERE/outputs"
"$KAGGLE" kernels output shamimahossain/umud-infer -p "$HERE/outputs"

echo "==> figures"
"$HERE/.venv/bin/python" "$HERE/scripts/make_result_figures.py"

echo "==> submitting"
"$KAGGLE" competitions submit -c "$COMP" -f "$HERE/outputs/submission.csv" -m "$MESSAGE"
"$KAGGLE" competitions submissions -c "$COMP" | head -5
