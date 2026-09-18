"""Pre-flight checks on a submission file.

Submissions are limited to five a day, so a malformed file is expensive. This
refuses to let one through: schema, row count, id set and order against the
competition's own manifest, plus physiological plausibility of the values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umud.config import N_TEST_IMAGES, PRIORS, SUBMISSION_COLUMNS
from umud.data import list_test_images


def read_flexible(path: Path) -> pd.DataFrame:
    """Read a submission written with either separator, BOM or not."""
    text = path.read_text(encoding="utf-8-sig")
    header = text.splitlines()[0]
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")


def check_provenance(path: Path, problems: list[str], notes: list[str]) -> None:
    """Verify the submission came from the run that sits beside it.

    ``kaggle kernels output`` does not always overwrite a file that already
    exists in the destination directory.  A stale submission.csv is perfectly
    valid -- right schema, right ids, sane values -- so every other check here
    passes it, and the only symptom is a leaderboard score identical to the
    previous one.  That cost a submission once; hence this check.

    predictions.csv is the source of truth: it carries the same run's per-frame
    values plus diagnostics, and the submission is a clipped projection of it.
    """
    predictions = path.parent / "predictions.csv"
    if not predictions.exists():
        notes.append("predictions.csv not found; provenance unverified")
        return

    import numpy as np

    submitted = read_flexible(path)
    source = pd.read_csv(predictions)
    merged = submitted.merge(source, on="image_id", suffixes=("_sub", "_src"))
    if len(merged) != len(submitted):
        problems.append("submission ids do not match predictions.csv")
        return

    for column in ("pa_deg", "fl_mm", "mt_mm"):
        low, high = getattr(PRIORS, column)
        expected = merged[f"{column}_src"].clip(low, high)
        drift = (merged[f"{column}_sub"] - expected).abs()
        if drift.max() > 0.01:
            problems.append(
                f"{column}: submission disagrees with predictions.csv on "
                f"{int((drift > 0.01).sum())} rows (max {drift.max():.2f}) — "
                "the submission file is STALE, from an earlier run"
            )
    if not problems:
        notes.append("provenance: matches predictions.csv from the same run")


def check(path: Path) -> int:
    frame = read_flexible(path)
    problems: list[str] = []
    notes: list[str] = []
    check_provenance(path, problems, notes)

    if tuple(frame.columns) != SUBMISSION_COLUMNS:
        problems.append(f"columns are {tuple(frame.columns)}, expected {SUBMISSION_COLUMNS}")
    if len(frame) != N_TEST_IMAGES:
        problems.append(f"{len(frame)} rows, expected {N_TEST_IMAGES}")
    if frame["image_id"].duplicated().any():
        problems.append("duplicate image_id values")

    try:
        expected = [p.name for p in list_test_images()]
        if set(frame["image_id"]) != set(expected):
            missing = sorted(set(expected) - set(frame["image_id"]))[:5]
            extra = sorted(set(frame["image_id"]) - set(expected))[:5]
            problems.append(f"id set mismatch; missing e.g. {missing}, unexpected e.g. {extra}")
        elif list(frame["image_id"]) != expected:
            notes.append("ids are correct but not in file order (harmless)")
    except Exception as exc:
        notes.append(f"could not verify ids against the data directory: {exc}")

    for column in ("pa_deg", "fl_mm", "mt_mm"):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            problems.append(f"{column}: {int(values.isna().sum())} non-numeric or NaN values")
            continue
        if not np.isfinite(values).all():
            problems.append(f"{column}: non-finite values")
        low, high = getattr(PRIORS, column)
        outside = int(((values < low) | (values > high)).sum())
        if outside:
            problems.append(f"{column}: {outside} values outside the physiological range {low}-{high}")
        if values.nunique() == 1:
            problems.append(f"{column}: every row has the same value ({values.iloc[0]}) — the pipeline failed")

    # Distribution sanity.  These bands come from two independent sources that
    # agree with each other: the published physiological ranges for lower-limb
    # muscle, and the summary statistics of a public leaderboard entry scoring
    # 0.45134 (Dread Development, "Vera - Seg-Centerline MT Correction").  We use
    # only the summary statistics as a plausibility band -- landing far outside
    # means something upstream is wrong, most likely the scale.
    reference_median = {"pa_deg": (13.0, 21.0), "fl_mm": (65.0, 105.0), "mt_mm": (16.0, 26.0)}
    for column, (low, high) in reference_median.items():
        if column not in frame:
            continue
        median = float(pd.to_numeric(frame[column], errors="coerce").median())
        if not (low <= median <= high):
            problems.append(
                f"{column}: median {median:.1f} is outside the plausible band "
                f"{low}-{high} — suspect the mm/px scale"
            )
        else:
            notes.append(f"{column}: median {median:.1f} (plausible band {low}-{high})")

    # A geometry cross-check: FL*sin(PA) should be close to MT if the three
    # numbers describe one muscle.  Large disagreement means the three outputs
    # were produced independently and are not mutually consistent.
    if all(c in frame for c in ("pa_deg", "fl_mm", "mt_mm")):
        implied = frame["fl_mm"] * np.sin(np.radians(frame["pa_deg"]))
        relative = np.abs(implied - frame["mt_mm"]) / frame["mt_mm"].clip(lower=1e-6)
        notes.append(f"FL*sin(PA) vs MT: median relative difference {relative.median():.3f}, "
                     f"p90 {relative.quantile(0.9):.3f}")

    print(f"checking {path}")
    print(frame[["pa_deg", "fl_mm", "mt_mm"]].describe().round(2).to_string())
    for note in notes:
        print(f"  note: {note}")
    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nOK — safe to submit")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "submission.csv"
    raise SystemExit(check(target))
