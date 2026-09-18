"""Precompute the content index the training split and evaluation rely on.

Produces ``src/umud/assets/content_index.json`` containing

* ``groups``   — filename -> content-group id, per task.  26 % of the fascicle
  images are byte-level repeats of another image in the same set (the
  competition forum reports the same thing), so a random split leaks: the model
  would be validated on frames it trained on.  Grouping by content and splitting
  on groups removes that.
* ``paired``   — the frames that carry BOTH an aponeurosis and a fascicle
  annotation.  These are the only frames where the full geometry — thickness,
  pennation angle and fascicle length together — can be reconstructed from
  ground truth, which makes them the end-to-end validation set.

Run locally once; the result is small and ships inside the package so the Kaggle
kernels do not have to re-hash 3 809 images on every run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
Image.MAX_IMAGE_PIXELS = None

from umud.config import COMPETITION_DIR, APO_IMAGES, FASC_IMAGES


def content_signature(path: Path) -> str:
    """Hash of a small normalised thumbnail.

    Thumbnailing before hashing catches images that are the same frame saved at
    a different resolution, which raw file hashing would miss.
    """
    with Image.open(path) as im:
        thumb = np.asarray(im.convert("L").resize((96, 96), Image.BILINEAR))
    return hashlib.md5(thumb.tobytes()).hexdigest()


def main() -> None:
    root = COMPETITION_DIR
    print(f"indexing {root}")

    signatures: dict[str, dict[str, str]] = {}
    for task, subdir in (("apo", APO_IMAGES), ("fasc", FASC_IMAGES)):
        files = sorted((root / subdir).glob("*.tif"))
        signatures[task] = {}
        for i, path in enumerate(files):
            signatures[task][path.name] = content_signature(path)
            if (i + 1) % 500 == 0:
                print(f"  {task}: {i + 1}/{len(files)}")
        distinct = len(set(signatures[task].values()))
        print(f"  {task}: {len(files)} files, {distinct} distinct images "
              f"({len(files) - distinct} duplicates)")

    apo_by_sig: dict[str, list[str]] = {}
    for name, sig in signatures["apo"].items():
        apo_by_sig.setdefault(sig, []).append(name)
    fasc_by_sig: dict[str, list[str]] = {}
    for name, sig in signatures["fasc"].items():
        fasc_by_sig.setdefault(sig, []).append(name)

    paired = [
        {"signature": sig, "apo": sorted(apo_by_sig[sig]), "fasc": sorted(fasc_by_sig[sig])}
        for sig in sorted(set(apo_by_sig) & set(fasc_by_sig))
    ]
    print(f"  paired frames (both annotations): {len(paired)}")

    out_dir = Path(__file__).resolve().parents[1] / "src" / "umud" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "groups": signatures,
        "paired": paired,
        "counts": {
            task: {"files": len(sig), "distinct": len(set(sig.values()))}
            for task, sig in signatures.items()
        },
    }
    out_path = out_dir / "content_index.json"
    out_path.write_text(json.dumps(payload))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
