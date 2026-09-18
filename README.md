# Measuring Muscle, Automatically

**Geometry-aware deep learning for muscle architecture estimation in B-mode ultrasound**

Shamima Hossain · ID 21141119 · CSE 754 Course Project
Kaggle: [UMUD Challenge — Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)

---

## The problem

Three numbers describe skeletal-muscle architecture — **muscle thickness** (mm),
**pennation angle** (°) and **fascicle length** (mm). They are still measured by
hand from B-mode ultrasound: slow, subjective, and dependent on operator
experience. The task is to produce all three automatically for 309 held-out
ultrasound frames.

What makes it harder than it looks:

| | |
|---|---|
| **No measurement labels** | Training data gives *masks*, never a thickness or an angle. The three targets have to be *derived* geometrically. |
| **No scale metadata** | The targets are millimetres; the network sees pixels. The mm/px scale has to be read back off the depth ruler the scanner burns into each image. |
| **Mixed devices** | 14 source datasets, six scanner families, frame sizes from 465×512 to 1640×1080, rulers on the left, right or bottom edge. |
| **Non-square pixels** | An angle measured on the pixel grid is not the anatomical angle. |
| **Dirty pairing** | ~70 % of training frames disagree in size with their own mask; 26 % of fascicle images are duplicates of another image in the same set. |

## Approach

```
frame
  ├─ ruler calibration ──────────────► mm/px, region of interest
  ├─ aponeurosis U-Net ──────────────► superficial + deep bands
  ├─ fascicle U-Net + orientation head ► mask + dense (cos 2θ, sin 2θ) field
  ├─ geometric reconstruction ───────► θ, thickness, L_f  (+ confidence)
  └─ sequence smoothing ─────────────► consensus across acquisition runs
```

Three things distinguish it from the standard segment-then-post-process recipe:

1. **Orientation is a first-class network output.** Instead of recovering
   fascicle direction afterwards with a structure tensor or Hough transform, the
   model predicts a dense orientation field as a doubled-angle unit vector
   `(cos 2θ, sin 2θ)`. Doubling removes the 180° ambiguity that makes raw angle
   regression discontinuous along every fascicle; predicting a *field* lets the
   downstream aggregation weight by segmentation confidence and report its own
   dispersion as a per-frame uncertainty.

2. **Fascicle length comes from an intersection, not from `MT / sin θ`.** The
   textbook formula assumes the two aponeuroses are parallel. They are not, and
   the error that assumption introduces grows like `1/sin` as pennation gets
   shallow. We fit each aponeurosis as a curve and intersect the fascicle
   direction with them directly.

3. **Everything is solved in millimetre space.** Pixel aspect ratio is corrected
   before any angle is measured, and pennation is referenced to the deep
   aponeurosis rather than to the image horizontal — which is what the clinical
   definition actually says.

## Repository layout

```
src/umud/
  calibration.py   ruler recognition per scanner family -> mm/px + ROI
  data.py          image/mask pairing, duplicate-aware splits, orientation targets
  models.py        U-Net + orientation head
  losses.py        BCE + Dice, masked cosine loss for the orientation field
  train.py         training loop
  geometry.py      masks + orientation -> PA, FL, MT, confidence
  sequence.py      acquisition-run recovery and consensus smoothing
  predict.py       end-to-end inference and submission writing
  evaluate.py      pseudo-ground-truth evaluation protocol and ablations
kaggle/
  src/             the package, shipped to Kaggle as a versioned dataset
  train/           GPU training kernel
  infer/           inference + evaluation + ablation kernel
scripts/           one-off local utilities
docs/              report, figures, original proposal
```

## Reproducing

All training and inference run on **Kaggle GPU kernels** — nothing heavy runs
locally. You need a Kaggle API token with access to the competition.

```bash
export KAGGLE_API_TOKEN=...              # or ~/.kaggle/kaggle.json

# everything below, end to end:
./scripts/run_pipeline.sh "run description"
```

Or step by step:

```bash
# 1. ship the source package to Kaggle as a dataset.  kaggle/src/umud is a
#    staging copy of src/umud and is not tracked in git, so populate it first.
mkdir -p kaggle/src/umud/assets
cp src/umud/*.py          kaggle/src/umud/
cp src/umud/assets/*.json kaggle/src/umud/assets/
kaggle datasets version -p kaggle/src -m "update" -r zip

# 2. train (two U-Nets on a T4; budget ~2 h)
kaggle kernels push -p kaggle/train
kaggle kernels status shamimahossain/umud-train-seg

# 3. infer, evaluate, ablate
kaggle kernels push -p kaggle/infer
kaggle kernels output shamimahossain/umud-infer -p outputs/

# 4. check before spending one of the five daily submissions
python scripts/check_submission.py outputs/submission.csv

# 5. submit
kaggle competitions submit \
  -c umud-challenge-muscle-architecture-in-ultrasound-data \
  -f outputs/submission.csv -m "geometry-aware pipeline"
```

To work on the code locally (no training):

```bash
uv venv --python 3.11 .venv && uv pip install -r requirements.txt
source .venv/bin/activate

pytest tests/                      # 20 tests, no data needed
python scripts/build_index.py      # rebuild the duplicate/paired index
python scripts/make_figures.py     # EDA figures for the report
python scripts/build_notebook.py   # regenerate notebooks/01_data_and_method.ipynb
```

The local checkout needs the competition data under `data/umud/` (or set
`UMUD_DATA_DIR`) for anything beyond the test suite:

```bash
kaggle competitions download -c umud-challenge-muscle-architecture-in-ultrasound-data -p data/raw
unzip -q data/raw/*.zip -d data/umud
```

## Results

See [`docs/REPORT.md`](docs/REPORT.md).

## Acknowledgements

The per-scanner ruler geometry in `calibration.py` follows the analysis
published by **AmbrosM** in the public notebook
[*UMUD Quick and Dirty*](https://www.kaggle.com/code/ambrosm/umud-quick-and-dirty)
(CC BY-SA). Which pixel column a given scanner draws its ticks in is an
empirical fact about the data rather than something derivable, and it is cited
rather than re-derived. The generic ruler fallback, pixel-aspect handling,
confidence reporting, and everything else in this repository are our own.

The task and data come from the UMUD benchmark (Ritsche et al., *BMC Medical
Imaging* 2026) and the DL_Track_US training sets (Ritsche et al., *Ultrasound in
Medicine & Biology* 2024).

## Licence

Code: MIT (see `LICENSE`). The competition data is **not** covered by it and
remains governed by the competition rules and its CC BY-NC-SA terms.
