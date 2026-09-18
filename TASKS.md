# CSE 754 Project — Task List

**Competition:** [UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)
**Deadline (course):** 2 days · **Deadline (competition):** 2026-11-14
**Compute:** all training/inference runs on Kaggle GPU kernels via the Kaggle API. No local training.

---

## Phase 0 — Recon  ✅

- [x] Read proposal (`comp_vis_project_idea_presentation.pdf`)
- [x] Authenticate Kaggle API
- [x] Map competition data layout
- [x] Determine submission schema
- [x] Determine metric direction + leaderboard range
- [x] Study strongest public notebooks

### What we learned

| Item | Value |
|---|---|
| Submission | `image_id;pa_deg;fl_mm;mt_mm`, 309 rows, `;`-separated |
| Metric | `UMUD_score`, **lower is better**; public LB 33 % of rows |
| Public LB top | 0.28263 · median entry ≈ 0.40 · DL_Track_US reference ≈ 0.78 |
| Limits | 5 submissions/day, 2 final scored submissions, teams ≤ 10 |
| Train — aponeurosis | 1 048 image/mask pairs |
| Train — fascicle | 2 761 image/mask pairs |
| Test | 309 images, mixed devices/formats (`.tif`, `.png`) |
| **No PA/FL/MT labels are given for training** | measurements must be *derived* from geometry |
| Scale | mm/px must be recovered from on-screen ruler tick marks, per device family |

---

## Phase 1 — Foundations

- [x] 1.1 Repo scaffold, `.gitignore`, licence, requirements
- [x] 1.2 **Scale calibration** — recover `px_per_cm` + image ROI from ruler ticks for every device family (test + train)
- [x] 1.3 EDA: device families, image sizes, mask statistics, scale distribution → figures for the report
- [x] 1.4 Data module: paired image/mask loading, train/val split grouped by device, augmentations
- [x] 1.5 Upload `src/umud` to Kaggle as a versioned dataset so kernels import one source of truth

## Phase 2 — Models (Kaggle GPU)

- [x] 2.1 U-Net (pretrained encoder) for **aponeurosis** segmentation
- [x] 2.2 U-Net for **fascicle** segmentation
- [x] 2.3 **Orientation head** — the proposal's core contribution: regress a dense fascicle-direction field (cos 2θ, sin 2θ) jointly with the mask, instead of recovering angle by post-hoc Hough
- [x] 2.4 Train on Kaggle GPU kernel; pull weights + curves back — apo Dice **0.836**, fascicle orientation error **2.74°**, 93.6 min on a T4

## Phase 3 — Geometry

- [x] 3.1 Aponeurosis masks → superficial/deep boundary lines → **muscle thickness**
- [x] 3.2 Fascicle mask + orientation field → dominant fascicle direction → **pennation angle**
- [x] 3.3 Geometric reconstruction → **fascicle length** (aponeurosis-intersection, not the naive `MT/sin θ`)
- [x] 3.4 Physiological clamping + failure fallbacks
- [x] 3.5 **Sequence module** — test frames come in near-duplicate groups; detect groups and smooth predictions across them (the proposal's "temporal coherence")

## Phase 4 — Evaluation

- [~] 4.1 Segmentation metrics: Dice / IoU on held-out split
- [x] 4.2 **Pseudo-ground-truth protocol**: run the geometry module on *ground-truth* masks vs *predicted* masks on held-out data → MAE in mm/degrees without needing expert labels
- [~] 4.3 Error breakdown by device family (the proposal's generalisation study)
- [~] 4.4 Ablations: − orientation head, − sequence smoothing, − calibration refinement
- [ ] 4.5 Compare error against published inter-expert variability

## Phase 5 — Submission & delivery

- [~] 5.1 Inference kernel → `submission.csv` → submit via API, record LB score  ← **running**
- [ ] 5.2 Iterate within the 5/day budget
- [ ] 5.3 `README.md` — full reproduction instructions
- [ ] 5.4 `docs/REPORT.md` — course report: method, results, ablations, honest limitations
- [ ] 5.5 Figures for the report
- [ ] 5.6 Push to GitHub

---

## Findings so far (things the data taught us, not assumed)

| finding | evidence |
|---|---|
| Image is a resize of the mask's frame, not a crop | intensity under resized mask 3.77x frame mean vs 1.17x for a rolled control |
| A naive split leaks 179 duplicate groups into fascicle validation | content index over all 3,809 frames |
| Test set: 100 % ruler coverage | implied field of view 2.97-6.95 cm deep, matching documented presets |
| Training set: only 60 % (apo) / 12 % (fasc) have a readable ruler | so offline evaluation must be scale-free |
| Generic ruler fallback was inventing scales for ruler-less crops | 49 px/cm vs true 77.8 -> FL pinned to its clamp on 59/78 frames; **fixed** |
| Orientation targets are sound | agree with independent per-segment PCA to 1.1 deg median, 100 % within 5 deg |
| Geometry is sound | PA from ground-truth masks: median 17.0 deg (literature 10-25 deg), needs no scale |
| Intersection vs MT/sin(PA) is a real effect | 42 % of frames have aponeuroses >2 deg from parallel; median 10.6 % difference in FL |
| Test set hides 27 five-frame acquisition runs | similarity bimodal: >0.99 within runs, <0.65 everywhere else |

## Stretch (after course submission)

- [ ] Multi-expert uncertainty calibration on the UMUD 35-image six-rater set
- [ ] True video/temporal module (optical flow) on UMUD video data
- [ ] Cross-muscle generalisation study
