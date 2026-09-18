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

- [ ] 1.1 Repo scaffold, `.gitignore`, licence, requirements
- [ ] 1.2 **Scale calibration** — recover `px_per_cm` + image ROI from ruler ticks for every device family (test + train)
- [ ] 1.3 EDA: device families, image sizes, mask statistics, scale distribution → figures for the report
- [ ] 1.4 Data module: paired image/mask loading, train/val split grouped by device, augmentations
- [ ] 1.5 Upload `src/umud` to Kaggle as a versioned dataset so kernels import one source of truth

## Phase 2 — Models (Kaggle GPU)

- [ ] 2.1 U-Net (pretrained encoder) for **aponeurosis** segmentation
- [ ] 2.2 U-Net for **fascicle** segmentation
- [ ] 2.3 **Orientation head** — the proposal's core contribution: regress a dense fascicle-direction field (cos 2θ, sin 2θ) jointly with the mask, instead of recovering angle by post-hoc Hough
- [ ] 2.4 Train on Kaggle GPU kernel; pull weights + curves back

## Phase 3 — Geometry

- [ ] 3.1 Aponeurosis masks → superficial/deep boundary lines → **muscle thickness**
- [ ] 3.2 Fascicle mask + orientation field → dominant fascicle direction → **pennation angle**
- [ ] 3.3 Geometric reconstruction → **fascicle length** (aponeurosis-intersection, not the naive `MT/sin θ`)
- [ ] 3.4 Physiological clamping + failure fallbacks
- [ ] 3.5 **Sequence module** — test frames come in near-duplicate groups; detect groups and smooth predictions across them (the proposal's "temporal coherence")

## Phase 4 — Evaluation

- [ ] 4.1 Segmentation metrics: Dice / IoU on held-out split
- [ ] 4.2 **Pseudo-ground-truth protocol**: run the geometry module on *ground-truth* masks vs *predicted* masks on held-out data → MAE in mm/degrees without needing expert labels
- [ ] 4.3 Error breakdown by device family (the proposal's generalisation study)
- [ ] 4.4 Ablations: − orientation head, − sequence smoothing, − calibration refinement
- [ ] 4.5 Compare error against published inter-expert variability

## Phase 5 — Submission & delivery

- [ ] 5.1 Inference kernel → `submission.csv` → submit via API, record LB score
- [ ] 5.2 Iterate within the 5/day budget
- [ ] 5.3 `README.md` — full reproduction instructions
- [ ] 5.4 `docs/REPORT.md` — course report: method, results, ablations, honest limitations
- [ ] 5.5 Figures for the report
- [ ] 5.6 Push to GitHub

---

## Stretch (after course submission)

- [ ] Multi-expert uncertainty calibration on the UMUD 35-image six-rater set
- [ ] True video/temporal module (optical flow) on UMUD video data
- [ ] Cross-muscle generalisation study
