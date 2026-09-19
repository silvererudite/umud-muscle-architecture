# Project presentation (5 minutes)

Beamer deck for the CSE 754 project, 11 slides at 16:9.

## Timing guide

| | Slide | Cumulative |
|---:|---|---:|
| 1 | Title | 0:00 |
| 2 | The problem | 0:15 |
| 3 | What the task actually is | 0:45 |
| 4 | Pipeline | 1:20 |
| 5 | Scale calibration | 1:55 |
| 6 | Geometric reconstruction | 2:20 |
| 7 | Evaluating without measurement labels | 2:40 |
| 8 | Results | 3:10 |
| 9 | Ablations | 3:40 |
| 10 | The main finding | 4:10 |
| 11 | Closing | 4:40 |

Roughly 30 seconds per slide. Slides 9 and 10 carry the argument, so if you run
short, compress 5 and 6 rather than those.

## Building

```bash
cd docs/slides
tectonic -X compile main.tex        # -> main.pdf
```

Or an Overleaf bundle from the repository root:

```bash
./scripts/make_overleaf_zip.sh slides   # -> umud-slides-overleaf.zip
```

## Notes

Colours and the accent rule match `docs/report` and `docs/review`, so the three
documents read as one set. Figures resolve through `\graphicspath`, which looks
in `figures/` then `../figures/`, so the deck compiles both inside the repository
and inside the Overleaf bundle.

The pipeline diagram is wrapped in `\resizebox` to fit the slide width; if you
add or remove a stage it will rescale automatically.
