# Course project report (LaTeX)

`main.tex` — the CSE 754 project report. `references.bib` — bibliography.

## Compiling

**Overleaf:** upload `main.tex`, `references.bib`, and the `figures/` directory,
then adjust `\graphicspath` if you flatten the folder structure. Press Recompile.

**Locally with tectonic** (no TeX installation needed):

```bash
brew install tectonic          # once
cd docs/report
tectonic -X compile main.tex
```

**Locally with a full TeX distribution:**

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Structure

Figures are pulled from `../figures/` via `\graphicspath`, so regenerating them
updates the report without touching the `.tex`:

```bash
python scripts/make_figures.py         # data / method figures
python scripts/make_result_figures.py  # results figures (needs outputs/)
```

Editable macros are defined at the top of `main.tex` (`\MT`, `\PA`, `\FL`).
Every number in the report traces to `outputs/evaluation.json`,
`outputs/ablations.json` or `outputs/training_summary.json`.
