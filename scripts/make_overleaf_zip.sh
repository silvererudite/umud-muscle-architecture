#!/usr/bin/env bash
# Build a self-contained Overleaf bundle for the course report.
#
#   ./scripts/make_overleaf_zip.sh          -> umud-report-overleaf.zip
#
# The bundle carries only the figures the report actually cites, so it stays
# small and nothing silently rots when a figure is renamed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/docs/report"
OUT="$HERE/umud-report-overleaf.zip"
STAGE="$(mktemp -d)/umud-report"
trap 'rm -rf "$(dirname "$STAGE")"' EXIT

mkdir -p "$STAGE/figures"
cp "$SRC/main.tex" "$SRC/references.bib" "$STAGE/"

# Copy exactly the figures cited by \includegraphics, and fail loudly if one
# is missing rather than shipping a bundle that will not compile.
missing=0
while IFS= read -r fig; do
  if [ -f "$HERE/docs/figures/$fig" ]; then
    cp "$HERE/docs/figures/$fig" "$STAGE/figures/"
  else
    echo "MISSING FIGURE: $fig" >&2
    missing=1
  fi
done < <(grep -o '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' "$SRC/main.tex" \
         | sed 's/.*{\(.*\)}/\1/' | sort -u)
[ "$missing" -eq 0 ] || { echo "aborting: figures missing" >&2; exit 1; }

cat > "$STAGE/README.txt" <<'TXT'
Measuring Muscle, Automatically -- CSE 754 course project report
================================================================

HOW TO USE ON OVERLEAF

  1. New Project -> Upload Project -> select this .zip
  2. Press Recompile. Nothing needs editing first.

  Overleaf's defaults are correct for this document (pdfLaTeX + BibTeX).
  If the bibliography shows as [?], press Recompile once more -- BibTeX needs
  a second pass to resolve citations.

CONTENTS

  main.tex        the report
  references.bib  bibliography (13 entries)
  figures/        every figure the report cites

EDITING NOTES

  * Notation macros are defined near the top of main.tex: \MT, \PA, \FL.
    Change them once and every occurrence updates.
  * Group Information is the tabular on the title page.
  * Figures resolve via \graphicspath, which is set to look in figures/ and
    then ../figures/ -- so this bundle and the git repository both work.
  * Section labels follow \label{sec:...}; cross-references use \ref{}.

  Every number in the report traces back to the generated artefacts in the
  repository (outputs/evaluation.json, outputs/ablations.json,
  outputs/training_summary.json).

SOURCE

  https://github.com/silvererudite/umud-muscle-architecture
TXT

rm -f "$OUT"
( cd "$(dirname "$STAGE")" && zip -q -r "$OUT" "$(basename "$STAGE")" )

echo "wrote $OUT"
unzip -l "$OUT" | tail -3
