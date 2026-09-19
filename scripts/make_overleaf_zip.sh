#!/usr/bin/env bash
# Build a self-contained Overleaf bundle for one of the LaTeX documents.
#
#   ./scripts/make_overleaf_zip.sh report   -> umud-report-overleaf.zip
#   ./scripts/make_overleaf_zip.sh review   -> umud-review-overleaf.zip
#   ./scripts/make_overleaf_zip.sh          -> both
#
# The bundle carries only the figures the document actually cites, so it stays
# small and a renamed figure fails loudly here rather than silently on Overleaf.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"

build_one() {
  local name="$1"                       # "report" or "review"
  local src="$HERE/docs/$name"
  local out="$HERE/umud-$name-overleaf.zip"
  local stage; stage="$(mktemp -d)/umud-$name"

  [ -f "$src/main.tex" ] || { echo "no main.tex in $src" >&2; return 1; }

  mkdir -p "$stage/figures"
  cp "$src/main.tex" "$stage/"
  [ -f "$src/references.bib" ] && cp "$src/references.bib" "$stage/"

  local missing=0 count=0
  while IFS= read -r fig; do
    [ -n "$fig" ] || continue
    if   [ -f "$HERE/docs/figures/$fig" ]; then cp "$HERE/docs/figures/$fig" "$stage/figures/"; count=$((count+1))
    elif [ -f "$src/figures/$fig" ];       then cp "$src/figures/$fig"       "$stage/figures/"; count=$((count+1))
    else echo "MISSING FIGURE: $fig" >&2; missing=1
    fi
  done < <(grep -o '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' "$src/main.tex" \
           | sed 's/.*{\(.*\)}/\1/' | sort -u)
  [ "$missing" -eq 0 ] || { echo "aborting: figures missing" >&2; return 1; }
  [ "$count" -gt 0 ] || rmdir "$stage/figures"

  cat > "$stage/README.txt" <<TXT
CSE 754 -- $name

HOW TO USE ON OVERLEAF
  1. New Project -> Upload Project -> select this .zip
  2. Press Recompile. Nothing needs editing first.

  Overleaf's defaults are correct (pdfLaTeX + BibTeX). If the bibliography
  shows as [?], press Recompile once more -- BibTeX needs a second pass.

CONTENTS
  main.tex        the document
  references.bib  bibliography
  figures/        every figure the document cites (if any)

SOURCE
  https://github.com/silvererudite/umud-muscle-architecture
TXT

  rm -f "$out"
  ( cd "$(dirname "$stage")" && zip -q -r "$out" "$(basename "$stage")" )
  rm -rf "$(dirname "$stage")"
  printf '  %-34s %s figures, %s\n' "$(basename "$out")" "$count" "$(du -h "$out" | cut -f1)"
}

targets=("${@:-report review}")
echo "building Overleaf bundles:"
for t in ${targets[@]}; do build_one "$t"; done
