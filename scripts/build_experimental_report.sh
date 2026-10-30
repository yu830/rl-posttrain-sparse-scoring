#!/usr/bin/env bash
# Build only the experimental supplement; keep the historical reports intact.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
ENGINE="${LATEX_ENGINE:-xelatex}"
REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-$ROOT/docs/reports}"
mkdir -p "$REPORT_OUTPUT_DIR"
for lang in en zh-CN; do
  name="memory-bounded-experimental.$lang"
  build="$ROOT/build/experimental/$lang"
  mkdir -p "$build"
  if [[ "$ENGINE" == xelatex ]]; then
    xelatex -halt-on-error -interaction=nonstopmode -file-line-error -output-directory="$build" "docs/reports/$name.tex" > "$build/first-pass.log"
    BIBINPUTS="$ROOT:" bibtex "$build/$name" > "$build/bibtex.log"
    xelatex -halt-on-error -interaction=nonstopmode -file-line-error -output-directory="$build" "docs/reports/$name.tex" > "$build/second-pass.log"
    xelatex -halt-on-error -interaction=nonstopmode -file-line-error -output-directory="$build" "docs/reports/$name.tex" > "$build/final-pass.log"
    cp "$build/$name.pdf" "$REPORT_OUTPUT_DIR/$name.pdf"
  elif [[ "$ENGINE" == tectonic ]]; then
    # stdin makes the repository root the input root for shared TeX, BibTeX and figures.
    "${TECTONIC_BIN:-tectonic}" --keep-logs -o "$build" - < "docs/reports/$name.tex" > "$build/compile.log" 2>&1
    cp "$build/texput.pdf" "$REPORT_OUTPUT_DIR/$name.pdf"
  else
    printf 'Unsupported LATEX_ENGINE: %s\n' "$ENGINE" >&2
    exit 2
  fi
  printf 'Built %s/%s.pdf\n' "$REPORT_OUTPUT_DIR" "$name"
done
