#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CJK_FONT="${CJK_FONT:-$PWD/assets/fonts/project-cjk-serif.ttf}"
if [[ ! -f "$CJK_FONT" ]]; then
  printf 'Font file does not exist: %s\n' "$CJK_FONT" >&2
  exit 1
fi
mkdir -p build/en build/zh-CN assets
ln -sfn "$CJK_FONT" build/cjk.ttf
python3 scripts/render_architecture.py --font "$CJK_FONT"
for lang in en zh-CN; do
  name="technical-report.$lang"
  out="build/$lang"
  xelatex -halt-on-error -interaction=nonstopmode -file-line-error -output-directory="$out" "docs/reports/$name.tex" > "$out/first-pass.log"
  BIBINPUTS="$PWD:" bibtex "$out/$name" > "$out/bibtex.log"
  xelatex -halt-on-error -interaction=nonstopmode -file-line-error -output-directory="$out" "docs/reports/$name.tex" > "$out/second-pass.log"
  xelatex -halt-on-error -interaction=nonstopmode -file-line-error -output-directory="$out" "docs/reports/$name.tex" > "$out/final-pass.log"
  cp "$out/$name.pdf" "docs/reports/$name.pdf"
  pdftoppm -f 1 -l 1 -r 120 -png -singlefile "docs/reports/$name.pdf" "assets/report-cover.$lang" >/dev/null 2>&1
  printf 'Built %s\n' "$name"
done
