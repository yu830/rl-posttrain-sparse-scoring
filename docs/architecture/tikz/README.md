# Teacher scoring architecture · LaTeX/TikZ

The two figures used in the public English/Chinese README are **editable technical illustrations**. The `.tex` files are canonical sources; PDF, SVG and PNG are rendered derivatives. The figures correspond to the pinned single-GPU Qwen2/FSDP2 scoring path.

| Figure | English | 中文 |
| --- | --- | --- |
| Worker execution & Dense/Chunked readout | [TeX](./architecture.en.tex) · [SVG](./architecture.en.svg) · [PDF](./architecture.en.pdf) | [TeX](./architecture.zh-CN.tex) · [SVG](./architecture.zh-CN.svg) · [PDF](./architecture.zh-CN.pdf) |
| LM Head tensor lifetime | [TeX](./readout-lifetime.en.tex) · [SVG](./readout-lifetime.en.svg) · [PDF](./readout-lifetime.en.pdf) | [TeX](./readout-lifetime.zh-CN.tex) · [SVG](./readout-lifetime.zh-CN.svg) · [PDF](./readout-lifetime.zh-CN.pdf) |

## Build

Requires XeLaTeX, TikZ/PGF, `fontspec`, `xeCJK`, Liberation Serif and Noto Serif CJK SC. **No font files are included.** From this directory:

```bash
for stem in architecture.en architecture.zh-CN readout-lifetime.en readout-lifetime.zh-CN; do
  xelatex -interaction=nonstopmode -halt-on-error "$stem.tex"
  pdftocairo -svg "$stem.pdf" "$stem.svg"
  pdftocairo -png -singlefile -r 150 "$stem.pdf" "$stem"
done
```

The SVG export may outline glyphs. Edit the TikZ source for text, formulas, node positions and connectors. The diagrams are monochrome vectors, not rendered stock artwork.

## Reading the figures

- **A0 Native Dense:** the full `[N,V]` logits are generated inside FSDP and reduced after its forward returns.
- **A1 Memory-Bounded:** temporary `[B,V]` logits are reduced per tile; only compact `[N,K]` outputs are retained.
- Inputs and policy are separate from the model's data path. Loss/gradient/Student update is a **separate restricted validation**, not a claim that a complete Ray Trainer run has been implemented.
- The memory budget models readout allocations. It is **not** a hard bound on the whole GPU process.

Historical figures in the parent folder are retained for research provenance.
