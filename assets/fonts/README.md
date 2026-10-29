# Document font

`project-cjk-serif.ttf` is a renamed, limited-glyph derivative built by merging the Latin and numbered 25, 100 and 106–119 regular-weight subsets of Noto Serif SC v36 distributed by [Fontsource](https://github.com/fontsource/font-files/tree/2ffd3aa25908df6ed1c864de40ce53e2ae1076d1/fonts/google/noto-serif-sc). Font metadata identifies the original font as version 2.003-H1 and credits Adobe (2017–2024); the distributed subset package credits Google Inc. The derivative contains the glyphs needed by the current report and diagrams, not a complete Chinese font.

The font remains under the [SIL Open Font License 1.1](./OFL.txt). Its SHA-256 is `56bd5d7eb730ed49aa33ff19e2bbbb8dbb78ddd656797013ad88f63814f6016d`. For new text requiring additional characters, set `CJK_FONT` to a complete Noto Serif CJK SC font file before rebuilding; inspect the build log for missing characters.
