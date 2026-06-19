# Paper — build instructions

`main.tex` is written to the **official Springer Nature journal API**
(`\documentclass[sn-mathphys-num]{sn-jnl}`, `\author*[1]{\fnm..\sur..}`,
`\affil[1]{\orgname..}`, `\abstract{}`, `\keywords{}`, `\bmhead{}`, …).

## Build offline (this repo)
A faithful **`sn-jnl.cls` shim** is bundled so the paper compiles anywhere
without the official class:

```bash
cd paper
xelatex -interaction=nonstopmode main.tex
bibtex   main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```
(XeLaTeX is used because the shim sets TeX Gyre Termes via `fontspec`.)

## Build for Springer submission (Q1 journal)
1. **Delete** the bundled `sn-jnl.cls` shim.
2. Get the **official** `sn-jnl.cls` from the Springer Nature LaTeX template
   (<https://www.springernature.com/gp/authors/campaigns/latex-author-support>)
   or open the "Springer Nature LaTeX Template" on Overleaf.
3. Compile with `pdflatex` (or the journal's required engine).

`main.tex` needs **no changes** — it already uses the official macro API. Pick the
journal article-class option your target journal specifies (e.g.
`sn-mathphys-num`, `pnas`, `default`); the shim ignores the option, the official
class honours it.

> The shim reproduces the author-facing macros and a clean single-column journal
> layout for offline preview only; the official class controls the exact Springer
> styling and is required for submission.
