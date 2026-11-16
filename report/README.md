# Final report

`report.tex` is an IEEEtran conference-format skeleton for the final report
(<= 8 pages excluding references and appendix). `references.bib` contains the
WaNet, BadNets, Neural Cleanse, Fine-Pruning, STRIP, and Grad-CAM references.

## Compile

- Overleaf: upload `report.tex`, `references.bib`, and the `figures/` folder
  (the LaTeX references `../figures/...`, so keep the same relative layout).
- Local: `pdflatex report && bibtex report && pdflatex report && pdflatex report`.

## Status

- The main tables and figures are in place.
- `\todo{...}` marks numbers that will be filled after the remaining runs
  (3 seeds, no-noise ablation, V1/V2, all-to-all).
- The artifact appendix needs the final repository URL.
