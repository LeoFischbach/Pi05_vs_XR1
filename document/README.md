# Computer Vision Project Paper

LaTeX paper scaffold using the official [CVPR 2026 author kit](https://github.com/cvpr-org/author-kit/tree/CVPR2026-v1%28latex%29). The copied `cvpr.sty` and `ieeenat_fullname.bst` are unchanged from that kit. Visible `[TODO: ...]` text marks content to replace.

This is a university project paper. It shows the authors and adds margin line numbers for course review. It is not configured as an anonymous CVPR conference submission.

## Build

Install a LaTeX distribution with `latexmk`, then run:

```sh
latexmk -pdf -outdir=out main.tex
```

The PDF is written to `out/main.pdf`. Replace the TODO placeholders in `main.tex` as the paper develops. When you add citations, put the BibTeX entries in `references.bib` and uncomment the bibliography block in `main.tex`.

## Files

- `main.tex`: paper source and author information
- `references.bib`: bibliography entries
- `cvpr.sty`, `ieeenat_fullname.bst`: official CVPR 2026 style files

The downloaded `cvpr/` folder is retained locally for reference and excluded from Git. Generated files in `out/` are also excluded.
