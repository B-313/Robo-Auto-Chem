# Export Markdown to PDF

Use [pandoc](https://pandoc.org/) to convert any Markdown file in `docs/` to a clean PDF.

## Prerequisites

| Tool | Install |
|---|---|
| **pandoc** | `sudo apt install pandoc` (Linux) / `brew install pandoc` (macOS) / [pandoc.org/installing](https://pandoc.org/installing.html) |
| **LaTeX engine** | `sudo apt install texlive-xetex` (Linux) / `brew install --cask mactex` (macOS) — needed for PDF generation |

Verify both are available:

```sh
pandoc --version
xelatex --version
```

## Convert `linkedin_post.md` to PDF

Run from the **repository root**:

```sh
pandoc docs/linkedin_post.md \
  --pdf-engine=xelatex \
  -V geometry:margin=2cm \
  -V fontsize=11pt \
  -o docs/linkedin_post.pdf
```

The output file `docs/linkedin_post.pdf` is listed in `.gitignore` and will not be committed.

## Convert any file in `docs/`

Replace the input and output paths as needed:

```sh
pandoc docs/<input>.md \
  --pdf-engine=xelatex \
  -V geometry:margin=2cm \
  -o docs/<output>.pdf
```

## Optional: HTML preview (no LaTeX required)

```sh
pandoc docs/linkedin_post.md -o docs/linkedin_post.html
```

Then open `docs/linkedin_post.html` in a browser and use **File → Print → Save as PDF**.

## Notes

- Generated PDFs (`docs/*.pdf`) and HTML files (`docs/*.html`) are excluded from version control via `.gitignore`.
- If `xelatex` is unavailable, substitute `--pdf-engine=pdflatex` (may require extra LaTeX packages for Unicode characters).
