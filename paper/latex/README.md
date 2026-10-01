# Progress report, LaTeX source

M.Tech Dissertation Phase-I. Kaustubh Pandey (2025PA17326), NSUT New Delhi.

## Compiling on Overleaf

1. Upload this whole folder as a zip (New Project → Upload Project).
2. **Menu → Compiler → XeLaTeX.** It will not build with pdfLaTeX, because the
   Devanagari and IPA fonts are loaded through `fontspec`.
3. Menu → Main document → `main.tex` if Overleaf does not pick it up.

Three passes are needed for the table of contents, the list of figures and the
list of tables to settle. Overleaf does this automatically; locally run
`xelatex main.tex` three times.

## What is in here

```
main.tex            the whole report, one file
figures/            17 PNGs at 300 dpi, plus the cover logo
fonts/              the two font files the project loads
README.md           this file
```

## Fonts

| Use | Font | Where it comes from |
|---|---|---|
| Body | TeX Gyre Termes | TeX Live. Metrically identical to Times New Roman |
| Devanagari | Noto Sans Devanagari | `fonts/`, shipped with the project |
| IPA | Noto Serif | `fonts/`, shipped with the project |

TeX Gyre Termes is used rather than Times New Roman itself because Overleaf has
no licence for the Microsoft font. The two are metrically identical, so line
breaks and page counts match. If you compile locally on a machine that does have
the real font, swap the `\setmainfont` line near the top of `main.tex`:

```latex
\setmainfont{Times New Roman}
```

IPA is wrapped in `\ipa{}` and Devanagari in `\dev{}` because TeX Gyre Termes
has no glyphs for either. Leave those commands in place when you edit.

## The cover logo

`figures/nsut_logo.png` is a **placeholder**. Replace it with the real NSUT
logo, keeping the same filename, before you submit.

## Formatting, against the department notice

| Requirement | Where it is set |
|---|---|
| Times New Roman, 12 pt | `\documentclass[12pt]` and `\setmainfont` |
| 1.5 line spacing | `\setstretch{1.5}` |
| Left 1.5 in, right 1 in, top and bottom 1 in | `\geometry{...}` |
| Chapter headings 14 pt bold uppercase | `\titleformat{\section}` |
| Section headings 12 pt bold | `\titleformat{\subsection}` |
| Subsection headings 12 pt bold italic | `\titleformat{\subsubsection}` |
| Table titles above, figure captions below | `\caption*` placement in each float |
| Roman front matter, arabic from Chapter 1 | `\pagenumbering` calls |

Current build: 48 pages, inside the 40 to 50 the notice allows. 8 pages of front
matter in roman numerals, 40 numbered pages from Chapter 1 onward, including the
references and the four appendices.

## References

Thirty entries, numbered in order of first citation, written out as an
`enumerate` rather than a `.bib` file so the numbering cannot drift. If you add
a citation, add its entry in the position where it is first cited.
