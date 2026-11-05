# Paper figures

This directory contains an editable three-slide figure deck and its PDF export.

## Figure map

- `figure-1.png` / slide 1: overview of the multimodal training path. It shows the long visual context and few anchors, one image-conditioned record reused across epochs, and objective-aware row selection.
- `figure-2.png` / slide 2: visual dependence of the drafter. Values are the fixed-trajectory top-1 changes from Table 2 of the draft, relative to the Full view.
- `figure-3.png` / slide 3: full-scale training savings. Bars use the reported 75K-update wall times: DFlash2 196,371 s → 179,051 s and DFly 225,894 s → 193,820 s.

## Source and rendering

- `make_figures.mjs` is the editable source script and uses native PowerPoint shapes and text.
- `VelaDraft_paper_figures_v2.pptx` is the editable deck.
- `VelaDraft_paper_figures_v2.pdf` is the three-page PDF exported from the PPTX with LibreOffice.

The deck uses Arial, a white canvas, navy body text, and blue, teal, orange, and coral accents. The figures intentionally avoid confidence bars or acceptance-rate language where the draft only reports means for the fixed-trajectory probe.
