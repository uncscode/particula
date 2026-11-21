---
name: particula-toc
description: Design or revise a square scientific table-of-contents (TOC) figure or graphical abstract to accompany a study title. Use for small-format composition, scientific visual metaphors, accessible descriptions and print-ready export.
---

# Design a TOC figure

Design the title and TOC figure together so a browsing reader gets the gist in about three seconds. Let the title name the finding and the picture show the relationship behind it. Use the report text and quantitative figures for the magnitude, uncertainty and explanation.

For an existing study, review the artwork again once the narrative is final. If a later analysis now leads the report, show its observable and reference case, and update the listing artwork and alt text together.

## Choose the image

- Write a private sketch brief naming the subject, the comparison and the direction supported by the evidence. Identify what remains unknown before choosing arrows, geometry or motion that could imply a mechanism.
- Sketch one recognizable subject cue and one relationship. Size, spacing, overlap, density or the separation of two marks can show what grows, disappears or diverges. For a schematic, amplify a supported difference in density, separation or size until its direction is easy to see. For example, two equal chamber outlines can show a full particle cloud beside a sparse one. Use alt text to identify the schematic and keep the quantitative result in the report. When a figure uses a numbered scale, percentages or a one-symbol-per-particle key, draw those values to scale. Use strong silhouettes, empty space and a small palette. Support important color differences with shape, position or line style.
- Use a few direct labels when they help readers identify the subject or compare marks. Typically use fewer than five visible words in total across the figure. Place words beside what they name. Use the remaining space for meaningful details such as the ion pair and beam in the CLOUD study. Keep the headline, explanatory sentences and precise values in the report.
- For example, the CLOUD title names the underpredicted humidity response. Two marks starting together and ending at different heights inside a humidity motif convey the gap. A drawing of ions sticking to a wall would imply a cause the measurements leave open.

## Color and lettering

Use the collection’s [TOC color guide](../../references/toc-design.md#collection-colors) for every square, including artwork written by a study script. Start with the paper-yellow canvas. Use ink for a measured or archived reference when it has its own mark. Label each comparison directly and give it a distinct position, shape or line style.

Read the figure beside its listing title at thumbnail size. Aim for about five visible words. Remove a repeated headline, subtitle, conditions or row of numbers before shortening a scientific name. Keep a unit or threshold when the graphic needs it. If several methods are the finding, keep each method name and let the marks carry the magnitude. Record precise endpoints, units and conditions in the accessible description and report.

## Export and check

For Particula, produce a **3 × 3 inch square**. Use a self-contained SVG with equal physical dimensions (for example, `width="3in" height="3in" viewBox="0 0 300 300"`), or a **900 × 900 pixel** raster with **300 pixels/inch** set on both axes. Preserve crisp edges and generous margins. Keep the detailed quantitative figure in the report.

Place the artwork beside or directly below the report title. Set `tocFigure` on the study’s resource entry in `website/topics.json` to its image path relative to `docs/images/` so the Studies and Topics carousels can display it with the study title. Supply concise alt text conveying the relationship; for standalone SVG include an accessible title and description. Inspect the 3 × 3 inch export and a **2 × 2 inch proof** beside the actual report title. At 300 pixels/inch the proof is 600 × 600 pixels. Display or print it at two inches across to judge readability. Check that labels, contextual details and the main relationship remain distinct in grayscale. Give the title and two-inch proof a three-second glance. Can you name what is compared and which side has more, without counting symbols or reading a scale? If it takes longer, simplify the scene or widen the visual contrast. If the title already supplies a detail, try removing it from the image.

For the advertising, branding, signage and scientific publishing sources behind these choices, see `../../references/toc-design.md`.
