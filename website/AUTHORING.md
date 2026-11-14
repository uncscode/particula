Keep an ordinary Python script as the only maintained executable source.
Use `# %%` for code sections and `# %% [markdown]` for explanatory comments.
The source remains downloadable and executable with Python.

## Refresh results deliberately

From the repository root, in the development Python environment:

```bash
python scripts/docs_examples.py refresh distributions
python scripts/docs_examples.py refresh condensation
python scripts/docs_examples.py refresh gpu-coagulation
```

The CUDA example requires CUDA and never falls back to CPU. A refresh runs in
a fresh kernel with temporary working storage, captures figures and outputs,
and saves a complete successful generation. After success, superseded
generations for that example are removed; retain only the current final bundle
in the checkout. Git history preserves earlier committed provenance. A failed
run preserves the previous successful bundle. Review and commit the result bundle together with the script change.
Save the displayed figures and text, not simulation-state dumps. Small numeric
series for point, line or bar plots may accompany a downloadable script. Do not
publish full particle states, checkpoint arrays, heatmap/image matrices, raw
trajectories or unused validation diagnostics. Regenerate those by running the
script. GPU campaign export uses an explicit allowlist of plotted summaries.

## Validate changes

```bash
python scripts/docs_examples.py check --changed --base upstream/main
```

Changed CPU examples execute for validation without replacing publication
results. Declared CUDA or long-running examples report an explicit execution
skip. Syntax, saved-file integrity, and freshness are checked for every example.
Use `--base` with the pull request base revision in CI.

The example registry declares its source, execution policy, device, timeout,
optional invocation, and input/helper files. Executable changes and declared
input bytes invalidate saved results; ordinary prose changes do not. Changes
inside Particula still need the existing scientific tests and a deliberate
result refresh when appropriate. The recorded library revision remains visible.

## Build without running simulations

```bash
npm --prefix website run build
npm --prefix website run preview
```

Pages are generated from current prose/code and saved result bundles. Code is
expandable; figures remain visible. Older results are labeled and grouped
separately from the updated code. Missing results show an awaiting-results note.
Generated Markdown, API dumps, and the built website are disposable outputs.

Starlight is the sole documentation build. Retired MkDocs assets and original
notebooks are preserved under `.trash/mkdocs-starlight-2026-11-13/` and excluded
from publication. Internal execution uses temporary notebook objects to capture
rich outputs, without introducing a second maintained source file.

Figures are exported at 300 DPI. Captured diagnostics retain repository-relative
filenames and line numbers; local account and checkout paths are redacted before
saving publication bundles. Terminal progress bars render in closed disclosures.

## CUDA studies and downloadable helpers

Published GPU examples use `policy: "cuda"` and require a real CUDA device.
Their deliberate refresh can run longer than the routine ten-minute benchmark
guide; the registry declares each timeout. CI checks source and saved-result
integrity without executing these studies. A timeout or missing GPU must never
produce a replacement CPU result.

Declare every imported local helper and data file in `inputs`. These bytes
participate in freshness checks and are included in the downloadable ZIP with
`example.py`, preserving their relative import paths (or `input_destinations`).
The plain Python download remains available; use the ZIP when helpers or data
are needed. Test extracted bundles independently before publication.
