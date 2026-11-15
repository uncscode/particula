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
a fresh kernel with temporary working storage and captures figures, text and
explicitly declared compact numerical plot data. Outputs live directly in
`website/results/EXAMPLE_KEY/`: `manifest.json`, `source.py`, figures, and
optional `data.json`. There are no timestamped run folders or pointer files.
Figure filenames use their order within the code section, independent of log
message counts. Git preserves earlier versions. Failed execution or rendering
keeps the previous successful bundle. Review and commit the bundle together
with source changes.

For a plotting-only change, reuse numerical data instead of rerunning physics:

```bash
python scripts/docs_examples.py replot gpu-environment-sweep
python scripts/docs_examples.py migrate-results  # one-time legacy layout migration
```

Replottable examples separate `simulate()` from `plot_results(report)`.
The registry declares `data_expression`, `replot_invoke`, `render_functions`
and `plot_inputs`. Only pure rendering functions/helpers belong in these
exclusions: they must not change scientific parameters, processes or diagnostics.
Replot checks saved-file hashes, the simulation fingerprint and library hash;
changes to simulation code or inputs require a full refresh. It executes the
renderer on CPU, without requiring CUDA or advancing a simulation. It preserves
the original execution timestamp, device, revision and versions, and records
rendering time/revision separately. `executed_source.py` preserves the original
source when the renderer changes. Missing numerical data requires one data-saving
refresh; images and printed endpoint tables cannot recover missing histories.

Save compact plotted summaries: observation times, counts, histogram bins,
quantiles and the settings needed to interpret them. Do not save full particle
states, checkpoints, trajectories or unused diagnostics. The serialized numeric
JSON has a **10 MiB hard limit per example**, enforced while writing. This limits
saved plot data, not simulation RAM or figure size. Oversized data fails clearly
and preserves the previous successful bundle; it is never silently truncated.

## Validate changes

```bash
python scripts/docs_examples.py check --changed --base upstream/main
```

Changed CPU examples execute for validation without replacing publication
results. After a plotting-only replot, validation reuses saved data and does
not repeat the simulation. Declared CUDA or long-running examples report an
explicit execution skip. Syntax, saved-file integrity, and freshness are checked
for every example.
Use `--base` with the pull request base revision in CI.

The example registry declares its source, execution policy, device, timeout,
optional invocation, and input/helper files. Executable changes and declared
input bytes invalidate rendered results; plotting-only changes can be replotted
from compatible saved data. Ordinary prose changes do not invalidate results. Changes
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

## Maintain the LLM documentation index

`website/public/llms.txt` is a small curated index published at
`https://uncscode.github.io/particula/llms.txt`. Keep roughly 10–15 direct links
to maintained guides, executable Python examples and Particula Helper resources.
Prefer `https://raw.githubusercontent.com/uncscode/particula/main/` file URLs;
use the published API Markdown index for generated API documentation. Do not link archives
or generated local files. Raw documents retain repository-relative links.

The index explicitly describes development `main`. If the site later publishes
versioned release documentation, update the index and its verifier together to
use matching tags or commits. Keep version guidance visible to readers.
The normal build checks the published index and its targets offline against the
checkout and built site. After deployment, verify the public index and links
separately; a local build does not prove that GitHub Pages has updated.
