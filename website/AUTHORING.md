Keep an ordinary Python script as the only maintained executable source.
Use `# %%` for code sections and `# %% [markdown]` for explanatory comments.
The source remains downloadable and executable with Python.

## Register a study

Choose one primary home below `docs/studies/`: `examples/`, `tutorials/`,
`literature/`, `validation/`, or `benchmarks/`. The main resource in
`website/topics.json` owns its stable study ID, topics, script keys and
companion sources. A report can own several runs or have no code. Register
each runnable Python source once in `website/examples.json`, preserving its
execution key when moving or renaming a source. Use precise relationships
for supporting guides, equations, API symbols and evidence. Verify the reverse
link at the target. Theory pages retain derivations and link at the equation;
the result page links back to that equation.

Keep a report, its script and a frozen specification together as one sidebar
choice. A collection with seven choices merits review for subject or task
groups. Make each group landing meaningful in the installed Markdown bundle.
Run `python scripts/docs_studies.py sync` to regenerate the marked lists in the
installed Markdown. Verify them with `python scripts/docs_studies.py check`.
Study reports list each companion calculation with its code and downloads in
the generated **Study resources** section after the report. Keep reproduction
instructions in the report’s closing sections. Check the generated catalog
and the source and ZIP downloads on the built site.

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
Replot checks saved-file hashes, the simulation fingerprint and the scientific
Python sources. Changes under `particula/plotting/` affect figure rendering.
The full package hash still identifies the original execution. Changes to
simulation code or inputs require a full refresh. Replot executes the
renderer on CPU, without requiring CUDA or advancing a simulation. It preserves
the original execution timestamp, device, revision and versions, and records
rendering time/revision separately. `executed_source.py` preserves the original
source when the renderer changes. Missing numerical data requires one data-saving
refresh; images and printed endpoint tables cannot recover missing histories.

The execution banner displays the original script's elapsed wall time from
`elapsed_seconds`, never `render_elapsed_seconds`. This measures the complete
original execution, including setup and any plotting within that script; it is
not an isolated physics-kernel benchmark. Replotting preserves this duration.

Save compact plotted summaries: observation times, counts, histogram bins,
quantiles and the settings needed to interpret them. Do not save full particle
states, checkpoints, trajectories or unused diagnostics. The serialized numeric
JSON has a **10 MiB hard limit per example**, enforced while writing. This limits
saved plot data, not simulation RAM or figure size. Oversized data fails clearly
and preserves the previous successful bundle; it is never silently truncated.

A historical study can pin its original revision, the full Python package
hash and the SHA-256 of its published `data.json` in the registry under
`historical_run`. To redraw it, point the same verification checks at a checkout of the executed revision:

```bash
python scripts/docs_examples.py replot gpu-prepared-cyclic-chamber --science-checkout ../particula-original
```

The current checkout supplies the renderer. The historical checkout supplies
the scientific source and inputs for verification. The page identifies the
historical simulation and links its executed source, arrays and original
device. A full refresh with `--science-checkout` runs the current study script
and its declared inputs against the pinned library. Subsequent replots check
that study and the digest of its freshly recorded data. The registry's data
digest identifies the original archive, whose timing fields may differ. To
publish a result from the current model, run a full refresh without the flag.

For the stochastic and lifecycle tutorials, the example files were first
committed after the scientific checkout. In a dedicated checkout of the pinned
`revision`, restore the published study files from its pinned `study_revision`
before verifying the saved observations. Both pins appear in the registry.

```bash
git restore --source=535d02b7b06ba782aa2bd5d1c60890e06adbb654 -- website/examples.json docs/examples/gpu/prepared/gpu_prepared_stochastic.py docs/examples/gpu/prepared/gpu_prepared_lifecycle.py scripts/docs_gpu_figures.py scripts/gpu_studies
```

Those two original runs saved images alone. The new native CUDA executions
save the returned trajectory report and its initial radius histogram. The
lifecycle panel uses the mean initial histogram across boxes. The saved
executed source and input hashes identify the scientific calculation.


The biomass aerosol tutorial generates four plots during a full CUDA run.
Use the current checkout to run its condensation control and three collision
models from the same 10,000-particle starting population.

```bash
python scripts/docs_examples.py refresh examples-simulations-biomass-burning-cloud-interactions
```

The tutorial records split intervals, maximum collision passes and total
mergers for each collision model. Its saved output includes the conservation
checks, device and library revision.

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
notebooks remain in Git history at `4f669de11`, recorded as `archive_revision`
in `website/migration-inventory.json`. They are no longer stored in the checkout.
Internal execution uses temporary notebook objects to capture
rich outputs, without introducing a second maintained source file.

Keep the scientific library unchanged during a refresh. Publication records its
source hash before execution and rejects a different hash afterward, preserving
the previous bundle. The reported Particula version comes from that checkout;
when installed distribution metadata differs, it is retained separately as
`particula-distribution` rather than used to label the executed source.

For styled scientific figures, follow the
[figure style guide](https://uncscode.github.io/particula/guide/getting-started/figure-styles/) and
[gallery](https://uncscode.github.io/particula/studies/tutorials/figure_styles/). Use one PNG per
figure for both website themes. Set `"full_canvas": true` in the example
registry for Particula preset figures. This preserves their physical canvas
and fingerprints the shared style and layout code. Call `plotting.fit(fig)`
after all labels and colorbars are in place.

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
are needed. Test extracted bundles independently before publication. For a
separately installed optional model, declare its source and parameter files
in `science_inputs`. These paths participate in the scientific fingerprint
and saved hash inventory while the model ships in its own distribution.

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

## Topic explorer and narrative review

Use the [documentation guide](https://uncscode.github.io/particula/contribute/documentation/) for topic
placement and the shared scientific narrative framework. The topic registry is
`website/topics.json`; local SVGs live in `website/public/topic-icons/`.
For an illustrated study, set its resource entry’s `tocFigure` to the file path
relative to `docs/images/`. Put that image, with descriptive alt text, in its
own paragraph and follow it immediately with the study’s opening paragraph.
For a Python study, use the same order in its opening Markdown cell. The page
template places the TOC in the right 40% of the reading column. The paragraph
starts on the left and continues across the full width beneath the figure. Narrow
reading columns stack the paragraph above the centered figure.
The Studies and matching topic carousels reuse that
resource’s title and destination. The home carousel selects study IDs in
`docs/index.md`.

Preparation validates coverage and generates topic MDX adapters. Do not edit
`website/src/content/docs/` directly. The offline entry remains maintained in
`docs/guide/topics/index.md`.

Review the opening question, code, figures, and conclusion together. Prose-only
pilot revisions retain the original scientific outputs when executable cell
fingerprints match. Topic icons are conceptual illustrations, never observations.
