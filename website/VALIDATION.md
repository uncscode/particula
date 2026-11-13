# Local demo validation

Validated 2026-11-12/13 on branch `codex/starlight-local-demo`, based on
`45f4d53e931e9fc4e57073b15895fe01a10f9cca`.

## Completed

- Production build: 253 content pages; post-build verification checks 261 HTML
  files including eight redirects, 237 API modules, local links, assets and
  fragment targets. No verification errors. Pagefind index generated.
- Ten documentation infrastructure tests pass, including stale-result display,
  output integrity, failed-refresh preservation, prose-only changes, section
  boundaries, declared inputs and CUDA execution skips.
- Focused infrastructure plus existing native/example tests: 30 passed.
- Ruff check and formatting pass for the added Python implementation/tests.
- Existing `mkdocs build --strict` and full `mkdocs-full.yml` strict builds pass.
- Explicit fresh runs saved distribution and condensation CPU figures and a
  CUDA coagulation result on the local NVIDIA RTX 5060. Ordinary validation
  reran the two CPU examples and skipped CUDA; it did not replace saved output.
- Source watcher regenerated the authoring page after a temporary text edit;
  the text was restored and the temporary development server stopped.
- Browser review: desktop and 390-pixel mobile layouts, light/dark themes,
  collapsible code, successful clipboard copy and Python download, loaded saved
  figures, inline/display mathematics, rendered Mermaid, API class anchors and
  source links. Search for ParticleData returned API results. No browser
  console warnings/errors were observed. No page-width overflow in tested views.
- Preview remains at `http://127.0.0.1:4321/particula/`.

Screenshots are local review artifacts in ignored `website/qa/`:
`home-desktop.jpg`, `example-desktop.jpg`, and `theory-mobile.jpg`.

## Boundaries and remaining migration work

This is a representative local demo, not a complete documentation migration or
release. Existing MkDocs publishing, notebooks, scientific sources and tests
remain in place. No GitHub Pages workflow was switched.

Griffe diagnostics are retained in `.cache/griffe.log`: three unresolved
aliases and docstring warnings in test helpers loaded before test modules are
pruned. Generated API coverage excludes test/build modules, matching the old
module inventory. Private modules are included to preserve that inventory;
listing an object does not establish a new public-API support promise.

Saved output records source fingerprints, executed source, UTC time, library
revision and environment. A matching example fingerprint is not evidence that
it was rerun after every library change. Scientific prose was carried over,
not independently scientifically re-reviewed. External published links outside
the demo subset are labeled; external URL availability is not part of the
local link validator.

Before full migration: audit remaining notebook-only content and Markdown
extensions; finish redirects and navigation; add CI publication/result policy;
review API aliases and support visibility; then review GitHub Pages cutover.
