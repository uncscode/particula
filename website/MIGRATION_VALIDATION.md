# Starlight migration validation

Local implementation and acceptance checks are complete. PR-hosted checks and
production verification after a future merge are separate gates. No package
release or production deployment was performed.

## Content and execution

- The baseline inventory covers all 101 Markdown sources, 59 existing Python
  sources, and 55 original notebooks; eight notebook-only theory pages now have
  percent-cell Python replacements. No baseline publication source is missing.
- All 67 runnable examples have fresh successful 300 DPI bundles, including
  eight native CUDA examples on the RTX 5060. Long scientific workloads retain
  their parameters; the soot run took 1,367 seconds. Historical benchmark viewers
  execute against their declared recorded inputs rather than new measurements.
- All 109 current PNG figures were visually reviewed. Code, figures, tables,
  text, input hashes, source revision/hash, dependency versions, device, and
  elapsed time are retained. Matching example fingerprints do not certify every
  later package revision; each bundle identifies its recorded execution state.
- A second validation-only run verified all 67 bundles, executed 53 ordinary CPU
  examples without rewriting their publication results, and explicitly skipped
  14 CUDA/long workloads whose refreshed bundles had already passed.
- Progress streams render in closed, scrollable disclosures. Scientific outputs
  remain visible. All current manifests pass the account/checkout-path scrub.
- All 55 original notebooks and the retired MkDocs implementation are preserved
  under `.trash/mkdocs-starlight-2026-11-13/`. The 84 archived files have verified
  SHA-256 mappings. There is no automatic deletion date.

- The final compatibility audit records and verifies all 1,860 legacy API
  anchors across 237 pages. Concrete-only members and special methods retain
  their documentation and source links; nested docstring anchors resolve to
  their owning symbol sections. The old Aerosol.__str__ link was checked in
  the local browser.

## Build and automated checks

- Exact committed checkout `3c5f08619e02795ced67b607b343105e8e27d930`:
  Python 3.12 installation with pinned documentation constraints, fresh Node
  26.13.0 `npm ci`, and full production build pass.
- Rendered verification: 538 HTML files (408 content pages and 130 redirects),
  237 API modules, no local link/asset/fragment errors. Compressed search-index
  and sitemap checks exclude redirects and archive routes.
- Source and example suites: **8,555 passed, 687 skipped**, using the hosted-CI
  CPU baseline. GPU execution was separately verified for all eight examples.
- Infrastructure: **163 passed**, including rejection of reintroduced notebook
  publication sources.
- Documentation wrapper/schema tests: 48 passed. JavaScript hero and Markdown
  extension test files pass. Focused graph lifecycle/example tests: 211 passed,
  2 skipped. Resident stream/resource/checkpoint tests: 262 passed.
- Wheel and source distribution built successfully from the committed checkout;
  each contains 556 entries, with no `.trash` or `node_modules` material. These
  are local packaging checks, not release artifacts or release approval.

## Defects resolved during fresh execution

- Resident stream reset allocated CUDA kernel inputs on CPU. Temporary lane and
  word arrays now use the validated state device; CPU and CUDA regression tests
  cover exact selected-lane behavior (`8643e0ae1`).
- Replay structural invalidation waited on its own launch lease. Validation now
  precedes lease admission under the shared lock; teardown/launch concurrency
  and prelaunch rejection remain covered (`0dd628a75`).
- The graph-capture example now releases its owned Warp 1.11 graph through Python
  lifetime cleanup and keeps annotation-only imports lazy. General resident
  invocations use the required canonical `cuda:0` device spelling.

An exploratory broad native-CUDA suite on unchanged baseline sources stopped
with five failures after 6,603 passes: one communication volume-evolution
numerical mismatch and four invalid-device fixture setup failures. This broad
CUDA baseline is **not** claimed passing. These are separate from the successful
migration example runs and focused regression checks. Griffe's narrowly scoped
external-alias/test-helper diagnostics are recorded in `API_DIAGNOSTICS.md`.

## Desktop and mobile review

Reviewed desktop 1440×1000 and mobile 390×844 layouts: top/left navigation,
sidebar preference across reloads, current-page ancestry, centered reading
layout, compact page outline, scoped search and result navigation, light/dark
themes, API signatures/source links, code expansion, math, 300 DPI figures,
progress disclosures, and icon-only hero pause/play. The homepage starts with
its shared menu collapsed. Same-site Markdown links remain inside previews.
Screenshots are retained locally in ignored `website/qa/`.

## Publishing and rollback

PR, main-push and merge-queue builds validate saved bundles and publish a
validation reports, execution inventory, source/archive
inventories and this report. Only successful upstream `main` builds deploy to
`gh-pages`; PR and merge-queue builds cannot publish production. Existing Python
and Ruff check names are unchanged. PR site previews are disabled; only
production builds upload the site for deployment.

The pre-cutover publication revision is
`6597906ba8dd09bdccb9c3e502d8155c6668a1d4`. The archive README documents restoring
that complete published tree without force-pushing, and reverting the complete
migration PR for a source rollback. Archived files alone are not a runnable
backup. After a future merge, verify production home/search assets, redirects,
and representative example/API pages before accepting the cutover.

## PR review follow-up

- Nucleation now accepts an explicit Warp device and its registry invocation
  selects CUDA. The refreshed bundle records native CUDA kernel loads; the
  superseded CPU bundle has a corrected device label and correction note.
- Drift-triggered replay cleanup releases native handles outside the global
  provenance lock. Admission reauthenticates after validation and rejects
  intervening revocation. Regressions cover a release callback acquiring the
  lock from another thread and revocation before launch admission.
- The build wrapper rejects external directories and symlink escapes even when
  they contain documentation marker files. External worktrees must be the
  agent's active workspace.
- Focused graph and nucleation suite: **202 passed, 1 skipped**, including both
  CPU and CUDA nucleation conservation checks. The affected native CUDA graph
  example was refreshed successfully. Mypy passes all 535 source files.
