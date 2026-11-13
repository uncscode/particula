# Migration validation record

This is an in-progress acceptance record, not approval to deploy or release.

## Verified locally

- Complete production build: 408 generated pages plus 130 compatibility
  redirects, 237 API modules; rendered link/fragment validation reports no errors.
- Source structure and navigation checks pass.
- Infrastructure suite after retiring the notebook CI scripts: 160 tests pass.
- Documentation tool and schema-inventory checks: 47 tests pass.
- Output-redaction and execution-report tests: 13 tests pass.
- A real distribution example exports 300 DPI PNGs, approximately 2,600 by
  1,600 pixels, compared with the previous approximately 870 by 550 pixels.
- Progress logs are contained in closed disclosures in rendered HTML; figures
  and scientific textual results remain outside those disclosures.

## Browser review, 2026-11-13

Reviewed the production preview at `/particula/`, at desktop 1440×1000 and
mobile 390×844:

- Hero motion starts, icon-only pause switches to play, and the left menu starts
  collapsed on the homepage.
- The desktop guide menu remembers its open state across a reload. Closing it
  returns to the centered reading layout.
- Only the current section and API ancestry expand automatically; adjacent API
  module groups remain collapsed. The current page is highlighted.
- Scoped search for “coagulation” with Examples checked returned 29 results
  linking to example pages.
- Mobile menu, centered search, compact page-outline control and table layout
  were reviewed. Theme selection remains accessible inside the mobile menu.
- Light and dark themes render correctly on the reviewed home/guide/API pages.
- The Aerosol API page shows class signatures, source links, symbol anchors and
  a compact desktop outline. Its signature wraps within the mobile viewport.

Screenshots are available locally under ignored `website/qa/`:
`mobile-api.png` and `desktop-api-navigation.png`. Temporary viewport and theme
changes were reset after review.

## Acceptance still pending

- All 67 current 300 DPI execution bundles, including the full long-running
  simulations and native CUDA examples; review figures/tables and provenance.
- Final source/example checks after the resident reset fix. The hosted-CI CPU
  baseline passed 8,554 tests with 686 skips. The broad CUDA run stopped after
  five failures (6,603 passes): one volume-evolution numerical mismatch and four
  invalid-device fixture setup failures in unchanged sources. These are not
  claimed passing. The reset-specific regression suite passes 262 tests.
- Archive all 55 original notebooks after replacement acceptance.
- Clean-checkout dependency installation/build and package artifact exclusions.
- Final browser review of refreshed figures, code expansion, math, mobile search
  and redirects; final source-to-route and search-index exclusion audits.
- Migration PR, downloadable CI artifact and successful PR checks.
- Post-merge production verification is conditional on a future merge and has
  not been performed. The existing publication is unchanged.

## CUDA defect found by fresh documentation execution

Native resident graph-capture execution exposed CPU temporary input arrays in
`StreamRegistry.initialize_selected`. The indexed writer launched on CUDA but
its lane/word arrays were allocated on CPU. Commit `8643e0ae1` allocates those
inputs on the validated state device. The existing exact lane-selection test
now covers both CPU and CUDA: before the fix CPU passed and CUDA failed; after
the fix 262 stream/resource/checkpoint tests passed. No public signature or
scientific parameters changed.

The two general resident examples also require canonical `cuda:0` in their
explicit documentation invocation, consistent with the resident API's exact
native-device contract. Direct examples retain their supported `cuda` alias.

High-resolution refresh update: 66 of 67 registered examples have current 300 DPI bundles, including all eight native CUDA examples. The full soot workload remains in progress. Regenerated distribution figures measure 2605×1639 pixels (previously 868×552). Graph replay structural invalidation exposed a self-owned launch-lease wait; validation now precedes lease admission under the shared lock. The example adapter now releases Warp graph ownership through Python lifetime cleanup, matching Warp 1.11, and keeps annotation-only imports lazy. Focused graph lifecycle/example tests: 211 passed, 2 skipped; Ruff passed. Final complete migration acceptance remains pending.

Clean-copy verification: a separate local clone with the working migration snapshot installed successfully using Python 3.12 and the pinned 83-distribution constraints, followed by a fresh npm ci and production build. The verifier reports 538 HTML pages, 237 API modules, 130 redirects, and zero errors. This precedes final notebook archival and will be repeated for the committed final tree.

Reviewed all 109 saved PNG figures as contact sheets (including the prior soot figures, whose fresh replacement is pending). No missing/blank plots or export clipping was observed. Desktop math/code expansion and mobile figure layout, closed/scrollable progress disclosures, scoped search results and a result navigation passed. The screenshot mobile-high-resolution-output.png is saved under ignored website/qa. Fifty-four accepted original notebooks were moved to the archive with SHA-256 mappings; the soot original remains active until its fresh run passes.

All 67 examples now have successful current 300 DPI bundles. The final soot workload completed in 1,367 seconds with unchanged parameters; all eight final soot figures were reviewed. All 55 original notebooks are archived with preserved paths and SHA-256 mappings. The complete execution inventory was generated successfully.
