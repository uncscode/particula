# Astro + Starlight migration plan

## Decision and scope

Use Astro + Starlight for a static GitHub Pages site. Keep ordinary percent-cell
Python scripts as the executable source of examples, with generated Markdown,
collapsible code and visible saved figures. Run Python during an explicit
refresh or selected validation, never as part of a documentation build.
Browser PyScript is a separate future experiment.

The current API pipeline uses Griffe through mkdocstrings; Griffe extracts a
structured Python/docstring model rather than being a general Markdown
converter. The demo retains Griffe and uses starlight-pydocs for Starlight
rendering, including signatures, symbol anchors and source links.

## Phase 1 — representative local demo (completed)

- Synchronize upstream/main, local main and the fork; create an isolated branch.
- Establish the GitHub Pages base path, branding and five reader paths.
- Include guides, two CPU plot examples, one explicitly run CUDA example,
  theory with equations/diagram, and the complete existing API module inventory.
- Introduce explicit result refresh, immutable generation bundles, provenance,
  source downloads, stale/missing-result states and changed-source validation.
- Verify production search, links, API coverage, desktop/mobile reading and
  coexistence with MkDocs. See VALIDATION.md and README.md.

## Phase 2 — content migration after demo review

1. Inventory every published route, notebook pair and notebook-only page.
2. Convert remaining notebook-only sources to percent-cell Python or narrative
   Markdown; retain figures and provenance. Validate the actual workloads.
3. Classify CPU, long-running and CUDA examples. Declare helper/data inputs and
   establish result size limits. Keep explicit refresh separate from CI checks.
4. Translate MkDocs-specific admonitions, tabs, includes, math and diagrams.
   Audit scientific references, downloadable assets and fragment links.
5. Complete navigation, redirects and public/re-export API policy; resolve
   material Griffe warnings. Compare the full route inventory against MkDocs.

## Phase 3 — CI and GitHub Pages cutover

1. Build from locked Node and Python dependencies; cache dependency downloads.
2. Validate changed executable examples without overwriting committed results;
   skip explicitly classified expensive/GPU cases with a visible report.
3. Check result integrity/freshness and rendered links/API coverage on every PR.
4. Upload validation reports and use a Pages build/deploy workflow with correct
   base paths and permissions. Upload the site only for production deployment;
   do not generate downloadable PR previews.
5. Approve a complete content/URL comparison, then switch publishing. Keep a
   straightforward rollback to the previous workflow for the first release.
6. Remove MkDocs/notebook pairing only after the replacement is accepted.

## Effort and benefits

The shell/theme change is the small part; executable content, output provenance,
API rendering differences and URL preservation dominate the work. Treat the
remaining full migration as roughly 1–2 developer weeks, subject to the complete
notebook-only inventory and scientific execution time. This is a planning
estimate, not a measured commitment.

Useful capabilities demonstrated here are fast static search, a responsive
reading layout, code copying/folding, source links and component-based page
composition. Equations, diagrams, search and dark mode are also possible in
MkDocs; the migration's main new flexibility is Astro/MDX components and a
script-first publishing workflow. Interactive client-side tools can be added
selectively later without making every page execute Python.
