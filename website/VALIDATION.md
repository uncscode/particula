# Local demo validation

## Centered and scoped search — 2026-11-13

The desktop search bar is centered independently of the logo/menu and theme
controls. On phones it occupies a full-width second header row. The search
dialog immediately offers Guide, Examples, Theory, API, Contribute and Home
checkboxes. No selections searches everything; multiple selections use OR,
and Clear filters restores the full-site search. Results retain section links,
highlighted excerpts and pagination. Escape and the close button dismiss the
dialog; Ctrl/Cmd+K opens it.

Every indexed page has a directory-derived Section filter: 238 API pages,
five Examples, four Guide, two Theory, two Contribute, and one Home. Browser
checks for “condensation” returned 41 API pages, two Examples, 43 combined and
49 without filters. The implementation uses Pagefind's explicit `any` operator
because its default array filtering is AND; see the
[Pagefind filtering API](https://pagefind.app/docs/js-api-filtering/).
Desktop and mobile controls, result links, reset, pagination and no-result
states were checked. Screenshots are under `qa/search-examples-filter.jpg`
and `qa/search-filter-mobile.jpg`.

## Drawer spacing and branch defaults — 2026-11-13

Desktop reading layouts reserve an extra gutter beside an open drawer while
keeping the article centered. At 1440 pixels, the drawer-to-text gap is 48
pixels; closing it restores the wider article. The right contents menu remains
compact. Mobile retains its overlay layout.

Navigation groups now start collapsed except for ancestors of the current page.
This applies recursively to generated API groups as well as the homepage menu.
Readers can still expand adjacent branches manually. Unrelated branch state is
no longer restored from earlier visits; the drawer's open/closed preference is
still remembered separately.

## Centered reading and unified drawer — 2026-11-13

Supersedes the separate homepage navigation and space-consuming desktop sidebar
described below. Every page now uses one left-hand popover navigation tree.
The hero starts closed regardless of the saved documentation preference.
Desktop documentation remembers explicit open/closed choices; mobile use does
not overwrite that preference. Opening the drawer expands the current page's
ancestor groups and centers its link within the drawer without scrolling or
shifting the article. Escape, the menu button, the drawer close button and
outside clicks dismiss it through native popover behavior.

The article uses equal outer gutters, with a separate sticky contents menu on
the right. That menu uses smaller text and its intrinsic width, capped at 14rem;
it does not grow with the viewport. At 1440 and 1920 pixel viewports, measured
article centers matched the available viewport center within one pixel.
At 1920 pixels, the guide's short contents menu measured about 122 pixels wide.
The nested ParticleData API link centered within one pixel of the drawer's
midpoint, with all ancestor groups open. Mobile hero and documentation menus
were reviewed at 390 pixels. Screenshots: `qa/centered-article.jpg`,
`qa/current-page-drawer.jpg`, and `qa/unified-menu-mobile.jpg`.

## Navigation follow-up — 2026-11-13

Added homepage links for Guide, Examples, Theory, API and Contribute, with a
native popover menu at smaller widths. Documentation pages now offer a
keyboard-operable Show/Hide sidebar button. The preference is stored locally
in the browser and restored before paint; blocked storage falls back to the
expanded layout. Mobile navigation remains independent of that preference.

Browser checks confirmed both preference states survive reloads, the collapsed
state survives page navigation, and content gains width when collapsed. At
390 pixels, the documentation menu still opens with the desktop sidebar hidden;
homepage menu links navigate and Escape dismisses the popover. Desktop and
breakpoint checks included 1280 and 1152 pixels. Screenshots are under
`website/qa/home-navigation-desktop.jpg`, `home-navigation-mobile.jpg`, and
`sidebar-collapsed-desktop.jpg`. The production build and complete local-link
verification pass with the new controls.

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
