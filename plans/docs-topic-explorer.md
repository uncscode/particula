# Topic explorer and scientific narrative rollout

## Current scope

The user approved the preview and expanded this PR to the full `docs/examples/`
collection and richer topic pages. The [conversion ledger](docs-example-conversion.md)
records all 88 reviewed pages, scientific corrections, refreshed evidence, and
final validation. The first-increment record below describes the earlier pilot.

## First increment (historical)

Branch: `codex/docs-topic-explorer`, based on upstream `103832806`.

The first increment maps the complete authored documentation collection, adds
an illustrated topic explorer, updates documentation/review skills, and revises
four pilot pages. Broad conversion of the remaining docs is deferred until the
user has reviewed the local preview. A map association is not a scientific review.

The canonical inventory is `website/topics.json`: every authored Markdown or
Python page has either a primary topic (and optional additional associations)
or an explicit supporting-page reason. Generated API symbols remain discoverable
through the canonical API overview and existing API navigation; they are not
claimed as individually curated topic entries.

The explorer retains the five section tabs and published article routes. Topic
IDs are stable, branching is two to four children, and leaves have resources.
Topics connect by parent, prerequisite, and reciprocal related relationships.
Alias lookup distinguishes equivalent names from broader search phrases.

## Pilot intent and evidence

| Page | Reader capability | Evidence boundary |
| --- | --- | --- |
| Size distributions | Distinguish density, bin number, and coordinate conversion | Existing calculations; normalization is built into the helper and does not prove tail coverage |
| Radius-bin condensation | Interpret an initial mass-density rate and its sign | No time evolution; the figure remains the initial PMF |
| Dynamic viscosity | Use the gas-property value with explicit units and limits | Single-temperature air correlation, not experimental validation |
| GPU environment sweep | Read size, number, and ion/charge diagnostics together | Retained single-realization CUDA data, no new statistical significance claim |

Executable cell contents were compared with the branch base and remain identical
in all four pilots. Saved outputs remain current; no scientific rerun or new
numerical evidence is implied by the editorial changes.

## Review before broad conversion

Review the local explorer on desktop and mobile. Try “humidity growth,”
“distribution normalization,” and “DNS” without using exact page titles.
Check topic transitions, breadcrumb/back navigation, ordinary links without
JavaScript, light/dark contrast, keyboard focus, and reduced-motion behavior.

For each pilot, read the opening question, computation, displayed output, and
closing interpretation as one argument. Confirm that a researcher can explain
what was established and identify a reasonable adaptation with necessary checks.

## Original editorial backlog and disposition

Items 1 and 3–6 are addressed by the full example conversion. Item 2 remains
a separate theory-page follow-up outside the expanded `docs/examples/` scope.

1. **Scientific consistency:** reconcile the cloud-chamber single-cycle closing
   mass-drift statement with its opening inventory checks; inspect remaining
   process examples for promises of integration when only rates are calculated.
2. **Theory interpretation:** improve DNS comparison closure using actual saved
   differences and digitized-source limitations. Review mean free path and
   adjacent theory pages for units, domain, limiting behavior, and connections.
3. **Process progression:** review condensation, coagulation, nucleation, and
   wall-loss families as learning sequences. Distinguish repeated API patterns
   from genuinely different scientific questions before merging any pages.
4. **Thermodynamics:** use the objective-oriented index as a starting point;
   distinguish activity, phase equilibrium, and implementation-pattern tutorials.
   Remove outdated claims about paired notebooks after verifying link handling.
5. **Applications:** improve chamber, atmospheric, combustion, and exposure
   examples with explicit controls, diagnostics, inference limits, and research
   adaptations. Do not infer physical validation from conservation assertions.
6. **Advanced methods:** revisit GPU and extension pages for prerequisites,
   execution evidence, and links between performance and scientific capability.

Use the packaged topic/narrative references and current contributor guide.
Preserve URLs unless a move solves a demonstrated navigation problem. Any later
scientific changes require complete workload refreshes and the relevant numerical
checks. The approved follow-up converts the complete example collection in this PR;
publication remains a separate action.

## First-increment validation record (historical)

- Production website build passed: 532 content pages; verification covered 662
  HTML pages including redirects, 303 API modules, and 1,860 legacy anchors,
  with zero errors. Build and tests were run sequentially because they share
  generated downloads.
- Website tests: 53 Python tests and 14 JavaScript tests passed. The notebook
  tests required local kernel access; the initial restricted run was stopped.
- Plugin distribution tests: 19 passed. Both changed skills passed the skill
  validator; packaged payload synchronization and all 11 skill packages checked.
- Topic, site, and structure checks: 44 focused tests passed, with zero structure
  errors. Ruff lint/format and whitespace checks passed.
- Saved-example check: all registered results are current; no executable changes
  selected against upstream main. The four pilots retain their executable cells.
- Browser review: 1280-pixel light and 390-pixel dark layouts, no horizontal
  overflow; phrase/alias lookup, global search topic results, keyboard activation
  and focus, back/forward, and ordinary links in a script-disabled frame worked.
  Reduced-motion rules were inspected in CSS; OS preference emulation was not
  available in the browser tool. This is not a full accessibility audit.
- Preview: `http://127.0.0.1:4321/particula/guide/topics/`.

The registry contains 33 topics, 160 resource entries (including the API overview),
and 35 explicitly classified supporting pages. This covers 194 authored pages.
The plugin version is 0.3.3; the installed cached plugin is not changed by this PR.
