# Maintaining the documentation

Choose the reader's task before choosing a filename. Public documentation has
five sections; extend an existing topic before adding another page.

| Section | Audience and purpose | Owner |
| --- | --- | --- |
| Guide | Users installing, constructing state, choosing a process, or migrating | `docs/guide/` |
| Examples | Users running a concrete scientific calculation | `docs/examples/` |
| Theory | Readers looking for equations, assumptions, units, and literature | `docs/theory/` |
| API | Call signatures and implementation reference | Generated from package docstrings |
| Contribute | Developers changing the package, validating results, or setting up assistants | `docs/contribute/` |

## Before adding a page

1. Identify its owning topic and distinct learning objective. Search that topic
   for an existing explanation; extend it when the purpose is the same.
2. Keep the canonical contract in one place. Link to it from examples, migration
   comparisons, and validation records. Keep short local warnings where misuse
   could change a scientific interpretation.
3. Use topic folders and descriptive lowercase filenames. Python sources
   use snake_case stems so the source remains importable.
4. Add the page to `website/navigation.json` and link it from its owning index. Never add a sixth top-level tab.
5. Add purpose, prerequisites, units, device requirements, and version context.
   Keep CPU tutorials separate from optional Warp and native-CUDA-only examples.

Implementation plans and task closeouts belong in `.opencode/plans/`, not in
public feature pages. Durable scientific evidence belongs under
`contribute/validation/` with revision/device provenance, limits, and open gates.
The public roadmap contains active or pending work only.

## Move, merge, or retire

Update links, percent-cell Markdown, source imports/loaders, tests, CI selectors,
release allowlists, and assistant instructions together. Move Python sources and declared data together. Follow [Add Examples](Feature_Workflow/Details/Add_Example.md)
for full-workload execution and retained outputs.

Record meaningful old routes and split-section anchors in
`docs/.assets/relocations.json`. Redirects are generated after the search index,
so old routes do not become duplicate navigation/search entries. Test targets in
the full rendered site. Retired-only routes do not need home-page redirects.

For v0.3.0, preserve outdated originals at
`.trash/docs-v0.3.0/<original-repository-path>` and document revision, reason, and
replacement in its README. Extract unique current guidance/evidence first.
Do not publish `.trash` or include it in example discovery. An old scientific
result can remain useful; do not confuse age with obsolescence or plan completion
with measured qualification.

## Check the change

```bash
python scripts/check_docs_structure.py
npm --prefix website run build
python scripts/check_docs_structure.py --site-dir website/dist
```

The structural check covers navigation targets, authored-page reachability,
local links, moved-path references, and registered Python sources. The rendered check also
validates redirect destinations/fragments, generated API entry, and retirement
from search. Review desktop/mobile navigation and saved example plots for
layout problems a structural check cannot identify.

Run `pytest examples_tests -q -Werror` when examples or public snippets change,
and `pytest scripts/tests -q -Werror` when infrastructure changes. Changing
release-example paths also requires `python scripts/run_release_tests.py --suite examples`.
Retain scientific assertions and tolerances. Keep missing CUDA measurements and
pending release approvals explicit.
