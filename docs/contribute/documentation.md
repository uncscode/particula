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

Implementation plans and task closeouts belong in `plans/`, not in
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

## Installed documentation bundles

`docs/` remains the only maintained source. Ordinary PEP 517 wheel builds use
`_particula_build.py` to stage the selected documentation under `particula/docs/`
in a temporary source tree, then delegate packaging to Flit. Builds do not
create a second maintained copy in the checkout. Editable installs resolve the
locator to this source tree instead of copying docs.

The bundle includes `index.md`, guides, theory, contributor guidance, Python
examples, and supporting data/images with their original relative paths and
bytes. Generated API pages, hidden directories, caches, build output, temporary
files, website output and `.trash/` are excluded. The generated manifest records
the package version and each source file's size and SHA-256 hash. Locate it with
`python -m particula_docs --json`; `particula-docs` prints only the directory.
Relative links within the selected trees are preserved. Repository tooling and
rendered-site links are not made into standalone offline pages.

Build and verify release artifacts using the selected development interpreter:

```bash
python -m build
python scripts/check_doc_bundle.py \
  --wheel dist/particula-0.3.0-py3-none-any.whl \
  --sdist dist/particula-0.3.0.tar.gz \
  --report .artifacts/docs-bundle.json
pytest scripts/tests/doc_bundle_test.py -q -Werror
```

Use the filenames for the version being released. The checker verifies source
hashes and wheel RECORD entries, rebuilds from the source archive with no Git or
network access, and installs both wheels in separate clean environments. It
checks searchable guides, examples, assets, relative links and the import-free
locator. The report records version, documentation size, artifact sizes and
hashes. PyPI publication requires these checks on the exact upload artifacts.
The regression suite additionally exercises ordinary `pip wheel`, editable
installs, excluded-content sentinels and rejection of corrupted bundles.

The source archive contains the build adapter, locator and selected docs;
rebuilding offline needs only Python and the declared Flit build dependency
already installed. The verification tools additionally require `build` (and
`pytest`/PyYAML for the regression suite), included in the development extra.
The local conda recipe stages these same documentation sources and build files.
