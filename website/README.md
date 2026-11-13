# Particula · local Starlight demo

This branch evaluates Astro + Starlight against the reorganized v0.3.0 docs.
The existing MkDocs publication and notebooks remain intact. Starlight uses
the Python scripts and saved result bundles; it never reads `.ipynb` files.

## Run locally

Use Node 22.23.4 or newer and Python 3.12 or newer. From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[extra]' -r website/requirements.txt
npm --prefix website ci
npm --prefix website run build
npm --prefix website run preview -- --port 4321
```

Open **http://127.0.0.1:4321/particula/**. Astro 7's preview command manages
a background server; use `npm --prefix website run preview -- status` and
`npm --prefix website run preview -- stop` to inspect or stop it.

`npm --prefix website run dev` watches the authored sources and regenerates
the preview without running simulations. Search is verified in the production
build served by `preview`. The Node launcher uses the repository's `.venv`
when present; set `PARTICULA_DOCS_PYTHON` to select another interpreter.

## Review tour

1. Home: existing branding, five reader paths, light/dark and mobile layouts.
2. Guide: first simulation, native construction, and CPU/GPU support.
3. Examples: distributions and condensation with saved CPU figures; a real
   CUDA coagulation run with device and revision metadata.
4. Saved-result states: labeled presentation fixtures explaining matching,
   stale, and missing results. Stale-result behavior also has automated tests.
5. Theory: activity equations and a Mermaid process diagram.
6. API: Griffe-generated module coverage, linked signatures, source links,
   symbol anchors, and Pagefind search.
7. Contribute: authoring and explicit refresh instructions.

Links marked ↗ lead to published documentation outside the demo subset.
New illustrative content is limited to the site layout and workflow fixtures;
scientific content and production code retain their existing ownership.

## Source and output ownership

- `docs/`: canonical narrative and scientific Python sources.
- `website/examples.json`: selected examples, execution policy, declared
  input/helper files, timeout, and optional explicit invocation.
- `website/results/`: committed figures, executed sources, and provenance.
  `current.json` selects the latest successful generation. Earlier generations
  can be retained as history; ordinary validation never changes the pointer.
- `website/templates/`: demo-specific landing, API overview, and workflow text.
- `website/src/content/docs/`, `public/results/`, `public/downloads/`,
  `public/images/`, `.cache/`, and `dist/`: ignored generated outputs.

Execution fingerprints include the ordered code sections and declared input
bytes. Prose and formatting changes do not invalidate output; changing code
section boundaries does, because outputs are associated with those sections.
Particula revision and environment metadata are recorded separately. A source
fingerprint does not prove that results were rerun against newer library code.
The CUDA policy skips execution during ordinary checks, even on a GPU machine.
An explicit refresh requires CUDA and never substitutes Warp CPU.

See [authoring instructions](AUTHORING.md) for refresh and validation commands.

## Checks and review artifacts

```bash
npm --prefix website test
.venv/bin/python scripts/docs_examples.py check --changed --base upstream/main
npm --prefix website run build
```

The build checks rendered local links/assets/fragments and all Griffe module
pages. It reuses the existing relocation inventory for routes included in the
demo and leaves redirects outside the search index. Reports live in
`website/.cache/verification.json`; Griffe diagnostics remain in
`website/.cache/griffe.log`. Browser screenshots and the validation record are
kept under ignored `website/qa/` and the tracked `website/VALIDATION.md`.

## Next stage after local review

Migrate remaining sources and notebook-only theory; replace notebook-pair CI
with script/result checks; carry over all redirects and publication features;
then switch GitHub Pages and PR previews. Do not remove MkDocs or notebook
sources as part of this representative local demo. Full versioned docs,
interactive simulations, and browser Python execution remain separate work.
