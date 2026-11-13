# Particula documentation

Astro + Starlight is the sole documentation build. Markdown and percent-cell
Python sources live under `docs/`; builds consume saved results and never run
scientific workloads. Retired MkDocs files and notebooks are reference material
under `.trash/mkdocs-starlight-2026-11-13/`.

## Run locally

Use the tested Node **26.13.0** and Python **3.12** environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[extra]' -r website/requirements.txt
npm --prefix website ci
npm --prefix website run build
npm --prefix website run preview -- --port 4321
```

Open http://127.0.0.1:4321/particula/. The preview command manages a background
server; `npm --prefix website run preview -- status` and
`npm --prefix website run preview -- stop` inspect or stop it.

`npm --prefix website run dev` watches authored sources. Verify search with the
production build served by `preview`. The Python launcher uses the repository's
`.venv` when present; `PARTICULA_DOCS_PYTHON` selects another interpreter.

## Source and output ownership

- `docs/`: canonical narrative, homepage, scientific Python and shared assets.
- `website/navigation.json`: the complete five-section navigation.
- `website/examples.json`: all runnable sources, entry points, CPU/CUDA policy,
  timeouts and declared helper/data dependencies.
- `website/results/`: saved figures, tables, text, executed source and provenance.
  `current.json` selects the latest successful generation.
- `website/templates/`: generated API overview text.
- `website/src/content/docs/`, `public/results/`, `public/downloads/`,
  `public/images/`, `.cache/`, and `dist/`: disposable generated outputs.

Results use 300 DPI figures. Local user and checkout paths are redacted, while
relative filenames, line numbers and warnings remain. Progress bars are grouped
in closed disclosures. Failed refreshes preserve the previous successful bundle.

Fingerprints cover ordered executable cells, invocation, device, input bytes and
figure export settings. Prose-only changes do not invalidate results. The
package revision and source hash are recorded separately; a matching example
fingerprint does not claim execution against every later library revision.
See [authoring instructions](AUTHORING.md) and
[API diagnostics](API_DIAGNOSTICS.md).

## Validation and publishing

```bash
npm --prefix website test
.venv/bin/python scripts/check_docs_structure.py
.venv/bin/python scripts/docs_examples.py check --changed --base upstream/main
npm --prefix website run build
.venv/bin/python scripts/docs_report.py
```

CI validates every saved bundle and executes changed ordinary CPU examples
without replacing published results. CUDA and long-running examples require an
explicit local refresh; hosted CPU skips are reported honestly.

PRs, main pushes and merge-queue builds upload `documentation-validation`,
containing validation and execution reports, source/archive inventories and
rollback notes. PRs and merge-queue builds do not upload a site preview.
Production builds alone upload the verified site for the deployment job.

Only successful upstream main builds publish to the existing `gh-pages` branch.
PRs and merge-queue builds never publish production. See the archive README for
the recorded previous publication revision and rollback procedure.
