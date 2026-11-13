# Example validation guide

Published examples use percent-cell Python and saved execution bundles. See
[documentation guide](documentation_guide.md) and
[Add Examples](../../docs/contribute/Feature_Workflow/Details/Add_Example.md).

```bash
.venv/bin/python scripts/docs_examples.py refresh EXAMPLE_KEY
.venv/bin/python scripts/docs_examples.py check --changed --base upstream/main
npm --prefix website run build
```

The former notebook-pairing guide is archived under
`.trash/mkdocs-starlight-2026-11-13/`. Generic notebook tools may still be used
for unpublished exploratory work, but they are not the documentation workflow.
