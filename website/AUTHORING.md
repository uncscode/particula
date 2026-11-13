Keep an ordinary Python script as the only maintained executable source.
Use `# %%` for code sections and `# %% [markdown]` for explanatory comments.
The source remains downloadable and executable with Python.

## Refresh results deliberately

From the repository root, in the development Python environment:

```bash
python scripts/docs_examples.py refresh distributions
python scripts/docs_examples.py refresh condensation
python scripts/docs_examples.py refresh gpu-coagulation
```

The CUDA example requires CUDA and never falls back to CPU. A refresh runs in
a fresh kernel with temporary working storage, captures figures and outputs,
and saves a complete successful generation. A failed run preserves previous
results. Review and commit the result bundle together with the script change.
Keep large raw data outside these publication bundles.

## Validate changes

```bash
python scripts/docs_examples.py check --changed --base upstream/main
```

Changed CPU examples execute for validation without replacing publication
results. Declared CUDA or long-running examples report an explicit execution
skip. Syntax, saved-file integrity, and freshness are checked for every example.
Use `--base` with the pull request base revision in CI.

The example registry declares its source, execution policy, device, timeout,
optional invocation, and input/helper files. Executable changes and declared
input bytes invalidate saved results; ordinary prose changes do not. Changes
inside Particula still need the existing scientific tests and a deliberate
result refresh when appropriate. The recorded library revision remains visible.

## Build without running simulations

```bash
npm --prefix website run build
npm --prefix website run preview
```

Pages are generated from current prose/code and saved result bundles. Code is
expandable; figures remain visible. Older results are labeled and grouped
separately from the updated code. Missing results show an awaiting-results note.
Generated Markdown, API dumps, and the built website are disposable outputs.

The current demo coexists with MkDocs. Existing notebook pairs remain available
to MkDocs until the complete migration is accepted; the Starlight build never
reads them. Internal execution uses temporary notebook objects to capture rich
outputs, without introducing a second maintained source file.
