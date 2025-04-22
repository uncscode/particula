# Add Examples

## Goal

Provide a runnable demonstration in a Jupyter notebook.

## Rules

- Runtime <30 s on a typical laptop (in most cases).
- Use only the public API – no private helpers.
  - `import particula as par`
- End with at least one plot or printed result.
- Contextualize the example with a short description of what it does and why it's useful.

## Steps

1. Create a new issue on GitHub and assign it to yourself.
1. Create a branch on your forked repo for this issue.
2. Create `docs/examples/**/<feature_name>.ipynb`.
   1. Add a new folder if needed, but please keep it organized.
   2. These can also be under `docs/Theory/**/<feature_name>.ipynb` to validate the implementation of the theory.
3. Interleave Markdown explanations with code cells.
4. Verify it runs top‑to‑bottom.
5. Commit the .ipynb file in your PR branch.
  1. The website builder will automatically convert it to markdown and HTML during PR.
6. Create your pull‑request description.
