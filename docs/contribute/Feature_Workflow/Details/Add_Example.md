# Add Examples

Provide a runnable demonstration with a clear physical purpose, units and
observable outcomes. Keep small tutorials quick; preserve the published
workload of scientific simulations and record their longer runtime explicitly.

## Native construction and process ownership

Install the repository checkout (`python -m pip install -e ".[extra]"`).
Construct native `Aerosol` from `ParticleData`, `GasData`
and `EnvironmentData`; use process-owned physics and explicit species maps.
Start with [native state access and replacement](../../../guide/data-model/ownership.md)
and [scientific process contracts](../../../guide/processes/index.md).

Declare the distribution type and representative volume. Resolved concentration
stores counts; discrete concentration stores number density; continuous PDFs
are radius densities dN/dr. Label plotted units and compute inventory using the
appropriate count/volume or quadrature conversion. Preserve ordered gas names,
unmapped inert gases and particle-material versus full-gas widths.

Use approved package exports. Required direct-GPU configuration and sidecars
may come from documented concrete modules; keep those ownership boundaries
explicit. Callers own transfers, synchronization and device selection. Optional
CUDA skips do not establish capture, performance or cross-device equivalence.

## Source and notebook workflow

Create source files under `docs/examples/` in the owning topic. Standalone
scripts are supported; notebook examples use paired `.py` percent-format and
`.ipynb` files. Edit the Python source first:

```bash
ruff check docs/examples/path/example.py --fix
ruff format docs/examples/path/example.py
python .opencode/tools/validate_notebook.py docs/examples/path/example.ipynb --sync
python .opencode/tools/run_notebook.py docs/examples/path/example.ipynb
```

Execute the standalone source too, inspect plots and retained notebook outputs,
and commit both files. Pure prose/formatting synchronization may retain prior
outputs only after verifying every code cell has the same AST. A changed
calculation requires execution of the complete published workload. Bounded test
fixtures supplement that execution; they do not replace it.

Add a topic-index link. Relative notebook links render in the full publication
build. Python files are excluded from the site, so link their repository source
when a reader needs the standalone entrypoint. `mkdocs-full.yml` renders saved
notebook outputs and generates API pages; the faster base build does neither.

## Validation and evidence

Put example assertions in `examples_tests/*_test.py`. Check independent
scientific outcomes, per-species inventory where valid, units, held-state
identity and rejection behavior. Keep deterministic tolerances separate from
stochastic aggregate bounds. Describe open reservoirs, sink processes and
retained numerical defects honestly; execution alone is not scientific
qualification. Never relax tolerances to hide a migration discrepancy.

Run the explicit example and infrastructure suites as well as package tests;
default pytest collection covers only `particula`. Follow the repository's
coverage and lint gates. For a migration, record every source, notebook,
public snippet and historical exception with literal dated execution evidence.
Historical API sketches must be labeled before a nonexecutable text fence.
Do not introduce historical facade APIs into supported examples.

Before creating a file, follow the [documentation placement rules](../../documentation.md):
extend an existing example when the learning objective is the same, otherwise
choose its owning topic and index. Keep Python/notebook stems in snake_case.
