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

## Python source and saved results

Create percent-cell Python under `docs/examples/` in the owning topic. Use
`# %% [markdown]` cells for prose and equations and `# %%` for executable code.
Python is the maintained source; do not create a paired notebook.

Register the page in `website/examples.json`, including its entry point,
CPU/CUDA requirement, timeout, and every helper or data input. Historical result
viewers must declare their recorded input data rather than rerun benchmarks.

```bash
ruff check docs/examples/path/example.py --fix
ruff format docs/examples/path/example.py
.venv/bin/python scripts/docs_examples.py refresh EXAMPLE_KEY
npm --prefix website run build
```

Refresh executes the complete workload and saves figures, tables, text,
executed source and provenance under `website/results/`. Inspect those outputs
and commit the Python source and successful result bundle together. Failed runs
preserve prior results and block acceptance. Do not reduce scientific parameters
or substitute CPU execution for a required CUDA run.

Add a topic-index link and navigation entry in `website/navigation.json`.
Generated pages show results with collapsed code and a Python download.
Documentation builds render saved bundles without executing simulations.
CI validates all bundles and executes changed ordinary CPU examples without
rewriting published results. CUDA and long-running changes require an explicit
local refresh; hosted CPU skips do not establish successful GPU execution.

## Validation and evidence

Put example assertions in `examples_tests/*_test.py`. Check independent
scientific outcomes, per-species inventory where valid, units, held-state
identity and rejection behavior. Keep deterministic tolerances separate from
stochastic aggregate bounds. Describe open reservoirs, sink processes and
retained numerical defects honestly; execution alone is not scientific
qualification. Never relax tolerances to hide a migration discrepancy.

Run the explicit example and infrastructure suites as well as package tests;
default pytest collection covers only `particula`. Follow the repository's
coverage and lint gates. For a migration, record every source, result bundle,
public snippet and historical exception with literal dated execution evidence.
Historical API sketches must be labeled before a nonexecutable text fence.
Do not introduce historical facade APIs into supported examples.

Before creating a file, follow the [documentation placement rules](../../documentation.md):
extend an existing example when the learning objective is the same, otherwise
choose its owning topic and index. Keep Python stems in snake_case.
