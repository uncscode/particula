# AGENTS.md — Particula development

Particula is a Python aerosol particle simulator. This checkout targets
**0.3.0**, requires **Python 3.12+**, and uses NumPy, SciPy, and NVIDIA Warp.
The upstream repository is https://github.com/uncscode/particula; local forks
may use a different `origin`. `particula/__init__.py` owns the package version,
which Flit reads dynamically. A version bump does not publish a release.

## Working in this repository

- Use the repository's existing environment when available (`.venv/bin/python`,
  `.venv/bin/pytest`, `.venv/bin/ruff`). Development installation is
  `python -m pip install -e '.[dev,extra]'`.
- Preserve unrelated work. Keep changes scoped to the requested behavior and
  inspect the final diff before handing work back.
- Use the direct test, lint, and documentation commands below. Legacy agent
  workflow files are historical context, not the current development process.
- Read the relevant maintained guide and implementation before changing a
  scientific or lifecycle contract. Historical plans and recorded results are
  not evidence of current behavior.

### CodeGraph

In an indexed repository (`.codegraph/` exists), use CodeGraph before text
search or file reads to locate or understand code. Load the `codegraph` skill
at `~/.codex/skills/codegraph/SKILL.md` for setup and navigation. Prefer
`codegraph_explore` with this repository's path and a concrete file or symbol;
`codegraph explore` is the CLI fallback. Reuse existing indexes. Initialize an
unindexed repository when needed and keep `.codegraph/` ignored by Git.
Use ordinary search for documentation, configuration, or unavailable coverage.

## Layout and source of truth

| Path | Purpose |
| --- | --- |
| `particula/particles/` | Native particle data, distributions, slots and properties |
| `particula/gas/` | Gas/environment data and thermodynamic strategies |
| `particula/dynamics/` | Coagulation, condensation, dilution, nucleation and wall loss |
| `particula/activity/`, `particula/equilibria/` | Activity and partitioning |
| `particula/execution/` | Execution metadata and concrete resident lifecycle |
| `particula/gpu/` | Warp containers, explicit transfers and direct kernels |
| `particula/util/` | Constants, units and input validation |
| `examples_tests/` | Tests that execute documentation examples |
| `scripts/`, `scripts/tests/` | Documentation and release tools and their tests |
| `docs/` | Maintained guides, scientific examples and validation records |
| `website/` | Astro/Starlight site, example registry and saved results |
| `conda/` | Recipe mirror and release-validation instructions |
| `pyproject.toml` | Dependencies, packaging, pytest, Ruff and mypy configuration |

Start with [contributing](docs/contribute/CONTRIBUTING.md),
[code specifications](docs/contribute/Code_Specifications/index.md), and
[architecture](docs/contribute/architecture/index.md). The
[roadmap](docs/contribute/roadmap.md) tracks pending work. Archived material in
`.trash/` is reference-only and excluded from source distributions.

## Validation

Run focused checks while editing, then the relevant source suites. The CI
commands are:

```bash
pytest -Werror -m "not slow and not performance"
pytest examples_tests -Werror -m "not slow and not performance"
pytest scripts/tests -Werror
ruff check particula/ examples_tests/ scripts/
ruff format particula/ examples_tests/ scripts/ --check
mypy particula/ --ignore-missing-imports
```

Plain `pytest` uses `testpaths = particula`; it does not collect the example or
infrastructure suites. Tests use the `*_test.py` suffix. The adjacent
`particula/dynamics/wall_loss/tests/` directory is excluded from normal recursive
collection; run it explicitly when changing that implementation.

- Test behavior, independent numerical oracles, conservation and failure
  atomicity. Avoid assertions that merely freeze prose or source layout.
- Keep deterministic tolerances explicit, with `np.float64` fixtures for
  precision work. Never weaken conservation or coverage thresholds to pass.
- Use aggregate statistics or sigma bounds for stochastic checks; do not
  promise exact CPU/Warp/CUDA per-seed trajectories.
- Warp CPU is the baseline when Warp is installed. CUDA tests are optional
  locally and must skip cleanly when unavailable. A skip is not CUDA evidence.
- Registered markers are `slow`, `performance`, `benchmark`, `warp`, `cuda`,
  `gpu_parity`, and `stochastic`. Only `--benchmark` changes collection.
- Full-package coverage can be measured with
  `pytest --cov=particula --cov-report=term-missing --cov-fail-under=80`.
  Focused tests alone do not establish full-package coverage.

### Release checks

```bash
python scripts/run_release_tests.py
python scripts/run_release_tests.py --suite examples
python scripts/conda_feedstock.py
```

The first two commands build fresh wheels and test isolated installations;
`conda_feedstock.py` additionally requires conda-build. See
[conda release validation](conda/README.md) for the exact staging contract and
external feedstock handoff. The release runner retains its CPU-focused
allowlists; a version bump does not broaden coverage or approve those exclusions
for a new release. Source CI still exercises GPU examples and `warp-lang`
remains a runtime dependency.

An advisory feedstock comparison is not strict parity or external build
success. External release readiness requires a passing strict contract check
and an actual external feedstock build. Record unavailable checks honestly.
The [0.3.0 validation record](docs/contribute/validation/releases/v0.3.0.md)
separates implementation evidence from maintainer review and release approval.

## Code style

- Four spaces, double quotes, 80-character lines; Ruff handles formatting.
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants.
- Imports: standard library, third-party, then local.
- Type hints for public APIs; Google-style docstrings with units, shapes,
  return values and relevant errors. Cite sources for scientific expressions.
- Prefer NumPy operations where appropriate. Use existing validation helpers
  such as `validate_inputs` for positive, nonnegative and finite inputs.
- Do not widen public exports casually. Concrete-module-only helpers and
  configuration records are intentional API boundaries.

## Native data and scientific contracts

Use the [data model](docs/guide/data-model/index.md),
[ownership guide](docs/guide/data-model/ownership.md), and
[0.3.0 migration guide](docs/guide/migration/v0.3.0/index.md) as the maintained
references for field shapes, units and supported construction.

- Construct native `Aerosol` from `ParticleData`, full ordered `GasData`, and
  `EnvironmentData`. Legacy `ParticleRepresentation`, `GasSpecies`,
  `Atmosphere`, and their obsolete builders/bridges have been removed.
- Population semantics must be explicit: resolved concentration is represented
  count, discrete concentration is number density, continuous PDF is dN/dr.
  Use representative volume when comparing density or extensive inventory.
- Particle-material width and full-gas width are independent. Preserve ordered
  species names, inert/nonpartitioning lanes and explicit process species maps.
  Physics configuration belongs to processes.
- CPU runnables remain single-box. Batched container storage does not establish
  general multi-box CPU process support.
- `Aerosol` holds its containers by identity. Use validated individual or
  coordinated replacement. Raw array edits do not provide transactional
  validation. Rebuild process maps and GPU bindings after layout replacement;
  never imply automatic rebinding.
- Preserve fixed-capacity slot semantics: active slots have finite positive
  concentration, finite nonnegative masses with positive total mass and finite
  charge; free slots have exactly zero concentration, all masses and charge.
  Other states reject. Do not silently resize, compact or truncate demand.
- Preserve the fixed-composition interpretation of binned coagulation. Do not
  present it as a general radius-varying, multispecies composition model.
- Preserve documented process-specific preflight and mutation boundaries.
  Sequential substeps may retain earlier successful updates if a later step
  fails; do not claim whole-call rollback without an explicit implementation.

See the [process guides](docs/guide/processes/index.md) for condensation,
coagulation, dilution, nucleation and wall loss. In particular, direct GPU
condensation uses four equal substeps with device vapor-pressure refresh and
inventory-limited gas-coupled transfer; this does not establish CPU-strategy
parity or adaptive stepping. Charged wall-loss image enhancement remains
active at zero wall potential.

## GPU and resident lifecycle

Read [supported GPU paths](docs/guide/gpu/support.md),
[transfers](docs/guide/gpu/transfers.md),
[slot exhaustion](docs/guide/gpu/slot-exhaustion.md),
[checkpoints](docs/guide/gpu/resident/checkpoints.md), and
[graph capture](docs/guide/gpu/resident/graph-capture.md) before changing these
boundaries.

- Direct step entry points import from `particula.gpu.kernels`; conversion
  helpers import from `particula.gpu`. Sidecars and lifecycle APIs documented
  as concrete-only stay out of package and top-level exports.
- Callers own device placement, stable-shape buffers, CPU/Warp transfers and
  synchronization. Do not introduce hidden transfers or CPU fallback.
- Gas names remain CPU-owned ordered metadata. Preserve them externally for
  semantic restores. Device vapor pressure is derived state and is dropped by
  CPU gas inspection/restore helpers.
- Direct coagulation/wall-loss RNG sidecars persist when supplied; repeating a
  seed does not reset them. Only explicit initialization/reset reseeds.
- Resident setup pins exact session, registry, guard and resource identities.
  Normal stepping does not checkpoint, restart, reseed or rebuild bindings.
- Schema-v5 checkpoints use immutable canonical device bytes and RNG
  continuation metadata. CPU inspection carriers are lossy. Restart is
  explicit, creates fresh identities, and requires an exactly compatible
  device; there is no inferred migration or cross-device continuation.
- Preserve read-only preflight rejection guarantees. Once an asynchronous
  writer launches, do not promise rollback. Faulted lifecycle state requires
  its documented explicit handling.
- Native graph capture is CUDA-only. Preserve qualification, admission,
  invalidation, retirement and teardown boundaries; no Warp-CPU emulation,
  automatic recapture, or automatic recovery is implied.

### Benchmark and profiling evidence

Benchmarks are opt-in (`pytest particula/gpu/tests/benchmark_test.py
--benchmark -v -s`). Historical coagulation artifacts are not current-version
or resident-capture evidence. The controlled coagulation output is
`.artifacts/benchmarks/gpu_benchmark_results.json`; the documentation viewer
uses the declared input `docs/data/gpu_benchmark_results.json`.

Resident benchmark evidence belongs at
`.artifacts/benchmarks/resident_capture_comparison.json`. The
[qualification record](docs/contribute/validation/gpu/graph-capture-qualification.md)
is the evidence authority: measured evidence remains `UNSHIPPED/BLOCKED`
until its final-revision, designated-device gates pass. Consult the
[resident benchmark record](docs/contribute/validation/gpu/resident-benchmarks.md),
[profiling record](docs/contribute/validation/gpu/graph-capture-profiling.md), and
[mass-precision study](docs/contribute/validation/numerics/mass-precision.md).
Never turn missing measurements, timing ratios or profiler overhead into
performance/parity claims.

## Documentation and examples

Astro and Starlight are the documentation build. See
[website development](website/README.md), [authoring](website/AUTHORING.md), and
[example contribution](docs/contribute/Feature_Workflow/Details/Add_Example.md).

- Maintain percent-cell Python sources in `docs/examples/` and `docs/theory/`.
  Do not maintain paired `.ipynb` sources. Jupytext remains a parser/execution
  dependency; removing notebook-pairing configuration does not remove that use.
- Declare source, entry point, CPU/CUDA device, timeout and helper/data inputs
  in `website/examples.json`.
- Execute changed scientific workloads in full, inspect figures/tables/text,
  and commit the sources with their saved `website/results/` bundles.
  Prose-only changes may retain current results.
- Failed refreshes preserve old results and block acceptance. Do not substitute
  CPU for required CUDA or weaken scientific parameters/assertions to pass.
- Builds render saved results without running simulations. Their provenance is
  revision-specific; a matching example fingerprint does not establish that it
  was rerun against every later library change.

```bash
python scripts/docs_examples.py refresh EXAMPLE_KEY
python scripts/docs_examples.py check
python scripts/check_docs_structure.py
npm --prefix website test
npm --prefix website run build
```

**Last updated:** 2026-11-13
