# Particula validation

Commands run from the selected checkout using its interpreter and configuration.
Use `.venv/bin/python -m pytest` or `.venv/bin/ruff` when those are the selected
tools. Consult current `pyproject.toml` and `.github/workflows/` for drift.

## Choose the suite

| Changed behavior | Checks |
| --- | --- |
| Numerical function or CPU process | Adjacent `tests/*_test.py`; independent physical oracle and edge cases |
| Direct Warp kernel/transfer | Kernel/property/conversion tests, Warp CPU baseline and optional CUDA |
| Resident lifecycle | Affected execution tests, identities, failure state, checkpoint/RNG continuity |
| Published example or README snippet | Matching `examples_tests/` modules; full changed workload and outputs |
| Release/docs/plugin tooling | Affected `scripts/tests/` plus the relevant build/check command |
| Concrete wall loss | Both `particula/dynamics/tests/wall_loss_strategies_test.py` and the adjacent wall-loss suite |

Tests use the `*_test.py` suffix in colocated `tests/` folders. The default
`testpaths = particula` omits `examples_tests/` and `scripts/tests/`.
`particula/dynamics/wall_loss/tests/` is excluded from recursive collection;
invoke it explicitly when changing that implementation. Do not rely on an old
fixed test count to establish coverage.

Current source-CI commands:

```bash
pytest -Werror -m "not slow and not performance"
pytest examples_tests -Werror -m "not slow and not performance"
pytest scripts/tests -Werror
ruff check particula/ examples_tests/ scripts/ --no-fix
ruff format particula/ examples_tests/ scripts/ --check
mypy particula/ --ignore-missing-imports
```

For development, fix/format only intended files, inspect the diff, and rerun
checks. `pyproject.toml` enables Ruff fixes by default, so use `--no-fix` during
a read-only review. The former wrapper-specific prohibition on `-Werror` does
not apply to direct pytest. Prefer the CI warning policy; fix unexpected
warnings, use `pytest.warns` for intentional warnings, and keep necessary
third-party filters narrow. Never broadly suppress warnings to pass.

Focused checks normally use `--no-cov`. Full-package coverage is:

```bash
pytest --cov=particula --cov-report=term-missing --cov-fail-under=80
```

A filtered test run cannot establish full-package coverage. For a required
changed-module gate, derive targets from the executable diff and keep per-target
rows plus the aggregate gate. Pass repeated `--cov=package.module` options;
a comma-joined list is not a list of separate pytest-cov targets. Keep the
normal 80% gate unless the owning contract requires a stronger threshold.
Prose and historical plans are not executable coverage targets.

## Scientific assertions

Use independent analytical/NumPy expectations rather than the production helper
under test. Compare each meaningful output separately; a large gas reservoir
must not hide particle-mass drift. Include relevant shapes, scalars/arrays,
zero/empty/inactive cases, invalid values, no-ops, aliasing and rejection before
mutation. Assert identities and untouched fields where the API promises them.
Do not add production instrumentation solely to expose test internals.

Keep three criteria separate:

1. Deterministic parity: explicit `assert_allclose` relative/absolute tolerances,
   with `np.float64` fixtures for precision baselines.
2. Conservation: independently check concentration/volume-normalized inventories
   using the existing tight process-specific tolerance.
3. Stochastic behavior: aggregate distributions or sigma bounds over repeated
   samples; do not demand exact cross-device per-seed trajectories or relax
   conservation because sampling is stochastic.

Warp CPU is required when Warp is installed. CUDA is optional locally and skips
cleanly if unavailable; record it as unavailable evidence. Use fixture/test-local
Warp guards when a module also contains CPU-only tests. Markers describe intent
but do not select a device. `--benchmark` is the only collection-affecting
custom option; benchmark-marked tests are opt-in. `slow` and `performance`
identify other expensive workloads. Do not include timings in ordinary checks.

GPU closeouts use the command matrix in the owning validation record. Native
capture/profiling needs qualified CUDA and cannot fall back to Warp CPU. Record
revision, devices, literal commands, passes/skips/failures and missing evidence.
Benchmark ratios and a successful import do not establish physical parity.

## Documentation and plugin checks

```bash
python scripts/check_docs_structure.py
python scripts/docs_examples.py check
npm --prefix website test
npm --prefix website run build
python scripts/build_particula_plugins.py --check
pytest scripts/tests/plugin_distribution_test.py -q -Werror
```

Run the relevant subset; documentation builds render saved outputs without
executing simulations. Follow `website/AUTHORING.md` for full-workload refresh
and visual inspection when scientific code changes.

## Release checks

```bash
python scripts/run_release_tests.py
python scripts/run_release_tests.py --suite examples
python scripts/conda_feedstock.py
```

The wheel runner uses separate package and allowlisted CPU-example environments;
its exclusions do not narrow source CI or make Warp an optional dependency.
The conda command requires conda-build. Use `conda/README.md` for exact staging,
external handoff, strict parity and real external-build requirements. Advisory
drift-check success is not external release readiness. A version bump does not
approve existing exclusions or count as a published release.
