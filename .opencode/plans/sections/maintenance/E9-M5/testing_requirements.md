# Testing Requirements

## Policy

Every phase includes self-contained co-located regression tests, committed with
changed functions/examples. Use `*_test.py` in owning module-level `tests/`
directories; existing documentation suites stay at their established locations.
Never lower coverage thresholds or remove enduring scientific assertions.
Focused file/folder/node/marker/name-filter checks use direct `pytest --no-cov`
as assertion evidence only. Focused-target coverage is invalid evidence, not a
fix failure. Comprehensive coverage comes only from the full applicable suite
using `.opencode/tools/run_pytest.py` without a target, with repository-configured
full-package scope and normal threshold. Do not override coverage source or
threshold. The generic template's numeric/pyproject coverage statement is not
authority over active repository testing policy. Do not add local `-Werror`.

## Per-phase assertions and scientific outcomes

P1 freezes the exact test/entrypoint list alongside inventory rows. Use existing
docs/example suites where possible; add adjacent tests for changed functions
and advanced snippets. Assert numerical outcomes independently, not just source
strings or absence of exceptions. Cover non-unit volumes, mixed gas categories,
ordered mapping, concentration-weighted inventories, identity and replacement
rejection. Preserve distribution semantics and M3/M4 failure contracts.
Use explicit deterministic tolerances, separate tight conservation checks and
aggregate stochastic criteria; no exact cross-device random trajectory claims.

Representative existing focused groups, extended by the approved inventory:

```bash
pytest particula/gpu/tests/data_containers_example_test.py -q --no-cov
pytest particula/tests/nucleation_docs_test.py -q --no-cov
pytest particula/dynamics/tests/wall_loss_strategies_test.py particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/gpu/tests/gpu_direct_kernels_example_test.py particula/gpu/tests/gpu_complete_process_sequence_example_test.py -q --no-cov
pytest particula/execution/tests/gpu_resident_session_docs_test.py particula/tests/gpu_resident_multi_timestep_docs_test.py -q --no-cov
pytest particula/gpu/tests/conversion_test.py particula/gpu/tests/kernel_exports_test.py -q --no-cov
```

The adjacent CPU wall-loss directory is excluded from recursive collection in
`pyproject.toml`; its explicit invocation must not disappear into a full-suite
claim. Warp CPU is the installed-Warp baseline for uncaptured supported paths;
optional CUDA passes or cleanly skips. Capture remains native-CUDA-only, without
Warp-CPU/CPU emulation. Preserve hardware-free docs checks when Warp is absent.

## Example and notebook execution

Execute every inventoried supported standalone script via `python <source.py>`
with its documented arguments and environment. For example:

```bash
python docs/Examples/cpu_dilution.py
python docs/Examples/Nucleation/cpu_nucleation.py
```

For each affected notebook pair use the prescribed sequence (substitute its
actual inventory paths), then inspect outputs and commit both files:

```bash
ruff check docs/Examples/path/to/file.py --fix
ruff format docs/Examples/path/to/file.py
python3 .opencode/tools/validate_notebook.py docs/Examples/path/to/file.ipynb --sync
python3 .opencode/tools/run_notebook.py docs/Examples/path/to/file.ipynb
```

Execution must cover the published workload; bounded regression fixtures are
additional evidence, not permission to skip heavy supported notebooks. Record
required optional-package/resource needs explicitly. Never replace unavailable
execution with stale notebook output or a claim that strict rendering ran code.

## Final gate

After focused assertions and all required example/pair executions pass:

```bash
.opencode/tools/run_pytest.py
.opencode/tools/run_linters.py
mypy particula/ --ignore-missing-imports
mkdocs build --strict
```

Also run Ruff on changed example Python sources and review Markdown links/API
directives and historical-code exclusions. Record revision, date, dependencies,
device availability, literal output and pass/fail/skip/unavailable per command.
Required unavailable evidence blocks M6 handoff; accepted optional device skips
do not prove performance or relax scientific gates. No tests are run as part
of this plan-only drafting task.
