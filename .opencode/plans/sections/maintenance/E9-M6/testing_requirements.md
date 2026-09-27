# Testing Requirements

## Policy and provenance

1. Test coverage thresholds must NEVER be lowered. Every modified function
   ships self-contained co-located tests in the same PR. Use `*_test.py` in
   adjacent `tests/`; integration tests belong in `particula/integration_tests/`.
2. Focused fix checks use coverage-disabled assertion evidence (`--no-cov`).
   Folder, file, node, name and marker selections are not coverage evidence.
3. Coverage comes only from the full applicable suite via untargeted
   `.opencode/tools/run_pytest.py`, with repository-configured full-package
   coverage and the normal threshold. Focused-target coverage is invalid
   evidence, not a fix failure. No local source/threshold override or arbitrary
   module target list, including borrowed historical resident coverage targets.
   The generic template's numerical gate does not override active runner policy.
4. Do not add local `-Werror`; use repository warning policy and intentional
   warning assertions. No test/lint/example execution is claimed by this draft.
5. Preserve dated final-revision commands, literal outputs, test counts,
   environment and device availability. Failures and unavailable required checks
   keep readiness blocked. Distinguish optional CUDA skips from Warp CPU results.

## Required assertion matrix

- Removed boundaries: root/package attributes and aliases, old concrete imports,
  obsolete builders/factories and compatibility dispatch. Negative tests must
  assert absence for the intended reason, not pass on an unrelated import error.
  Pair them with positive imports, native presets and real composed execution.
- Native state: exact held-object getters, accepted individual/coordinated
  replacement identity, rejected old/candidate state preservation, direct
  mutation revalidation, no automatic prepared GPU/resident rebind.
- Scientific behavior: M1-approved distribution normalization at V=0.25/1/4
  m^3, unequal weights, distinct ordered species and mixed/all-false partitioning;
  distinguish per-particle mass, population density and extensive inventory.
  Retain condensation uptake/evaporation/current-gas coupling, coagulation
  mechanisms/conservation, neutral/charged wall-loss sinks, all-gas dilution,
  nucleation conservation and per-attempted-substep failure semantics.
- CPU single-box admission, empty/zero-time/inactive cases, rejection before
  writes and documented partial-failure behavior; composition order unchanged.
- Transfers: ParticleData/GasData/EnvironmentData round trips, field shapes,
  dtype/device and identity ownership, ordered CPU gas names, partitioning
  bool/int32 conversion, GPU-only vapor pressure and explicit synchronization.
- GPU/direct/resident: supported export boundaries, caller-owned sidecars,
  persistent RNG, prepared identity drift, checkpoint/restart and capture
  lifecycle. No hidden transfer/fallback or new public resident exports.
- Independent analytical/NumPy oracles with explicit deterministic tolerances;
  tight per-species conservation separately from stochastic aggregate bounds.
  Do not weaken thresholds or demand exact cross-device RNG trajectories.

## Commands and phase gates

Run sequentially in the implementation worktree. P2/P3 add the proposed
`particula/tests/legacy_api_removal_test.py`; it must exist before its command
is accepted as evidence. Update concrete node lists from the implementation
diff, not an arbitrary coverage-target calculation.

```bash
pytest particula/tests/legacy_api_removal_test.py -q --no-cov
pytest particula/particles/tests/ particula/gas/tests/ particula/tests/ -q --no-cov
pytest particula/dynamics/condensation/ -q --no-cov
pytest particula/dynamics/coagulation/ -q --no-cov
pytest particula/dynamics/tests/ particula/dynamics/nucleation/ -q --no-cov
pytest particula/dynamics/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/integration_tests/ -q --no-cov
pytest particula/gpu/tests/conversion_test.py particula/gpu/tests/kernel_exports_test.py -q --no-cov
pytest particula/gpu/ -q --no-cov
pytest particula/execution/tests/ -q --no-cov
pytest particula/execution/tests/exports_test.py particula/tests/execution_exports_test.py -q --no-cov
pytest particula/tests/gpu_coagulation_docs_test.py particula/execution/tests/graph_capture_docs_test.py -q --no-cov
```

BOTH named wall-loss commands are mandatory and separate. Active
`pyproject.toml` excludes `particula/dynamics/wall_loss/tests/` from recursive
collection. Neither the ordinary suite nor full runner substitutes for this
explicit adjacent suite; separate calls avoid same-basename collection issues.

Warp CPU is the required installed-Warp baseline. Record runtime absence
explicitly; a permitted runtime skip is not executed GPU evidence. Optional
CUDA rows must pass or cleanly skip and never fall back to CPU/Warp CPU capture:

```bash
pytest particula/gpu/ particula/execution/tests/ -q -m "warp and cuda" --no-cov
```

## Examples, notebooks and final readiness

Revalidate every supported row in M5's inventory, not only a new quick start.
Include CPU dilution/nucleation, foundational/process/simulation/advanced
examples and maintained direct/resident workflows. Python entrypoints include:

```bash
python docs/Examples/data_containers_and_gpu_foundations.py
python docs/Examples/cpu_dilution.py
python docs/Examples/Nucleation/cpu_nucleation.py
python docs/Examples/gpu_direct_kernels_quick_start.py
python docs/Examples/gpu_complete_process_sequence.py
python docs/Examples/gpu_resident_session.py
python docs/Examples/gpu_resident_multi_timestep.py
```

Use each example's supported device/configuration and M5-recorded invocation;
native-CUDA-only `gpu_resident_graph_capture.py` remains qualified-device-only,
with unavailable CUDA recorded rather than CPU fallback. Additional scripts
and every supported pair come from M5's exhaustive inventory. Optional package
availability must not silently remove required CPU example rows.

For each affected notebook pair, substitute its concrete inventory paths in
this command pattern, record the literal resulting commands, and retain both
files. Edit the Python source, never notebook JSON directly:

```bash
ruff check <paired-source.py> --fix
ruff format <paired-source.py>
python3 .opencode/tools/validate_notebook.py <paired-notebook.ipynb> --sync
python3 .opencode/tools/run_notebook.py <paired-notebook.ipynb>
```

Run example/documentation contract tests alongside the executed sources and
repeat the old-API audit after outputs are regenerated. Finish at the final
revision, rerunning affected evidence after any fix:

```bash
.opencode/tools/run_linters.py
mypy particula/ --ignore-missing-imports
ruff check docs/Examples/
ruff format --check docs/Examples/
mkdocs build --strict
.opencode/tools/run_pytest.py
```

No benchmark/performance claim is required. If removal unexpectedly changes
scientific stepping, stop and reopen the owning upstream gate rather than
rebaseline results. No required unavailable command can be inferred successful;
the closeout remains BLOCKED until validated and reviewed.
