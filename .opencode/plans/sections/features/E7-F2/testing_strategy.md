# Testing Strategy

## Co-Located Phase Coverage

Every production phase ships its tests in the same change. No standalone unit
testing phase is deferred. Place neutral execution tests under
`particula/tests/` and GPU adapter/contract tests in the appropriate
`particula/gpu/**/tests/` directory using the `*_test.py` convention.

## Unit and Contract Matrix

- Capability matrix: supported CPU, Warp CPU, optional CUDA, isothermal,
  latent-heat, and representable activity/surface modes.
- Negative matrix: staggered GPU, unsupported BAT mapping, missing Warp/CUDA,
  malformed state/configuration, wrong shape/dtype/device, aliases, non-finite
  time, mixed environment sources, and missing thermodynamics.
- Dispatch: exact adapter, arguments, call count, result identity, exception
  propagation, and no implicit fallback.
- Ownership: fixed shapes and identities for particle/gas/environment,
  transfer/scratch/thermal sidecars, and output buffers across repeated calls.
- Atomicity: selection and pre-launch rejection leave physical state and
  supplied outputs unchanged; later substep failure follows the existing
  documented partial-commit boundary.

## Shipped P1 Coverage

`particula/tests/execution_test.py` now covers the pure metadata boundary:
the 36 CPU and 8 Warp-profile declarations, exact four-member requirements,
supported and rejected Warp configurations, type and backend-first validation,
immutable/read-only behaviour, non-composable requirement semantics, and
fresh-process isolation from `warp` and `particula.gpu`. These tests intentionally
do not claim runtime availability, native-device, adapter, or GPU execution
coverage; those remain later-phase responsibilities.

## Parity and Scientific Validation

Use independent CPU/NumPy references rather than comparing the Warp adapter to
itself. Cover one-box and multi-box, uptake, evaporation, disabled partitioning,
zero gas, inactive slots, multiple species, constant/Buck thermodynamics,
isothermal, and latent-heat cases. Compare particle masses and gas
concentrations separately with recorded `rtol`/`atol`; verify per-box,
per-species concentration-weighted particle-plus-gas conservation, using the
existing tight conservation target (`rtol=1e-12`, `atol=1e-30`) where the
fixture supports it. Algorithmic differences use justified parity tolerances,
not a false bitwise-equality promise.

## Transfer and Compatibility Assertions

Spy on conversion, restore, synchronization, and CPU runnable calls. A normal
Warp adapter step performs no bulk transfer or hidden synchronization. Existing
direct CPU/GPU imports and behavior remain unchanged; concrete scratch and
thermodynamic internals are not accidentally broadened through exports.

## Commands and Gates

- Focused new tests with `pytest ... -q -Werror`.
- Existing condensation kernel, thermodynamics, process-sequence, and export
  regressions.
- At least 80% coverage for changed modules without lowering configuration.
- `ruff check`, `ruff format --check`, and `mypy` for changed source.
- `mkdocs build --strict` and documentation contract tests in P6.
- Warp CPU must pass when Warp is installed; CUDA rows skip cleanly if absent.
