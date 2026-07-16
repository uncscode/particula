# Testing Strategy

Every implementation phase ships tests in the same change. Coverage thresholds
are never lowered; changed code maintains at least 80% coverage. GPU tests use
the repository's `*_test.py` convention and collect cleanly without Warp.

## Per-Phase Coverage

- **P1 -- physics helpers:** Probe Warp helpers in
  `particula/gpu/dynamics/tests/coagulation_funcs_test.py`. Compare kinematic
  viscosity and scalar ST1956 rates against independently evaluated CPU/NumPy
  equations across radius, temperature, dissipation, and density scales. Assert
  symmetry, cubic diameter-sum scaling, finite non-negative rates, and explicit
  fp64 tolerances.
- **P2 -- input contract:** In
  `particula/gpu/kernels/tests/coagulation_test.py`, cover scalar broadcast and
  direct `(n_boxes,)` Warp inputs with heterogeneous values. Reject missing,
  bool/integer/NumPy, zero, negative, NaN/Inf, wrong-shape, unsupported-dtype,
  and wrong-device inputs. Snapshot masses, concentration, charge, collision
  buffers/counts, and RNG state to prove fail-before-mutation behavior.
- **P3 -- majorant and execution:** Enumerate all active unordered pairs and
  prove every independent ST1956 rate is finite, non-negative, and no greater
  than the device majorant. Cover zero/one/two active slots, inactive gaps,
  equal/different radii, one-box and heterogeneous multi-box state, bounded
  trial scheduling, sorted/in-range/disjoint pairs, mass conservation, output
  buffer identity, and persistent RNG reuse/reset. Use repeated-run means or
  sigma bounds for stochastic rates, never exact CPU/Warp pair replay.
- **P4 -- documentation:** Validate links, direct import/API names, SI units,
  examples, support-table wording, and explicit no-DNS language.

## Device and Numerical Policy

- Warp CPU is required when Warp is installed and is the release baseline.
- CUDA is parametrized when available and skips cleanly otherwise.
- Deterministic fixtures use explicit `np.float64` values and declared
  scale-appropriate `rtol`/`atol`; conservation stays tight per box/species.
- Expected values come from the public CPU ST1956 function or direct NumPy
  equations and do not call the new Warp helper.
- No test imports or executes turbulent DNS implementations as evidence for
  this feature.

## Coverage Impact

Primary orchestration coverage remains colocated in
`particula/gpu/kernels/tests/coagulation_test.py`. A focused
`particula/gpu/kernels/tests/turbulent_shear_coagulation_test.py` may be created
if clarity requires it. Required correctness tests need no slow/performance
marker; E5-F5 makes no throughput claim.
