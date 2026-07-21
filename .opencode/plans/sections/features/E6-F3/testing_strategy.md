# Testing Strategy

Every implementation phase includes co-located fast tests in the same change.
The configured minimum 80% coverage is retained or improved; no threshold or
scientific tolerance may be weakened. Test modules use the `*_test.py` suffix.

## Per-Phase Approach

- **P1 transport primitives:** In `particula/gpu/properties/tests/` and
  `particula/gpu/dynamics/tests/`, compare each reused or new device function to
  an independent CPU implementation over representative radius, density,
  temperature, and pressure values. Include small-argument limits for Debye and
  coth-like terms, finite outputs, and explicit domain rejection.
- **P2 coefficient functions:** In
  `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`, compare spherical and
  rectangular fp64 coefficients against
  `get_*_wall_loss_coefficient_via_system_state`. Cover vectors, gravity/
  diffusion dominance, chamber dimensions, and recorded `rtol`/`atol`.
- **P3 contract/preflight:** In
  `particula/gpu/kernels/tests/wall_loss_test.py`, assert public signature,
  geometry exclusivity, scalar/per-box/environment forms, units, and import
  boundaries. Reject nonfinite/nonpositive values, malformed dimensions,
  unsupported distribution/configuration, wrong shape/rank/dtype/device, and
  inconsistent box dimensions before allocation, RNG mutation, or particle
  writes. Snapshot all caller-owned arrays and identities.
- **P4 removal:** In the same kernel test module, cover all-inactive, sparse
  gaps, one/multi-box, one/multi-species, exact `time_step == 0` no-op,
  controlled all-survive/all-remove cases, and complete clearing of mass,
  concentration, and charge. Assert density, volume, survivor values, shapes,
  devices, dtypes, and object/array identities are unchanged.
- **P5 persistent RNG:** Reproduce the coagulation lifecycle matrix: omitted
  convenience state; caller-owned initialize-once; repeated calls without
  reseeding; explicit reset; independent per-box advancement; supplied-buffer
  identity; and invalid-call non-initialization/non-advancement. A fixed seed
  must reproduce Warp runs after explicit reset, not CPU NumPy bitstreams.
- **P6 parity:** In
  `particula/gpu/kernels/tests/wall_loss_parity_test.py`, require Warp CPU for
  deterministic spherical/rectangular coefficient comparisons and repeated
  Bernoulli survival statistics. Compare observed survivor counts to
  `exp(-k*dt)` using predeclared binomial confidence/sigma bounds and enough
  independent particles/seeds to avoid flaky single-seed assertions. Run the
  same matrix on CUDA when available and skip cleanly otherwise.
- **P7 documentation:** Validate links, import snippets, SI units,
  support/deferred tables, and focused commands.

## Regression and Focused Commands

- Keep CPU tests in `particula/dynamics/wall_loss/**/` and
  `particula/dynamics/**/wall_loss*_test.py` green; E6-F3 must not alter CPU
  strategy semantics.
- Keep direct coagulation persistent-RNG and fixed-slot tests green because they
  define shared invariants.
- Focused commands should include:
  `pytest particula/gpu/dynamics/tests/wall_loss_funcs_test.py -q -Werror` and
  `pytest particula/gpu/kernels/tests/wall_loss_test.py
  particula/gpu/kernels/tests/wall_loss_parity_test.py -q -Werror`.
- Deterministic coefficient acceptance uses documented float64 tolerances.
  Stochastic acceptance uses expected distributions, never exact CPU/GPU draw
  order. Zero-time and inactive-slot no-ops remain exact.
