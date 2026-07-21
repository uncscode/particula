# Phase Details

- [ ] **E6-F4-P1:** Freeze charged wall-loss configuration and semantics with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Extend the E6-F3 contract with wall potential and scalar/vector field inputs while explicitly freezing neutral fallback, units, support, and atomic preflight behavior.
  - Files: `particula/gpu/kernels/wall_loss.py`, `particula/gpu/kernels/tests/wall_loss_test.py`
  - Tests: Accepted geometry-specific forms; finite potential/field validation; wrong rank, length, dtype, or device; unsupported mode; complete particle/RNG immutability on failure.

- [ ] **E6-F4-P2:** Implement image-charge enhancement device primitives with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Port the CPU image-charge factor in fp64, including zero-potential enhancement, clipping, and exact per-slot factor one for zero charge.
  - Files: `particula/gpu/dynamics/wall_loss_funcs.py`, `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`
  - Tests: Positive/negative/zero charge, representative radii and temperatures, clip-domain extremes, finite output, and CPU array-oracle parity.

- [ ] **E6-F4-P3:** Implement electric-field drift and charged coefficient composition with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Match CPU field magnitude, potential scaling, signed mobility drift, and nonnegative finite composition around E6-F3's neutral coefficient.
  - Files: `particula/gpu/dynamics/wall_loss_funcs.py`, `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`
  - Tests: Spherical scalar and rectangular vector fields, potential-only drift, charge sign, zero field, zero charge, geometry scale, clipping, and fp64 CPU parity.

- [ ] **E6-F4-P4:** Integrate charged mode into the neutral fixed-shape wall-loss step with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Select neutral or charged coefficient calculation inside the E6-F3 step without changing active-slot, removal-clearing, environment, identity, or persistent-RNG contracts.
  - Files: `particula/gpu/kernels/wall_loss.py`, `particula/gpu/kernels/tests/wall_loss_test.py`, `particula/gpu/kernels/__init__.py`
  - Tests: Mixed charged/neutral sparse slots, all-survive/all-remove controls, exact zero-time no-op, complete removed-slot clearing, survivor preservation, supplied RNG reuse, and invalid-call non-advancement.

- [ ] **E6-F4-P5:** Add deterministic parity and stochastic neutral-fallback validation
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Demonstrate charged and fallback coefficients match CPU behavior and survival outcomes match their expected distributions on Warp CPU, with optional CUDA evidence.
  - Files: `particula/gpu/kernels/tests/wall_loss_parity_test.py`, `particula/gpu/kernels/tests/wall_loss_test.py`
  - Tests: Geometry/charge/field matrix, zero-potential image charge, exact deterministic neutral fallback, repeated-step RNG lifecycle, binomial confidence bounds, sparse and multi-box fixtures, CUDA clean skips.

- [ ] **E6-F4-P6:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document charged direct use, image-charge/field semantics, neutral fallback, RNG ownership, validation commands, and supported/deferred boundaries.
  - Files: `AGENTS.md`, `docs/Features/`, `.opencode/guides/`, E6 plan sections as needed
  - Tests: Markdown links, imports, SI units, focused commands, and support/deferred table review.
