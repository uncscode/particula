# Phase Details

## Sequencing

E6-F3 must ship first. Complete P1 before P2/P3, then integrate both physics
terms in P4 before P5 validation and P6 documentation.

- [x] **E6-F4-P1:** Freeze charged wall-loss configuration and semantics with unit tests
  - Issue: #1409 | Size: S | Status: Shipped
  - Goal: Delivered neutral/charged configuration validation, potential/field schema and ownership checks, rectangular-field ordering and preflight atomicity while deliberately retaining neutral execution.
  - Files: `particula/gpu/kernels/wall_loss.py`, `particula/gpu/kernels/tests/wall_loss_test.py`
  - Tests: Accepted geometry-specific forms; finite potential/field validation; wrong rank, length, dtype, or device; unsupported mode; field-before-charge ordering; field/particle/RNG immutability on failure; charged/neutral zero-charge equivalence.

- [x] **E6-F4-P2:** Implement image-charge enhancement device primitives with unit tests
  - Issue: #1410 | Size: S | Status: Shipped
  - Goal: Delivered private fp64 Warp self-pair Coulomb-ratio and image-charge
    enhancement helpers, including the `-200` raw-ratio floor, absolute value,
    `[-50, 50]` exponent clipping, and exact factor one for zero charge.
  - Files: `particula/gpu/dynamics/wall_loss_funcs.py`, `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`
  - Tests: Independent NumPy/Warp parity covers direct-ratio ordinary and
    saturated lanes, signed/zero charges, finite positive radii and
    temperatures, equal-magnitude charge symmetry, nonzero enhancement, and
    both clipping domains on Warp CPU with optional CUDA rows.
  - Boundary: No public export/API, kernel integration, configuration or
    preflight change, potential/field composition, CPU change, or RNG change.

- [x] **E6-F4-P3:** Implement electric-field drift and charged coefficient composition with unit tests
  - Issue: #1411 | Size: S | Status: Shipped
  - Goal: Delivered private fp64 Warp geometry-scale, spherical/rectangular
    field-resolution, signed mobility-drift, and safely composed charged-
    coefficient helpers matching CPU semantics.
  - Files: `particula/gpu/dynamics/wall_loss_funcs.py`, `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`
  - Tests: Independent NumPy/Warp tests cover signed spherical and rectangular
    vector fields, potential-only and explicit fields, signed and scaled charge,
    zero field/charge controls, radius and geometry guards, cancellation, NaN,
    negative infinity, and positive-overflow composition on Warp CPU with
    optional CUDA rows.
  - Boundary: No public export/API, direct-kernel integration, configuration or
    preflight change, CPU change, caller-state mutation, or RNG change; P4 owns
    direct-step integration.

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
