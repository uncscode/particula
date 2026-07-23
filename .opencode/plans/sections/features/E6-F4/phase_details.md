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

- [x] **E6-F4-P4:** Integrate charged mode into the neutral fixed-shape wall-loss step with unit tests
  - Issue: #1412 | Size: S | Status: Shipped
  - Goal: Delivered geometry-specialized charged selection in the existing
    fixed-shape direct step without changing neutral execution, fixed-slot
    clearing, environment, identity, or persistent-RNG contracts.
  - Files: `particula/gpu/kernels/wall_loss.py`, `particula/gpu/kernels/tests/wall_loss_test.py`, `particula/gpu/kernels/__init__.py`
  - Tests: Mixed charged/neutral sparse slots, image-only and field/drift
    controls, exact zero-charge fallback, selected-slot clearing, survivor and
    caller-owned field preservation, supplied RNG reuse, and exact invalid/zero-
    time/all-inactive non-advancement.

- [x] **E6-F4-P5:** Add deterministic parity and stochastic neutral-fallback validation
  - Issue: #1413 | Size: S | Status: Shipped
  - Goal: Delivered regression evidence for the existing charged direct boundary;
    no production kernel, API, CPU strategy, or documentation change.
  - Files: `particula/gpu/kernels/tests/wall_loss_parity_test.py`
  - Tests: Independent charged CPU/Warp deterministic coefficient matrix with
    non-mutation/rectangular-field ownership checks; exact zero-charge neutral
    coefficient, survivor-state, and RNG fallback; invalid/no-op regressions;
    and the frozen eight-stratum, 4,096-observation exact-binomial charged
    survival validation plus a persistent-sidecar lifecycle case. Warp CPU is
    baseline; CUDA is optional.

- [x] **E6-F4-P6:** Update development documentation
  - Issue: #1414 | Size: XS | Status: Shipped | Completion: 2026-07-23
  - Goal: Delivered documentation-only direct charged use, image-charge/field
    semantics, neutral fallback, RNG ownership, validation commands, and
    supported/deferred boundaries without code, API, or test behavior changes.
  - Files: `AGENTS.md`, `docs/Features/`, `.opencode/guides/`, E6 plan sections as needed
  - Tests: Markdown links, imports, SI units, focused commands, and support/deferred table review.
