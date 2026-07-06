# E2-F7 Architecture Design

## Design Goals

- Keep captured GPU execution fixed-shape and deterministic.
- Separate setup, allocation, validation, and stress-case generation from the
  graph-capturable timestep body.
- Preserve the current scalar `temperature`/`pressure` API until E2-F2 provides
  `EnvironmentData` and E2-F3 provides `WarpEnvironmentData`.
- Treat fp64 as the reference precision unless E2-F6 explicitly authorizes a
  different precision envelope.
- Treat gas-coupled production condensation integration as an explicit
  follow-up requirement for physical completeness, including conservation
  checks that account for gas depletion; if that implementation grows beyond
  this feature's issue size, split it into an explicit follow-up feature rather
  than leaving the requirement implicit.
- Make future gradient paths explicit: no stochastic theta modes, no unbounded
  adaptive loops, and documented clamp/guard behavior.

## Proposed Components

1. **Stress-case catalog**
    - Deterministic fixtures that construct particle, gas, and environment
      states across high-stiffness and low-stiffness regimes.
    - Cases should include nanometer high-supersaturation conditions,
      accumulation-mode aerosol, and droplet-like larger particles.
    - P1 implemented this directly in
      `particula/gpu/kernels/tests/condensation_test.py` as
      `CondensationStiffnessCase` definitions with explicit `n_boxes`,
      `n_particles`, `n_species`, scalar baseline `temperature`/`pressure`, and
      deterministic particle/gas/environment builders.

2. **Stability metric helpers**
    - P1 implemented reusable helper checks for metadata validity,
      non-negative mass, finite values, bounded fractional mass change,
      zero-mass stability, and explicit stable/unstable classification.
    - Threshold semantics are executable and inclusive at the exact boundary.
    - The current GPU path does not update gas concentration, so the
      classification result marks the particle-only update caveat explicitly
      rather than implying gas-particle conservation.

3. **Explicit timestep scan**
    - A fixed list of timestep candidates per stress case.
    - No adaptive while loop is required for captured execution; scans can live
      in test/benchmark code outside graph capture.
    - P2 implemented this as a recorded-grid sweep in
      `particula/gpu/kernels/tests/condensation_test.py`, with one reused
      caller-owned `mass_transfer` buffer per case/device and fresh rebuilt
      particle/gas inputs for every trial.
    - The shipped evidence remains particle-only: particle masses change,
      gas concentration stays unchanged, and environment inputs stay split
      between scalar single-box coverage and direct Warp `(n_boxes,)` arrays for
      the multi-box case.

4. **Integration candidate evaluation**
    - Baseline: fixed-count sub-stepping with preallocated buffers.
    - Candidate: deterministic semi-implicit/asymptotic first-order update that
      preserves fixed shapes and avoids dynamic control flow.
    - Prior art: CPU staggered update, used for comparison only unless a fixed
      deterministic batch schedule is designed.
    - This remains future work; no integrator comparison shipped in P1.

## Data Flow

```text
stress case -> CPU/GPU containers -> preallocated scratch buffers
            -> explicit or candidate step -> stability metrics -> stiffness map
```

## Graph-Capture Boundary

- Capturable region should contain only fixed-count kernel launches and static
  loops.
- Buffer allocation, shape/device validation, host property calculation, and
  timestep scans remain outside the captured region.
- Active-particle changes must be represented by masks or inactive slots, not
  resized arrays.

## Autodiff Boundary

- The recommended path should be deterministic and avoid stochastic batches.
- Hard clamps must be documented; if a candidate relies on clamps, note the
  expected gradient behavior and whether a guarded smooth alternative is needed
  before optimization use.
