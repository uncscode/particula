# E2-F7 Architecture Design

## Design Goals

- Keep captured GPU execution fixed-shape and deterministic.
- Separate setup, allocation, validation, and stress-case generation from the
  graph-capturable timestep body.
- Preserve the current scalar `temperature`/`pressure` API until E2-F2 provides
  `EnvironmentData` and E2-F3 provides `WarpEnvironmentData`.
- Treat fp64 as the reference precision unless E2-F6 explicitly authorizes a
  different precision envelope.
- Require gas-coupled production condensation integration for physical
  completeness, including conservation checks that account for gas depletion; if
  that implementation grows beyond this feature's issue size, split it into an
  explicit follow-up feature rather than leaving the requirement implicit.
- Make future gradient paths explicit: no stochastic theta modes, no unbounded
  adaptive loops, and documented clamp/guard behavior.

## Proposed Components

1. **Stress-case catalog**
   - Deterministic fixtures that construct particle, gas, and environment
     states across high-stiffness and low-stiffness regimes.
   - Cases should include nanometer high-supersaturation conditions,
     accumulation-mode aerosol, and droplet-like larger particles.
   - Each case records expected shape, particle/gas inventory assumptions, and
     reference tolerances.

2. **Stability metric helpers**
   - Metrics for non-negative mass, bounded fractional mass change, monotonic
     convergence toward equilibrium where applicable, CPU/GPU parity, and
     conservation caveats.
    - The current GPU path does not update gas concentration, so metrics must
      clearly distinguish particle-only GPU behavior from full gas-particle
      mass conservation and identify the gas-coupled production integration gap.

3. **Explicit timestep scan**
   - A fixed list of timestep candidates per stress case.
   - No adaptive while loop is required for captured execution; scans can live
     in test/benchmark code outside graph capture.
   - Results become the stiffness map.

4. **Integration candidate evaluation**
   - Baseline: fixed-count sub-stepping with preallocated buffers.
   - Candidate: deterministic semi-implicit/asymptotic first-order update that
     preserves fixed shapes and avoids dynamic control flow.
   - Prior art: CPU staggered update, used for comparison only unless a fixed
     deterministic batch schedule is designed.

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
