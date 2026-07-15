# Architecture Design

## High-Level Design

```text
independent CPU equations per box
             |
deterministic fp64 fixtures
             v
E4-F1..F5 production condensation step
  -> Warp CPU (required) / CUDA (optional)
  -> parity assertions (physics tolerance)
  -> conservation assertions (strict, separate)
  -> mutation and reusable-buffer assertions
  -> fixed-four-loop capture/replay evidence
  -> bounded out-of-place smooth-interior autodiff experiment
             |
             v
documented evidence matrix and limitations
```

Tests observe the direct production entry point and derive expected values with
the independent NumPy fixed-four-substep/P2/gas-coupled oracle. The delivered
P1 matrix uses two NumPy-owned descriptors and separately asserts particle mass
and gas concentration, avoiding circular validation through GPU buffers or
aggregate-only comparisons.

## Data / API / Workflow Changes

- **Data Model:** No container-schema change. Existing particle masses,
  gas concentration, environment/configuration arrays, transfer outputs, energy
  sidecars, and caller-owned fixed-shape scratch remain authoritative.
- **API Surface:** No new public diagnostics API. Test helpers may provide
  reference assembly, invariant calculation, capture setup, and bounded
  autodiff probes. Complete caller-owned scratch is reused on-device.
- **Workflow Hooks:** Pytest device fixtures always include Warp CPU and add
  CUDA only when available. Cases remain discoverable in `*_test.py` modules
  with `warp`, `gpu_parity`, and other established markers.

Exactly four substeps are retained. Capture setup occurs outside capture;
replay performs no hidden host transfer or adaptive allocation. Autodiff
evidence is limited to an out-of-place smooth-interior slice and explicitly
excludes clamps, inventory gates, and in-place mutation.

## Security & Compliance

There are no permission or external-input changes. Validation must reject
mixed-device, wrong-shape, nonfinite, or invalid configuration before any state
mutation. Tests must avoid treating optional hardware absence as success for
the mandatory Warp CPU baseline.
