# Architecture Design

## High-Level Design

This implemented evidence feature is layered on the shipped direct-kernel API.
The walkthrough owns fixture construction, the independent oracle, explicit
Warp sidecars, and synchronized observations; production condensation code is
unchanged.

```text
Immutable physical fixture constants
  +--> independent NumPy fixed-four-substep oracle
  |      +--> expected particle mass / gas concentration / transfer / energy
  |
  +--> separately allocated Warp fp64 state and sidecars
         +--> condensation_step_gpu(device="cpu")
                +--> synchronize and copy observations

expected + observations
  +--> physics: masses / gas / P2 total and exact vapor pressure
  +--> conservation: concentration-weighted particle-plus-gas drift
  +--> energy: P2 total transfer times latent heat
```

`run_example(device="cpu")` returns the completed oracle and, when enabled,
the restored particle/gas state plus copied raw-proposal, total-transfer,
energy, and vapor-pressure observations. Its immutable `AcceptanceResult`
blocks report all three categories without numerical-mismatch short-circuiting.
No-Warp routes return all blocks as `unavailable` after the exact message
`oracle completed; no kernel ran`. Enabled-route errors propagate; the detached
Warp source and sidecars must be discarded and rebuilt before retrying.

## Data / API / Workflow Changes

- **Data Model:** No production schema changes. The example may use a small
  local immutable fixture/result structure solely for deterministic reporting.
- **API Surface:** No package export. The shipped user-facing artifact is
  `docs/Examples/gpu_condensation_parity_walkthrough.py`.
- **Workflow Hooks:** The co-located pytest module validates CPU-safe,
  force-disabled, fake enabled, failure-recovery, Warp-CPU, and optional CUDA
  routes. CUDA remains optional.
- **Ownership Boundary:** CPU expected arrays must not alias or be generated
  from Warp output arrays. Caller-owned `energy_transfer` is synchronized and
  observed as diagnostic state, not treated as a third return value.

## Acceptance Categories

| Category | Quantity | Implemented criterion |
|----------|----------|-------------------|
| Physics | Final particle masses, gas concentrations, and P2 total transfer vs independent oracle; constant-mode vapor pressure | `rtol=2e-10`, `atol=1e-30`; vapor pressure exact. |
| Conservation | Per-box/per-species concentration-weighted particle-plus-gas drift | `rtol=1e-12`, `atol=1e-30`. |
| Energy | `sum_particles(P2 finalized transfer) * latent_heat` by box/species | `rtol=1e-12`, `atol=1e-18`. |

Raw proposal remains a P1 diagnostic and is not an acceptance input. The
walkthrough does not claim high-level CPU-strategy parity.

## Security & Compliance

No permissions, network access, credentials, or external data are added. The
walkthrough must remain deterministic, avoid unsafe deserialization, validate
all shapes/devices before execution, and make optional dependency/device skips
explicit. Documentation must not represent a skipped CUDA path as evidence.
