# Architecture Design

## High-Level Design

`EnvironmentData` is a third CPU data container that sits beside, not inside,
the existing particle and gas-species containers.

```text
Simulation state
  ├─ ParticleData      -> per-box particles and concentrations
  ├─ GasData           -> per-box gas-species concentrations
  └─ EnvironmentData   -> per-box thermodynamic state
        ├─ temperature: (n_boxes,) float64
        ├─ pressure: (n_boxes,) float64
        └─ saturation_ratio: (n_boxes, n_species) float64

Current processes
  -> continue accepting scalar temperature/pressure until migration tracks
Future processes
  -> read EnvironmentData per-box fields and mutate only documented fields
```

## Data / API / Workflow Changes

- **Data Model:** Add a mutable CPU dataclass with one-dimensional per-box
  arrays for `temperature` and `pressure`, plus a species-resolved
  `saturation_ratio` array shaped `(n_boxes, n_species)`. The container must not
  own or mutate simulation volume; `ParticleData.volume` remains authoritative.
- **API Surface:** P1 ships the dataclass at the direct module path
  `particula.gas.environment_data.EnvironmentData` only. Package exports,
  `n_boxes`, and `copy()` are intentionally deferred. Existing dynamics method
  signatures are unchanged in this phase.
- **Workflow Hooks:** Downstream GPU mirror and process-migration tracks can use
  this CPU schema as their source of truth. Future phases should preserve the
  validation order already implemented: coercion -> ndim -> shared-box shape ->
  finiteness -> physical bounds.

## Validation Rules

- `temperature` and `pressure` must be one-dimensional arrays with identical
  `(n_boxes,)` length.
- `saturation_ratio` must be a two-dimensional array shaped
  `(n_boxes, n_species)`.
- `n_boxes` is derived from `temperature.shape[0]`.
- Temperature must be finite and positive in Kelvin.
- Pressure must be finite and strictly positive so the container represents a
  physically valid per-box environment and downstream transport/property
  calculations do not need a parallel zero-pressure contract.
- `saturation_ratio` must be finite and nonnegative; values above `1.0` are
  valid supersaturation and must not be rejected.

## Security & Compliance

No permissions, network, file-system, or security-sensitive behavior changes.
The primary robustness concern is rejecting invalid numeric state before it can
enter numerical kernels.
