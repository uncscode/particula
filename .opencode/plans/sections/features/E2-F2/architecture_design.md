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
        └─ humidity/saturation state: (n_boxes,) float64

Current processes
  -> continue accepting scalar temperature/pressure until migration tracks
Future processes
  -> read EnvironmentData per-box fields and mutate only documented fields
```

## Data / API / Workflow Changes

- **Data Model:** Add a mutable CPU dataclass with one-dimensional per-box
  arrays. The likely fields are `temperature`, `pressure`, and one clearly
  named humidity/saturation field chosen to match E2-F1 schema terminology.
- **API Surface:** Export `EnvironmentData` through `particula.gas`. Do not
  change existing dynamics method signatures in this feature.
- **Workflow Hooks:** Downstream GPU mirror and process-migration tracks can use
  this CPU schema as their source of truth.

## Validation Rules

- All fields must be one-dimensional arrays with identical length.
- `n_boxes` is derived from `temperature.shape[0]`.
- Temperature must be finite and positive in Kelvin.
- Pressure must be finite and nonnegative or positive according to E2-F1's
  schema decision; prefer physically positive pressure unless zero-pressure
  tests are intentionally supported.
- Humidity/saturation state must be finite and nonnegative. If named relative
  humidity, enforce an upper bound of `1.0`; if named saturation ratio, allow
  supersaturation values above `1.0`.

## Security & Compliance

No permissions, network, file-system, or security-sensitive behavior changes.
The primary robustness concern is rejecting invalid numeric state before it can
enter numerical kernels.
