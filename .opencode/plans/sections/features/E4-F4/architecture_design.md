# Architecture Design

### High-Level Design

```text
validated latent heat + E4-F3 caller-owned scratch
  -> for each of exactly four substeps
     -> E4-F1 refresh saturation pressure from current temperature
     -> E4-F2 compute shared activity/Kelvin surface pressure
     -> compute isothermal rate and fp64 thermal correction
     -> bound transfer against available particle mass
     -> mutate mass; accumulate transfer and signed Δm * L
  -> preserve existing particle/transfer return
  -> optional energy sidecar contains whole-call totals
```

The latent branch consumes the same surface pressure as pressure-delta logic.
Conductivity and correction inputs refresh per box and substep. Energy uses
transfer actually applied after clamping.

### Data / API / Workflow Changes

- **Data Model:** No container changes. Latent heat is fp64 `(n_species,)`;
  thermal work and `(n_boxes, n_species)` energy are caller-owned sidecars.
- **API Surface:** Add keyword-only optional latent and output/scratch buffers
  to `condensation_step_gpu()`. Existing positional calls and two-item return
  remain valid. Omitted latent configuration stays isothermal.
- **Workflow Hooks:** E4-F1 supplies saturation pressure, E4-F2 common surface
  pressure, and E4-F3 fixed iteration/scratch. E4-F5 adds gas coupling; E4-F6
  consumes parity evidence.

### Security & Compliance

No permission or network changes. Validate shape, fp64 dtype, device, finite
values, and nonnegative latent heat before allocation, launch, or mutation.
Never perform hidden host copies/reductions or expose uninitialized buffers.
