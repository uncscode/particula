# Architecture Design

### High-Level Design

```text
validated latent heat + thermal-work sidecars
  -> P1: retain both caller-owned sidecars without consuming them
  -> future P2/P3: for each of exactly four substeps
     -> E4-F1 refresh saturation pressure from current temperature
     -> E4-F2 compute shared activity/Kelvin surface pressure
     -> compute isothermal rate and fp64 thermal correction
     -> bound transfer against available particle mass
     -> mutate mass; accumulate transfer and signed Δm * L
  -> preserve existing particle/transfer return
  -> optional energy sidecar contains whole-call totals
```

P1 provides private fp64 conductivity, thermal-resistance, and latent-rate
helpers, but deliberately does not attach the corrected rate to the production
kernel. The supplied sidecars are validated atomically before environment
normalization, allocation, refresh, launch, or mutation; valid sidecars remain
unchanged. Future P2 consumes latent heat with the same surface pressure as
pressure-delta logic, and P3 derives energy from transfer actually applied
after clamping.

### Data / API / Workflow Changes

- **Data Model:** No container changes. P1 accepts caller-owned fp64,
  nonnegative, finite `(n_species,)` `latent_heat` (J/kg) and `thermal_work`
  sidecars on the active device; neither is allocated, attached to a WarpData
  schema, or consumed in P1.
- **API Surface:** `condensation_step_gpu()` now exposes both sidecars as
  keyword-only inputs. Existing positional calls and its two-item return remain
  valid, and the production path remains isothermal in P1.
- **Workflow Hooks:** E4-F1 supplies saturation pressure, E4-F2 common surface
  pressure, and E4-F3 fixed iteration/scratch. E4-F5 adds gas coupling; E4-F6
  consumes parity evidence.

### Security & Compliance

No permission or network changes. Validate shape, fp64 dtype, device, finite
values, and nonnegative latent heat before allocation, launch, or mutation.
Never perform hidden host copies/reductions or expose uninitialized buffers.
