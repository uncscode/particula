# Architecture Design

### High-Level Design

```text
validated latent heat + thermal-work sidecars
  -> P1: validate both caller-owned sidecars atomically
  -> P2: for each of exactly four substeps
      -> E4-F1 refresh saturation pressure from current temperature
      -> E4-F2 compute shared activity/Kelvin surface pressure
      -> use that pressure for the pressure delta and fp64 thermal correction
      -> bound transfer against available particle mass
      -> mutate mass and accumulate transfer
  -> P3: reduce signed applied transfer times latent heat into optional energy output
  -> P4: exercise scalar and explicit-environment routes against the same oracle
```

P1 provides private fp64 conductivity, thermal-resistance, and latent-rate
helpers. P2 attaches the corrected rate to the production kernel on all four
substeps. The supplied sidecars are validated atomically before environment
normalization, allocation, refresh, launch, or mutation; valid arrays remain
unchanged. A supplied latent array is passed by identity; omitted latent heat
uses an unread existing fp64 placeholder without allocation. `thermal_work`
is still validated but deferred. P3 derives energy from transfer actually
applied after clamping; P4 confirms that composed scalar and explicit
environment inputs retain that contract without host transfer.

### Data / API / Workflow Changes

**P3 implementation:** `energy_transfer` is an optional keyword-only,
caller-owned `wp.float64` `(n_boxes, n_species)` output on the particle device.
It requires valid latent heat and is metadata-validated before normalization,
allocation, clear, launch, or mutation; its finite/NaN/Inf contents are not
inputs. One clear launches after preflight and one post-four-substep kernel
assigns each box/species output from the sum of `total_mass_transfer` over
particles times latent heat. This avoids contended fp64 atomics, host readback,
new device buffers, and any energy work when omitted.

- **Data Model:** No container changes. P1/P2 accept caller-owned fp64,
  nonnegative, finite `(n_species,)` `latent_heat` (J/kg) and `thermal_work`
  sidecars on the active device; neither is allocated, attached to a WarpData
   schema. P2 consumes nonzero `latent_heat` entries only; `thermal_work`
   remains unconsumed.
- **API Surface:** `condensation_step_gpu()` now exposes both sidecars as
  keyword-only inputs. Existing positional calls and its two-item return remain
   valid. Omitted latent heat and zero entries take the explicit isothermal
   rate path.
- **Workflow Hooks:** E4-F1 supplies saturation pressure, E4-F2 common surface
  pressure, and E4-F3 fixed iteration/scratch. E4-F5 adds gas coupling; E4-F6
  consumes parity evidence.

### Security & Compliance

No permission or network changes. Validate shape, fp64 dtype, device, finite
values, and nonnegative latent heat before allocation, launch, or mutation.
Never perform hidden host copies/reductions or expose uninitialized buffers.
