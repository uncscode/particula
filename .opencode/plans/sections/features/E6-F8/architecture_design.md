# Architecture Design

## High-Level Design

The entry point is a staged device transaction. Host-visible metadata and
device values are validated before caller outputs are cleared. Rate and source
kernels write only work sidecars. A complete E6-F5/E6-F6 capacity plan and
conservation precheck gate the one commit sequence; no box commits early.

```text
WarpParticleData + WarpGasData + config + dt + fixed-shape sidecars
                              |
           metadata and read-only device-value preflight
                              |
       E6-F7 equations -> potential events and source mass
                              |
       shared per-box/species gas admission factor
                               |
                 provisional demand sidecars
                              |
       E6-F5 slot discovery -> enough fixed slots?
                              | no
                   E6-F6 complete exhaustion plan
             resampling first; scaling transforms demand
                              |
             every box feasible and conservative?
                  no -> error, no caller writes
                  yes -> commit once on device
                              |
        particle source added == participating gas removed
```

For each box, the port preserves E6-F7's `E_pot = J*dt*V*f_survival` and
`m_event,s = n_s*M_s/N_A`. It computes one `alpha` across participating species
so `E_admit = alpha*E_pot`; therefore no species becomes negative and source
composition is not skewed. Admitted demand reaches capacity planning. If E6-F6
chooses representative scale `s`, it finalizes `V_new=s*V` and
`E_new=s*E_admit` together before slot packaging and commit. No demand is
discarded within the final represented domain.

## Data / API / Workflow Changes

- **Data Model:** No required container fields. Add concrete-module
  `NucleationConfig`, `NucleationScratchBuffers`,
  `NucleationFinalizedDemandBuffers`, and `NucleationDiagnosticBuffers`
  containing explicit same-device, fixed-shape arrays. Per-box fields use
  `(n_boxes,)`; species diagnostics use `(n_boxes, n_species)`; request fields
  use `(n_boxes, n_particles[, n_species])` with `wp.int32` valid-prefix counts.
  Supplied arrays retain identity and unrequested index tails use `-1`.
- **API Surface:** Add keyword-oriented
  `nucleation_step_gpu(particles, gas, ..., config=..., scratch=...)` under
  `particula.gpu.kernels.nucleation`; lazily expose only the intended step from
  `particula.gpu.kernels`. Keep config and sidecars concrete-module APIs.
- **Mutation Contract:** Success may change selected particle mass,
  concentration/weight and charge, matching gas concentration, and only the
  E6-F6-authorized volume/weights. Density, metadata, shapes, devices, dtypes,
  container identities, input configuration, and unselected state stay fixed.
- **Workflow Hooks:** E6-F5 and E6-F6 are mandatory capacity dependencies;
  E6-F7 is the scientific and numerical oracle; E6-F9 consumes this low-level
  step in an explicit-transfer integrated sequence.
- **Failure Boundary:** Invalid scientific values, metadata, device data,
  aliasing, scratch, capacity, policy, or conservation fail before particle,
  gas, diagnostic, work-buffer, volume, or RNG mutation. Post-launch device
  faults cannot promise rollback and must be documented separately.

## Security & Compliance

No network, persistence, or permission behavior changes. Scientific safety
requires finite physical state, closed-domain validation, inventory-limited
admission, per-species conservation, explicit tolerances, and no misleading
claims about unsupported nucleation physics. Device safety requires same-device
typed arrays, overlap checks, bounded integer counts, stable shapes, and no
hidden transfer, allocation-based resize, or host fallback.
