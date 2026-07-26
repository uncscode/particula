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
        P3 exact int32 count conversion -> E6-F5 slot diagnostics
                               |
              retained full counts + free-slot selectable prefix
                               |
                    E6-F6 complete exhaustion plan (deferred)
             resampling first; scaling transforms demand
                              |
             every box feasible and conservative?
                  no -> error, no caller writes
                  yes -> commit once on device
                              |
        particle source added == participating gas removed
```

For each box, shipped P2 preserves E6-F7's `E_pot = J*dt`, where survival is
already included in `J`, and `m_event,s = n_s*M_s/N_A`. It computes one common
inventory limit across participating species, so `E_admit <= E_pot` and
`removal_s = E_admit*m_event,s`; therefore no species becomes negative and
source composition is not skewed. Gas-limited diagnostics encode the first
lowest-index limiting participating species. P2 commits only its demand,
 removal, and gate-code sidecars. Shipped P3 privately converts each exact,
 finite, nonnegative `accepted_demand * particles.volume` product in the
 inclusive int32 range, reads one conversion status, then calls E6-F5
 `get_slot_diagnostics_gpu` with supplied diagnostics. Its sole writer retains
 the full count and clears/fills selected indices with the deterministic prefix
 limited by count, free count, and capacity. Capacity planning, representative
 scaling, activation, and the particle/gas commit remain P4--P5 work.

## Data / API / Workflow Changes

### P1/P2 implementation status (#1438, #1439)

P1 and private P2 implement the initial stages in
`particula/gpu/kernels/nucleation.py`: frozen `NucleationConfig` and the three
caller-owned sidecar records, plus private `_preflight_nucleation`. Preflight
validates fixed-shape Warp metadata, sidecars, aliasing, physical state,
species/count constraints, input-source rules, and gates without writes,
fallback allocation, or transfer. P2 calculates device-resident
survival-included rates, potential demand, common inventory-limited admission,
planned precursor removal, and gate diagnostics, then commits only its
  designated sidecars. P3 adds `free_slot_indices (B, N)`,
  `active_slot_counts (B,)`, and `free_slot_counts (B,)` to
  `NucleationDiagnosticBuffers`; all five P3 arrays are same-device contiguous
  `wp.int32` storage covered by overlap checks. P3 does not mutate particles or
  gas and no symbol is exported from `particula.gpu.kernels`; the staged
  particle transaction and later writer stages remain P4--P5 work.

- **Data Model:** No required container fields. Add concrete-module
  `NucleationConfig`, `NucleationScratchBuffers`,
  `NucleationFinalizedDemandBuffers`, and `NucleationDiagnosticBuffers`
  containing explicit same-device, fixed-shape arrays. Per-box fields use
  `(n_boxes,)`; species diagnostics use `(n_boxes, n_species)`; request fields
  use `(n_boxes, n_particles[, n_species])` with `wp.int32` valid-prefix counts.
  Supplied arrays retain identity and unrequested index tails use `-1`.
- **API Surface (deferred beyond P1):** Add keyword-oriented
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
- **Failure Boundary:** P3 conversion rejection occurs before P3/E6-F5 output
  writes; E6-F5 retains its preflight-before-diagnostic-write contract. Its
  commit starts only after both stages succeed. No rollback is promised after a
  launched E6-F5 or P3 writer kernel; callers synchronize before consuming
  outputs and may rebuild sidecars from unchanged particle/gas/P2 state.

## Security & Compliance

No network, persistence, or permission behavior changes. Scientific safety
requires finite physical state, closed-domain validation, inventory-limited
admission, per-species conservation, explicit tolerances, and no misleading
claims about unsupported nucleation physics. Device safety requires same-device
typed arrays, overlap checks, bounded integer counts, stable shapes, and no
hidden transfer, allocation-based resize, or host fallback.
