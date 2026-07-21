# Open Questions

- [ ] Which neutral transport primitives are genuinely absent from
  `particula/gpu/properties/`, and which existing symbols can be reused without
  changing their current public status?
  - Resolution target: E6-F3-P1 inventory before new device code is approved.
- [ ] Should `time_step == 0` skip all RNG draws so the caller-owned state is an
  exact no-op, or consume draws while preserving particle state?
  - Proposed resolution: skip draws and preserve RNG exactly; freeze and test in
    P3 because this affects repeated-step reproducibility.
- [ ] What exact fixed-slot active predicate should E6-F3 accept before generic
  half-active normalization/rejection ships in E6-F5?
  - Proposed resolution: require positive concentration and positive finite
    total mass for active slots; all-zero mass/concentration/charge is inactive;
    reject contradictory concentration/mass states before RNG or mutation.
- [ ] Should wall-loss RNG expose a wall-loss-specific initializer or reuse a
  shared private/public RNG initialization utility extracted from coagulation?
  - Resolution target: P5; preserve `(n_boxes,)` `wp.uint32`, explicit reset,
    same-device validation, and no cross-process shared-stream assumption.
- [ ] Should deterministic coefficient diagnostics be returned by the public
  step or tested through a concrete-module helper only?
  - Proposed resolution: keep the production return contract minimal and expose
    no new caller-owned diagnostic unless E6-F9 demonstrates a real integration
    need; deterministic helper testing is sufficient for E6-F3.
- [x] Must CPU and Warp use identical random-number sequences?
  - Resolved 2026-07-21: No. Parent E6 requires deterministic coefficient parity
    and statistically bounded outcomes, not exact RNG sequence parity.
- [x] Does E6-F3 include charged wall loss, dynamic slot management, high-level
  GPU runnable integration, or scheduler/backend selection?
  - Resolved 2026-07-21: No. These belong to E6-F4, E6-F5/F6, and Epic G.
