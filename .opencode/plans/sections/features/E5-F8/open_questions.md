# Open Questions

- [x] Which exact canonical per-field `rtol`/`atol` pairs apply to the selected
  walkthrough fixture?
  - Resolved 2026-07-16: final particle mass, gas concentration, and total
    transfer use `rtol=2e-10, atol=1e-30`; constant vapor pressure is exact with
    `rtol=0, atol=0`. Per-box/species conservation uses the existing
    `max(scale * 1e-12, 1e-30)` bound. Signed energy transfer uses
    `rtol=1e-12, atol=1e-18`. Keep physics, conservation, and energy as separate
    acceptance categories.
- [x] Should the optional CUDA run be exposed by a command-line flag or remain a
  pytest-only selection?
  - Resolved 2026-07-16: keep the script Warp-CPU-default with no new CLI flag.
    Permit programmatic `run_walkthrough(device="cuda")` for guarded tests and
    expose optional CUDA through existing pytest markers and skip helpers.
- [x] Does Epic F formally accept both `thermal_work` consumption/temperature
  feedback and adaptive condensation stepping?
  - Resolved 2026-07-16: no. Both are out of E5 scope and require a future,
    separately approved condensation numerical-method plan. E5 records the
    non-claim and does not broaden Epic F to absorb them.
- [x] What plan ID will own phase-aware surface tension and BAT activity?
  - Resolved 2026-07-16: E5 assigns no current plan ID. Both are out of scope and
    require a future condensation-physics expansion plan whose ID and validation
    contract must exist before implementation begins.
- [x] Is CUDA required for E5-F8 completion?
  - Resolved 2026-07-15: No. Warp CPU is required whenever Warp is installed;
    CUDA is optional additive evidence and must skip cleanly when unavailable.
- [x] Does this walkthrough establish high-level CPU strategy parity?
  - Resolved 2026-07-15: No. It establishes only the bounded independent NumPy
    fixed-four-substep oracle versus low-level Warp direct-kernel criteria.
