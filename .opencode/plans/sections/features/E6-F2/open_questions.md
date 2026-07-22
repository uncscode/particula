# Open Questions

All E6-F2 planning questions were resolved on 2026-07-21 from the E6-F1
reference decision and current direct-kernel conventions.

- [x] Which finite-step update must GPU dilution mirror?
  - Decision: apply the E6-F1 exact exponential factor per box. No GPU-specific
    integrator, limiter, or competing formula is permitted.
- [x] Which coefficient and time shapes are accepted?
  - Decision: `coefficient` accepts a Python/NumPy floating scalar or a
    same-device `wp.float64` array shaped `(n_boxes,)`; `time_step` remains one
    finite nonnegative scalar for the whole step.
- [x] How is a floating scalar coefficient represented on device?
  - Decision: P1 broadcasts an accepted scalar into a private same-device
    `wp.float64` buffer. A valid caller-owned Warp array is retained by identity;
    host arrays are not transferred implicitly.
- [x] Are concentration scans optional?
  - Decision: P3 requires finite nonnegative per-box coefficient and
    concentration scans, with exact float64 same-device Warp schemas, before
    every no-op, allocation, or launch. Rejected-call atomicity is guaranteed by
    read-only preflight; rollback after a launched-kernel failure remains P4
    scope. There is no validation opt-out.
- [x] What CPU/Warp tolerance is frozen?
  - Decision: begin with `rtol=1e-12`, `atol=0` for changed finite nonzero
    concentrations and exact equality for no-ops, zeros, identities, and
    protected fields on required Warp CPU and optional CUDA. P4 records measured
    errors and may use only the smallest fixture-specific relaxation justified
    by evidence.
