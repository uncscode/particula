# Documentation Updates

- Update `.opencode/guides/testing_guide.md` with the canonical E5
  cross-mechanism fixture matrix, deterministic/stochastic separation, marker
  commands, and conservation assertion policy if existing guidance is not
  sufficient to reproduce the suite.
- Publish `docs/Features/Roadmap/coagulation-validation.md` with a table for
  Brownian, charged, Brownian-plus-charged, SP2016, ST1956, approved two-way
  combinations, and the full four-way row. Record deterministic parity,
  stochastic bounds, mass/charge, multi-box/inactive, Warp CPU, and optional
  CUDA status separately.
- Document focused commands such as deterministic `gpu_parity`, bounded
  `stochastic`, optional `cuda`, and full coagulation regression runs.
- State explicitly that CUDA is additive evidence, exact CPU/Warp pair replay is
  not required, and no claims extend to unsupported mechanism rows, DNS,
  non-unit sedimentation efficiency, or performance.
- Provide E5-F9 with stable mechanism names, tested input conditions,
  tolerances/bounds, and evidence-file references for user-facing support docs
  and examples.
- Update `.opencode/plans/sections/features/E5-F7/` with shipped statuses,
  issue numbers, final commands, and resolved questions as phases complete.

No `README` or top-level public import update is required unless E5-F9 chooses
to surface the validation commands there; E5-F7 introduces no production API.

## Status

Issues #1362 and #1363 added private validation-only P1/P2 coverage and made no
user-facing documentation changes. The planned public evidence matrix and
documentation updates remain P4 work.
