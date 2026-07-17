# Documentation Updates

## P2 Status

Issues #1342 and #1343 required no user-facing documentation rollout. Shipped
documentation is limited to docstrings in
`particula/gpu/kernels/coagulation.py` and
`particula/gpu/dynamics/coagulation_funcs.py`, which describe exact
charged-only particle-resolved support, fp64 caller-owned charge, private
total-mass scratch, forced finite-charge preflight, and charge-conserving merge.
No public document should claim Brownian-plus-charged execution is available.

- Update public development documentation only in P4 after P3 establishes the
  combined configuration; describe charged-only as the currently executable
  low-level configuration and retain all ownership limitations.
- Update `docs/Features/data-containers-and-gpu-foundations.md` with the charged
  execution support row, `WarpParticleData.charge` authority, explicit transfer
  boundary, supported devices, and persistent RNG guidance.
- Update `docs/Features/coagulation_strategy_system.md` to distinguish CPU
  strategy composition from the bounded direct GPU configuration and list the
  E5-approved charged model only.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` to mark T3 charged-only
  execution complete while retaining Brownian-plus-charged execution as P3 work.
- Update `AGENTS.md` quick-reference notes only after behavior and focused
  reproduction commands are verified.
- Update E5 parent/child plan sections with shipped phase status, evidence,
  constraints, and any resolved charged-model/majorant decisions.
- Defer the canonical direct coagulation example and final support table to
  E5-F9; P4 may add only minimal executable snippets needed to explain this
  feature's verified low-level API.

Validation includes Markdown link checking, import/reference verification, and
execution of any added snippet on Warp CPU when Warp is installed. Do not claim
high-level runnable support, exact stochastic parity, unsupported charged
variants, or mandatory CUDA coverage.
