# Documentation Updates

## P1 Status

Issue #1342 required no user-facing documentation update. The only shipped
documentation is the concise internal O(n²) correctness-first rationale beside
the private charged majorant. Public documentation remains deferred until P2/P3
make charged execution available; it must not claim that charged coagulation is
executable today.

- Update `particula/gpu/kernels/coagulation.py` docstrings to describe the
  exact charged-only and Brownian-plus-charged configuration, one-pass additive
  semantics, required charge state, and unchanged return/buffer/RNG ownership.
- Update `docs/Features/data-containers-and-gpu-foundations.md` with the charged
  execution support row, `WarpParticleData.charge` authority, explicit transfer
  boundary, supported devices, and persistent RNG guidance.
- Update `docs/Features/coagulation_strategy_system.md` to distinguish CPU
  strategy composition from the bounded direct GPU configuration and list the
  E5-approved charged model only.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` to mark T3 charged-only and
  Brownian-plus-charged execution complete while leaving E5-F4-F7/F9 open.
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
