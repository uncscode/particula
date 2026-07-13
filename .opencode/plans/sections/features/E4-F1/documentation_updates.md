# Documentation Updates

- Shipped code-level documentation: `thermodynamics.py` documents the
  concrete-module-only `refresh_vapor_pressure_gpu` import, its validated Warp
  `float64` boundary, constant Pa semantics, canonical Buck water/ice behavior,
  and Buck's reserved/unused parameter slots.
- Migrated the executable GPU direct-kernels quick-start call with the required
  keyword-only sidecar while keeping the runnable example concise.
- Updated `AGENTS.md` with the concrete-module GPU vapor-pressure refresh API,
  validated Warp `float64` temperature input, and constant/Buck behavior.
- Issue #1285 shipped P5 documentation-only work in the canonical foundation
  guide and Epic D roadmap: exact configuration/derived-buffer ownership,
  constant/Buck support, refresh ordering, import boundary, and deferred
  ownership. It made no production-code or supported-model expansion.
- Deferred ownership is E4-F2 activity/effective surface tension; E4-F3 fixed
  four-substep orchestration; E4-F4 latent-heat correction and energy
  diagnostics; E4-F5 gas-coupled inventory/partitioning and conservation;
  E4-F6 independent Warp/CUDA parity, conservation, and readiness evidence;
  and E4-F7 the final support contract, examples, troubleshooting, and support
  matrix.
