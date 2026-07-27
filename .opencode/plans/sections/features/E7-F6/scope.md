# Scope

E7-F6 delivers issue #1451 Track T6 on top of E7-F1: capability failures,
backend availability, explicit CPU fallback boundaries, deliberate exports, and
API-stability rules. It freezes these cross-cutting contracts before E7-F2,
E7-F3, and E7-F4 build GPU adapters and resident sessions.

## In Scope

- Typed execution errors for unknown backend/device, unavailable runtime/device,
  unsupported process/capability, invalid state, and disallowed fallback.
- Deterministic pre-execution availability resolution without probing by
  launching kernels.
- Fallback disabled by default and enabled only by an explicit typed request.
- CPU fallback before upload/mutation when authoritative CPU state is available,
  or after a caller-requested checkpoint/finalize restore boundary.
- Preservation of the original error reason and an observable resolution/result
  showing the requested and selected backend.
- Narrow exports from `particula.execution` and, where E7-F1 specifies, the
  top-level package; concrete registries, adapters, sidecars, and kernel configs
  remain internal or concrete-module-only.
- Experimental labeling and compatibility/deprecation policy for existing
  low-level `particula.gpu.*` APIs.
- Unit, import-surface, optional-Warp, no-transfer, and negative integration
  tests with unchanged coverage thresholds.

## Out of Scope

- Catching a kernel/runtime failure and retrying on CPU.
- Automatic checkpoint, restore, synchronization, conversion, or backend
  movement inside normal resident timesteps.
- GPU condensation/coagulation adapters (E7-F2/E7-F3), resident lifecycle
  implementation (E7-F4), or scheduler policy (E7-F5).
- New GPU physics, kernel rewrites, transport, RNG policy, multi-GPU,
  distributed execution, graph capture, autodiff, or performance claims.
- Removing or renaming shipped direct GPU APIs in this feature.
