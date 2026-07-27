# Scope

E7-F1 delivers issue #1451 Track T1: the typed vocabulary and CPU reference
boundary on which every later backend-selected process and resident session
depends. Selection is placed in a separate execution context rather than on
strategies, builders, or the existing `Aerosol`-typed runnable hierarchy.

## In Scope

- A typed backend identifier, device request, process/capability descriptor,
  execution-state protocol, and result contract.
- A declarative capability matrix with deterministic query and validation.
- Explicit state ownership, in-place mutation, identity, and return semantics.
- A CPU adapter over existing `RunnableABC.execute()` behavior; CPU is the
  reference path and performs no Warp import or conversion.
- Early validation for malformed backend/device/capability combinations.
- Deliberate public exports and positive/negative contract tests.
- Documentation of extension points consumed by E7-F2 through E7-F6.

## Delivered in P1 (issue #1462)

- `particula/execution.py` supplies frozen, standard-library-only metadata for
  closed backend identity, opaque native devices, process and capability names,
  exact capability requirements, declarations, and a capability matrix.
- Matrix lookup is structural and read-only: nonempty requirements require one
  complete declaration, and an empty request succeeds only for an otherwise
  declared device/process base.
- `particula/tests/execution_test.py` covers validation, immutability, exact
  matching, non-mutation, and an import path guarded against optional Warp/GPU
  imports.

## Out of Scope

- GPU condensation or coagulation adapters (E7-F2 and E7-F3).
- Resident Warp containers, sidecars, checkpoints, or lifecycle (E7-F4).
- Full-process scheduling and thermodynamic refresh order (E7-F5).
- Final fallback/error taxonomy, deprecation, and stability policy (E7-F6),
  beyond the T1 invariant that no fallback or transfer is implicit.
- Multi-box transport, persistent cross-process RNG policy, and closeout
  regressions (E7-F7 through E7-F9).
- Kernel-physics changes, GPU staggered condensation, dynamic resizing,
  multi-GPU/distributed execution, graph capture, optimization, or autodiff.
