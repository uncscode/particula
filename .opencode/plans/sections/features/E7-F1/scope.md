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

## Delivered in P2 (issue #1463)

- `particula/execution.py` now provides frozen `ExecutionRequest` validation,
  `ExecutionContext`, and a private per-context `_AdapterRegistry` keyed only
  by exact `(Process, Backend)` pairs.
- Resolution validates the typed request and CPU spelling, calls
  `CapabilityMatrix.require()`, then performs exactly one exact adapter lookup.
  Supported but unregistered requests raise `LookupError`; rejected requests do
  not reach lookup.
- CPU normalizes only `Device(Backend.CPU, "cpu")`; Warp native identifiers are
  retained verbatim without optional-backend import, availability probing, or
  device resolution.
- `particula/tests/execution_test.py` verifies validation order, private
  registry non-mutation and context locality, identity selection, no execution
   or fallback, and guarded dependency-neutral import behavior.

## Delivered in P3 (issue #1464)

- `particula/execution.py` now contains dependency-neutral internal P3
  contracts: runtime structural `ExecutionState` and `ExecutionAdapter`
  protocols, closed `MutationScope`/`MutationDeclaration` values, opaque
  `BackendResult`, immutable `ExecutionResult`, and
  `validate_execution_result()`.
- The validator preserves the supplied caller state and valid result by
  identity; it validates result layout, ordered immutable metadata, explicit
  mutation permission, and backend-result wrapper type without executing an
  adapter or inspecting opaque payloads/results.
- `particula/tests/execution_test.py` covers structural protocols, exact frozen
  layouts, accepted/rejected mutation declarations, identity and opacity,
  rejection non-mutation, and P2 selection separation.

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
- Public exports and user-facing documentation for the P3 contracts; P3 remains
  internal until a later deliberate publication phase.
