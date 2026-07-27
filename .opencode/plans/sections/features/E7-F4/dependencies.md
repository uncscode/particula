# Dependencies

## Upstream

- **E7-F1 (T1): Backend-Selection and Execution-Context API** must define the
  typed request, execution state, adapter, ownership, and result vocabulary.
  E7-F4 places resident state under that layer rather than adding a competing
  backend selector.
- **E7-F6 (T6): Fallback, Capability Errors, and API Stability** must freeze
  Warp/device availability checks, explicit fallback boundaries, exception
  taxonomy, and deliberate export policy. E7-F4 never catches a runtime error
  to move state to CPU.
- Shipped E2 CPU/Warp fixed-shape schemas and explicit conversion helpers.
- Shipped direct GPU condensation, coagulation, dilution, wall-loss, and
  nucleation entry points plus E6-F9 integrated fixtures.
- Python 3.12+, NumPy, and optional Warp. Warp CPU is the required validation
  baseline; CUDA hardware is optional.

## Downstream

- **E7-F5** requires the resident session and sidecar views for deterministic
  full-process scheduling.
- **E7-F7** extends the registry/session with explicit fixed-shape transport and
  volume-evolution resources.
- **E7-F8** specializes the opaque RNG checkpoint seam into persistent per-box
  stream identity and restart semantics.
- **E7-F9** consumes checkpoint, transfer-spy, diagnostics, and complete-loop
  evidence for epic closeout.
- Epics H and I depend on a stable resident boundary, but graph capture,
  performance optimization, and autodiff do not enter E7-F4.

Authoritative issue #1451 chain:
`E7-F1 -> E7-F6 -> {E7-F2, E7-F3, E7-F4} -> E7-F5 -> {E7-F7, E7-F8} -> E7-F9`.

## Phase Ordering

P1 freezes ownership and lifecycle before P2 performs conversion. P3 adds
resources to the established dimensions/device contract. P4 exposes guarded
step lifecycle only after state and resources are stable. P5 builds checkpoint
and restart on those invariants. P6 freezes failure/close semantics before P7
publishes documentation. Each production phase includes co-located tests.
