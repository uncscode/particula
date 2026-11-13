# Dependencies

## Upstream

- **E8 parent:** fixes stable shapes, process order, communication maps, explicit
  setup/replay/teardown, no hidden operations, and three-way validation as the
  governing contract.
- **E8-F1:** supplies capture capability, compatibility signature, lifecycle,
  canonical invalidation reasons, resident fault integration, and explicit
  recapture eligibility.
- **E8-F2:** supplies the complete immutable prepared resident plan and private
  device-only enqueue sequence shared with uncaptured execution.
- **E8-F3:** supplies the atomically published, complete, nonaliasing,
  identity-stable capture resource set and associated requirement binding.
- **E7 resident execution:** `gpu_session`, `gpu_resources`,
  `resident_scheduler`, `resident_communication`, `diagnostics`, and
  `checkpoint` provide session ownership, token, failure, process-order,
  continuation, and terminal lifecycle semantics.
- **E6 fixed-capacity process seams:** activation/exhaustion allow active
  populations to change without structural resizing.
- **Warp capture APIs:** production capture requires `capture_begin`,
  `capture_end`, and `capture_launch` on a qualified CUDA device.

## Downstream / Sibling Features

- **E8-F5:** measures box-first and particles-per-box scaling against this exact
  captured replay implementation.
- **E8-F6:** models graph-lifetime resources and memory using the finalized
  ownership and teardown contract.
- **E8-F7:** profiles the correctness-qualified replay path and publishes
  machine-bounded performance evidence.
- **E8-F8:** publishes the runnable graph-capture example, limitations,
  recapture runbook, and final closeout; it must not redefine E8-F4 launch or
  lifecycle semantics.

## External Dependencies

- Existing `warp-lang` runtime; no new runtime package is required.
- Existing pytest, pytest-cov, Ruff, mypy, and MkDocs development tools.

## Phase Ordering

P1 establishes the graph owner and native API boundary. P2 can then capture the
prepared sequence. P3 launches only the handle produced by P2. P4 integrates
invalidation, fault, teardown, and recapture after replay semantics are stable.
P5 supplies full-loop evidence and documentation last. Unit tests ship with
P1-P4; P5 contains integration and documentation validation rather than a
standalone testing-only phase.
