# Documentation Updates

Issue #1521 updated `.opencode/guides/architecture_reference.md` with the
concrete-only resident contract: session-owned P1 metadata; one coagulation-only
sidecar initialized on first acquisition and retained by identity; resident
Brownian dispatch forced to `initialize_rng=False`; and checkpoint/finalize
rejection after sidecar publication. The documentation explicitly excludes
wall-loss RNG integration, reset/inspection APIs, hidden transfer/synchronization,
public exports, persistence, and restart continuation.
