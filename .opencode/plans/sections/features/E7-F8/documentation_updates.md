# Documentation Updates

Issue #1521 updated `.opencode/guides/architecture_reference.md` with the
concrete-only resident contract: session-owned P1 metadata; one coagulation-only
sidecar initialized on first acquisition and retained by identity; resident
Brownian dispatch forced to `initialize_rng=False`; and checkpoint/finalize
rejection after sidecar publication.

Issue #1522 shipped the corresponding wall-loss implementation contract in the
concrete execution modules and regression suite: an independent canonical-manifest
sidecar, exact resource identity/nonaliasing, scheduler-resolved selected-box
gating, and unchanged disabled/prelaunch-skipped/zero-time/no-work lanes.

Issue #1523 updated `.opencode/guides/architecture_reference.md`,
`.opencode/guides/architecture/architecture_outline.md`, and
`.opencode/guides/testing_guide.md` for the concrete-only P4 boundary:
immutable metadata inspection; explicit selected reset; exact ACTIVE,
session/registry/closed-guard binding; published-sidecar-only scope; selector
preflight; no ordinary-dispatch reset, readback, or synchronization; and no
checkpoint/restart persistence. `architecture_guide.md` still describes
reset/inspection as deferred and must be reconciled in P7. Public exports and
direct-kernel APIs remain unchanged.
