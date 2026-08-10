# Documentation Updates

Issue #1521 updated `.opencode/guides/architecture_reference.md` with the
concrete-only resident contract: session-owned P1 metadata; one coagulation-only
sidecar initialized on first acquisition and retained by identity; resident
Brownian dispatch forced to `initialize_rng=False`; and checkpoint/finalize
rejection after sidecar publication.

Issue #1522 shipped the corresponding wall-loss implementation contract in the
concrete execution modules and regression suite: an independent canonical-manifest
sidecar, exact resource identity/nonaliasing, scheduler-resolved selected-box
gating, and unchanged disabled/prelaunch-skipped/zero-time/no-work lanes. P3 did
not add a reset/inspection API, direct-kernel API or physics change, hidden
transfer/synchronization, public export, persistence, or restart continuation;
the broader documentation phase remains P7.
