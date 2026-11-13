# Documentation Updates

Issue #1521 updated `.opencode/guides/architecture_reference.md` with the
concrete-only resident contract: session-owned P1 metadata; one coagulation-only
sidecar initialized on first acquisition and retained by identity; resident
Brownian dispatch forced to `initialize_rng=False`.

Issue #1522 shipped the corresponding wall-loss implementation contract in the
concrete execution modules and regression suite: an independent canonical-manifest
sidecar, exact resource identity/nonaliasing, scheduler-resolved selected-box
gating, and unchanged disabled/prelaunch-skipped/zero-time/no-work lanes.

Issue #1523 updated `.opencode/guides/architecture_reference.md`,
`.opencode/guides/architecture/architecture_outline.md`, and
`.opencode/guides/testing_guide.md` for the concrete-only P4 boundary:
immutable metadata inspection; explicit selected reset; exact ACTIVE,
session/registry/closed-guard binding; published-sidecar-only scope; selector
preflight; and no ordinary-dispatch reset, readback, or synchronization.

Issue #1525 updated `.opencode/guides/architecture_reference.md`,
`.opencode/guides/architecture/architecture_outline.md`,
`.opencode/guides/architecture/architecture_guide.md`,
`.opencode/guides/architecture/decisions/ADR-007-resident-session-checkpoint-finalize-restart.md`,
`.opencode/guides/testing_guide.md`, and
`docs/Features/gpu_resident_checkpoints.md`. They now document schema-v3
optional continuation, one capture synchronization/readback boundary, fresh
exact-device reconstruction, current-word authority, explicit-only reset,
v1/v2 compatibility, normal-dispatch no-readback/no-reseed behavior, and
construction cleanup/no-post-writer-rollback boundaries. Public exports and
direct-kernel APIs remain unchanged.

Issue #1526 completed P7 documentation evidence in the foundations, checkpoint,
roadmap, and contributor references. It records stable logical IDs and separate
namespaces, one-time resident acquisition, metadata-only inspection, explicit
reset, non-reseeding dispatch, schema-v3 current-word continuation, and bounded
fresh exact-device restart without adding a public API or runtime implementation.
