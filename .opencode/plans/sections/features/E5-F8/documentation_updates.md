# Documentation Updates

Implemented `docs/Examples/gpu_condensation_parity_walkthrough.py` as the
standalone deterministic Warp-CPU-default walkthrough. Its module documentation
states the direct-kernel-only boundary, explicit transfers, no-Warp result,
focused commands, caller-owned mutable outputs, and recovery after a failed
enabled call.

Issue #1368 extended the walkthrough's self-documenting result interface and
emitted `physics`, `conservation`, and `energy` labels, including explicit
no-Warp `unavailable` outcomes. Issue #1369 added
`docs/Features/Roadmap/condensation-parity-walkthrough.md` as the P3 ownership
record and `particula/tests/condensation_parity_walkthrough_docs_test.py` to
validate its 14 deferred-capability routes, anchors, links, and focused commands.

The ownership record preserves the bounded direct-kernel versus independent
fixed-four-substep NumPy-oracle evidence and routes deferred work to future
approved numerical-method or physics-expansion work and Epics G--I. It makes no
claim of high-level CPU strategy, `Runnable`, or general CPU workflow parity.
P4 canonical inbound links and indexes remain pending.
