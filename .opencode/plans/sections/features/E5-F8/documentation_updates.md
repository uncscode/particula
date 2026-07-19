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

Issue #1370 completed P4 integration in
`docs/Features/condensation_strategy_system.md`,
`docs/Features/data-containers-and-gpu-foundations.md`,
`docs/Examples/index.md`, `docs/Features/Roadmap/index.md`, and
`docs/Features/Roadmap/data-oriented-gpu.md`. Each page links exactly once to
the walkthrough and ownership record, preserves the fixed-four-substep
low-level direct-kernel boundary, and identifies separate `physics`,
`conservation`, and `energy` evidence. The pages document Warp CPU as the
installed-Warp baseline and CUDA as optional additive evidence; they also state
`kg * J/kg = J` and retain `energy_transfer` as caller-owned, write-only
diagnostic output rather than a return value or temperature-feedback mechanism.
