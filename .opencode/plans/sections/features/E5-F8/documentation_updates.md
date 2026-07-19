# Documentation Updates

Implemented `docs/Examples/gpu_condensation_parity_walkthrough.py` as the
standalone deterministic Warp-CPU-default walkthrough. Its module documentation
states the direct-kernel-only boundary, explicit transfers, no-Warp result,
focused commands, caller-owned mutable outputs, and recovery after a failed
enabled call.

Issue #1368 extended the walkthrough's self-documenting result interface and
emitted `physics`, `conservation`, and `energy` labels, including explicit
no-Warp `unavailable` outcomes. No roadmap report, documentation index, or
canonical-feature-document updates were made for #1367/#1368. Those broader
documentation and deferred-capability ownership changes remain future E5-F8
work.

The example treats `energy_transfer` as caller-owned output and avoids claims of
temperature feedback, general CPU strategy parity, graph replay, broad
autodiff, adaptive stepping, performance, or required CUDA support.
