# Dependencies

**Parent:** E9. **Track:** T1. **Upstream maintenance prerequisite:** None.
Start requires epic scope approval and access to the existing v0.2.13 baseline
containers, scientific tests and process-owned configuration contracts.
No new external dependency is required. Earlier GPU epics are protected
context, not parentage or new profiling prerequisites; Epic I follows E9.

## Required order

`E9-M1 -> E9-M2 -> E9-M3 -> E9-M4 -> E9-M5 -> E9-M6`

Every arrow means **completed and validated**. Drafted plans, merged changes
with pending tests, or provisional decisions do not satisfy a gate. No parallel
track implementation is allowed. Internally P1 → P2 → P3 → P4 → P5 follows
the same evidence-first ordering.

The revised P1 scientific gate is not satisfied by earlier raw-count/raw-PDF
characterization or the later issue #1612 clarification alone: reconcile
fixed-V=1 PMF/PDF versus variable-V resolved units, explicit PDF quadrature,
builder validation and post-mutation interpretation, then record independent
evidence and Kyle's P1 approval. Only then may P2 specify and validate its
complete replacement/ownership matrix; P3 requires separate explicit P2
approval at a recorded revision. Neither gate has been promoted here.

| Consumer | M1 output consumed | Boundary retained |
|---|---|---|
| E9-M2 | Approved replacement specification, units/order/ownership contract and tested helpers | M2 implements flat construction and actual replacement |
| E9-M3 | Concentration interpretation and process-owned configuration mapping | M3 migrates scientific consumers only after M2 |
| E9-M4 | Unified mixed-gas semantics, mutation/revalidation rules | M4 owns runnables, dilution, nucleation and adapters after M3 |
| E9-M5 | Delivered API and ownership reference | Broad examples/notebooks/docs start after M4 |
| E9-M6 | Helper/compatibility inventory and enduring behavioral evidence | Deletion last, only after M5 |

M1 handoff requires the final revision, resolved blocking decisions, complete
helper/acceptance matrices, required test/lint/docs evidence and maintainer
approval. Contract changes reopen affected gates; do not silently update
downstream assumptions. This drafter does not modify sibling dependency records.
