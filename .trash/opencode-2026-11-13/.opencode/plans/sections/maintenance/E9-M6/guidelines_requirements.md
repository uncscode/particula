# Guidelines and Requirements

## Functional requirements

1. Start only after reviewed M5 completion and validation, including its native
   example inventory, paired execution and strict docs results, with accepted
   M1–M4 handoffs. Upstream contract drift reopens affected gates.
2. P1 inventories every legacy consumer and temporary seam, its owner,
   replacement, enduring tests and removal proof. Unexpected supported legacy
   consumers block deletion and return to the owning migration track; do not
   silently add a facade shim or complete missing upstream migrations in M6.
3. Remove obsolete definitions and all import/export aliases together. Test
   absence at root, package and former concrete-module boundaries plus absence
   of fallback branches. Removal means no deprecated-but-working substitute.
4. Classify tests by assertion, not filename. Preserve units, non-unit volume,
   species mapping, per-species conservation, identity, rejection and partial
   failure regressions on native fixtures. Delete only facade mechanics whose
   contract is intentionally removed, with a reviewed ledger explanation.
5. Preserve M1's declared distribution normalization and ordered species map;
   mixed/nonpartitioning gas remains represented. Process configuration stays
   process-owned. CPU processes remain single-box. Native getters and accepted
   replacements preserve supplied identity; failed individual/coordinated
   replacement leaves held and candidate state unchanged.
6. Preserve supported CPU-Warp conversion helpers, explicit transfer/device/
   synchronization ownership, gas-name ordering and lossy CPU inspection
   semantics. CPU replacement never automatically rebinds prepared/resident
   state. Retain RNG, checkpoint/restart and graph-capture lifecycle boundaries.
7. Reconcile active docs, notebooks, API navigation, error messages and examples
   against final imports. Historical references require explicit version and
   unsupported/non-executable labels; exclude them narrowly, not by directory.

## Quality bars

- Every changed function ships co-located `*_test.py` tests in its PR. Removal
  PRs include negative boundary tests and positive native regression evidence.
- Independent analytical/NumPy oracles, explicit deterministic tolerances,
  tight conservation checks and separate stochastic aggregate criteria.
- Ruff 80-column formatting, type hints and Google-style docstrings for retained
  changed APIs. Required lint/type and documentation checks remain intact.
- Record commands, revision, date, environment, literal output and disposition.
  A missing required run is blocked evidence, not a passing implementation.

## Constraints

P1–P6 are serial and issue-sized; co-located testing is not deferred to P5.
M6 is removal-last, not an excuse to remove tests to recover a green suite.
Do not weaken thresholds, use focused coverage as evidence, broaden exports or
resolve unrelated historical GPU measurement gaps. Roll back a faulty removal
PR and revalidate rather than reinstate a permanent compatibility facade.
