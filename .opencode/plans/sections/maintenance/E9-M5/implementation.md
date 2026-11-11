# M5 single-PR implementation record

## Accepted delivery — 2026-11-11

The maintainer confirmed M4 completion and requested implementation of M5,
followed by one PR. Base: `3393e47fa` (merged upstream PR #1633).
Branch: `codex/e9-m5-native-examples-docs`; target: `uncscode/particula:main`;
issue: #1609. This supersedes per-phase issue generation and separate PRs.

## Serial checkpoints

1. Scope and exhaustive inventory (in progress).
2. P1 foundational tutorials with runtime fixtures and executed pairs.
3. P2 condensation/coagulation and repository helper example.
4. P3 wall loss, dilution and nucleation.
5. P4 composed simulations, customization, Activity and Equilibria.
6. P5 direct-Warp and resident examples.
7. P6 migration, replacement guidance and public quick starts.
8. P7 contributor guidance and final consumer audit.
9. Final validation and M6 handoff.

Each example-changing checkpoint includes its tests and executed notebook
outputs. Preserve existing paths and scientific content. No production physics,
new API, facade workaround, legacy deletion, release or performance claim.
Native capability gaps block the owning migration rather than justify reducing
its scientific workload. M6 acceptance and PR merge remain separate decisions.

## Inventory and evidence

`example_inventory.json` enumerates all 58 Python sources and 46 paired
notebooks at the base. Pending rows are not migration or execution evidence.
Expand the ledger with published snippets, links, quantities, tolerances,
commands and final-revision results during the owning checkpoint. Audit the
whole tree again at closeout; absence of legacy spellings is insufficient.

Use the repository `.venv`, existing notebook sync/execution tools, and current
`examples_tests/` runtime suites. Old plan references to removed documentation
tests do not authorize recreating prose-only assertions. All published workloads
must execute; timeout skips remain incomplete evidence. Required final gates:
untargeted repository coverage, examples and infrastructure suites, both explicit
CPU wall-loss suites, GPU/resident boundary checks, lint/type checks, payload
consistency and strict MkDocs. Preserve thresholds and scientific tolerances.

No implementation tests or notebook executions have passed for M5 yet.
