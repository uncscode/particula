# E9-M2 final implementation and handoff evidence

## Scope and acceptance

All six planned phases are implemented as sequential stacked PRs. This final
phase adds developer guidance, corrected planning records, the completed
capability/temporary-consumer ledgers, and eight data-only integration cases.
P1–P5 each passed local focused, full-package and lint/type gates before the
next phase began. See their phase records for exact predecessor commits.

The final implementation revision and literal final validation results will be
recorded below after the committed-revision run. Current preliminary P6 checks:
eight integration cases passed; lint/type and both strict docs commands passed;
both developer-reference Python snippets executed successfully.

**Final M2 acceptance is pending Kyle Gorkowski.** Metadata remains In Progress
for review and final sign-off; this does not mean construction implementation is
missing. No PR merge, release, M3 admission or later-track migration is claimed.

## Phase PRs

| Phase | PR | Implementation commit |
|---|---|---|
| P1 | https://github.com/uncscode/particula/pull/1625 | d5ec8ffb2 |
| P2 | https://github.com/uncscode/particula/pull/1626 | 19ed66933 |
| P3 | https://github.com/uncscode/particula/pull/1627 | 378aaa2c0 |
| P4 | https://github.com/uncscode/particula/pull/1628 | e24cb7990 |
| P5 | https://github.com/uncscode/particula/pull/1629 | 5e042230c |
| P6 | Pending final validation/publication | Recorded after commit |

## Protected boundaries

M1's structural and representation validators are reused without redesign.
Binned distributions remain density-based at V=1; only resolved density inputs
multiply by physical volume. Aggregate construction/replacement retains supplied
identities; initializers allocate owned arrays. Species mapping stays process
owned, and unnamed ordering cannot be certified by shape checks.

Legacy scientific/runnable/example consumers retain explicit temporary imports.
The exact inventory and M6 removal conditions are in `legacy_consumers.md`.
Tagged particle upload remains rejected because the Warp metadata carrier is
absent; untagged transfer, device ownership and resident identity remain
unchanged. No GPU normalization or checkpoint support is claimed.

## Validation policy

Use repository `.venv` with host GPU/cache access for GPU-bearing tests. Record
actual available devices and skips; no skip is a measured CUDA result. Focused
checks use `--no-cov`; only the untargeted repository runner supplies coverage
with its configured threshold. Explicit wall-loss suites, source examples,
infrastructure/helper packaging, lint/type and strict docs remain required.

Notebook import-routing changes are synchronized from Python with Jupytext,
with original outputs preserved. Full tutorial workloads and refreshed outputs
remain M5-owned; executed source example suites are recorded separately.
