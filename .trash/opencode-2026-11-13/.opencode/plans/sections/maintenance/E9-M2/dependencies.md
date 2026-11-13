# Dependencies

M1 → M2 → M3 → M4 → M5 → M6 remains strictly serial, with each implementation
track completed and validated before the next starts. M2's six phases follow
the same local implementation/validation order; the separate PRs are stacked.

M1 handoff was approved on 2026-11-10 and PR #1624 merged at `229bdb4a1`.
M2 began from that merged baseline. Its phase evidence links the exact
predecessor commit and PR. M2 does not implement any later track.

- M3 owns scientific process admission, mapping and numerical migration.
- M4 owns runnable/adapters, dilution/nucleation orchestration and resident work.
- M5 owns supported examples/notebooks and broad migration documentation.
- M6 removes temporary legacy APIs only after named consumers and enduring
  behavioral tests migrate, then owns release-readiness validation.

Existing native builders, structural/representation validators and distribution
generators are reused. No new package or hardware prerequisite is introduced.
Explicit untagged CPU/Warp transfer remains protected; new tagged data still
rejects unsupported upload. No checkpoint or GPU normalization claim is made.

Final local checks and maintainer acceptance in `validation_evidence.md` admit
M3. Open PRs do not by themselves supply that acceptance.
