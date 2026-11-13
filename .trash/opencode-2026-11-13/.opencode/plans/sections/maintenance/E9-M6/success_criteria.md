# Success Criteria

Implementation validation is complete. Final maintainer scientific/release
acceptance remains pending and is not inferred from passing tests or PR creation.
The authoritative result details are in `implementation.md`, the assertion
ledgers, `example_execution.json`, `notebook_output_review.json`, and
`docs/Features/Roadmap/data-native-v030-closeout.md`.

- [x] Enter M6 from maintainer-reported completed M5 and merged base
  `fb25078b6c0e8136ed06feeda59f0928b85c1b34`.
- [x] Final audit identifies no supported executable legacy consumers; remaining
  removed names are intentional negative guards, migration history or filenames.
- [x] ParticleRepresentation, GasSpecies, Atmosphere, obsolete construction,
  bridges and temporary CPU/GPU compatibility paths are removed.
- [x] Removed root/package/concrete-module imports reject intentionally;
  native construction, replacement and composition pass, including installed wheel.
- [x] Enduring scientific assertions have native successors or explicit justified
  dispositions; 1,606 successor references resolve and regression suites pass.
- [x] Both CPU wall-loss strategy suites pass independently (16 and 43 tests).
- [x] Required CPU-Warp transfer, ownership, direct-kernel and resident checks
  pass; unavailable native CUDA is explicitly distinguished from Warp CPU.
- [x] All supported M5 examples plus native state access are accounted for:
  58 source successes, one optional native-CUDA skip, and 46 executed notebooks.
  All 98 PNG outputs are reviewed and source/notebook ASTs match.
- [x] Untargeted normal coverage passes at 92% without lowering the 80% threshold;
  required lint/type, source-example, infrastructure, plugin and publication
  checks pass against final production/example content.
- [x] Removals, native replacements, explicit ownership and retained scientific
  limitations are documented; no version/schema bump or release is introduced.
- [ ] Kyle/Gorkowski explicitly accepts final scientific and release readiness.

| Metric | Final implementation evidence |
| --- | --- |
| Supported legacy consumers and temporary seams | Zero supported remaining paths; reconciled removal inventory |
| Removed API surfaces | Negative source and installed-wheel tests plus positive native execution |
| Scientific assertion retention | Original bodies/dispositions and native successors in adjacent ledgers |
| CPU wall loss | Both explicit suites pass |
| Examples and notebooks | Complete execution and hash-qualified plot review records |
| GPU/resident contracts | Warp CPU evidence and separately qualified optional CUDA skips |
| Full coverage | 8,333 passed, 684 skipped; 92% at unchanged threshold |
| Installed wheel | 4,559 passed, 25 skipped, 820 deselected; separate examples 20 passed |
| Documentation | Standard and full strict publication pass without warning suppression |
| Release readiness | Awaiting explicit maintainer review; no release publication |

Optional hardware skips are not measured success. Historical failed attempts
remain in the implementation record; final successful reruns supersede them
only for the same validated scope.
