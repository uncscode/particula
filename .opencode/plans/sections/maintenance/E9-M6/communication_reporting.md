# Communication and Reporting

- Report to the E9 orchestrator and issue #1602 maintainers at phase entry,
  each removal PR and phase closeout; named implementation/review owners remain
  to be assigned. No calendar deadline has been imposed.
- Before P1, obtain the accepted M5 handoff and transitive M1–M4 evidence.
  Record exact revisions, not just plan status labels. Notify the owning track
  immediately if a supported legacy consumer or missing native behavior remains.
- Every removal PR lists definitions/exports/branches/messages deleted,
  migrated scientific assertion mappings, remaining temporary seams, historical
  exceptions, positive native tests and negative boundary results.
- Update the appendix-defined audit/evidence ledgers per phase, preserving
  literal command output, final revision, devices, pass/fail/skip/unavailable
  and blocker ownership. Keep focused assertions distinct from full coverage.
- P5 reports CPU science, BOTH wall-loss suites and retained GPU/resident
  contracts separately. Optional CUDA skips do not imply hardware success;
  required missing validation blocks readiness.
- P6 publishes the proposed roadmap closeout and breaking-change guidance,
  links supported replacement examples and pairs, and requests explicit E9
  maintainer readiness review. Report release approval separately; this plan
  does not tag/publish v0.3.0 or close earlier GPU measurement gaps.
- Escalate scope expansion or new physics/API requests instead of silently
  growing M6. Log tooling gaps reactively; preserve accurate partial status.

Draft report: six phases, twelve canonical sections, no implementation tests
run and all execution gates pending. Completion of drafting is not completion
of T6 or evidence that E9-M5 has passed its entry gate.
