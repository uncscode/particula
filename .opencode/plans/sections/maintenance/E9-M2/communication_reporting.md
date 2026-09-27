# Communication and Reporting

- Report phase entry/completion and blockers in the owning PR and issue
  #1602/E9 handoff record. Notify the maintainer, scientific reviewer and
  downstream M3/M4 owners. Named owners remain unassigned until maintainer review.
- Before P1, obtain explicit M1 gate evidence and transitional seam approval.
  Before P4/P5, review the retained construction capability inventory, native
  module/API placement and sampling/default semantics.
- Each implementation PR records changed construction functions, old-to-native
  behavioral-test mapping, exact assertion commands/results, units/identity
  decisions and any temporary compatibility addition. No new diagnostics track.
- Review metrics and open questions at every phase gate. Escalate M1 contract
  gaps and M3/M4 consumer incompatibilities instead of silently expanding M2.
  A scope change must preserve strict completed-and-validated track order.
- P6 publishes revision/date, required validation outputs, optional device
  availability/skips, compatibility removal owners and explicit M3 admission
  approval. Required unavailable checks are blockers, not passing evidence.
- Reconcile parent/sibling records through their owner; this drafting task
  edits E9-M2 only. Record significant decisions in this plan's change log.
  There is no fixed calendar deadline or fabricated progress percentage.
