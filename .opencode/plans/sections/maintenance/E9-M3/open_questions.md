# Open Questions

- [ ] Which final M1/M2 revisions and approval records establish entry?
  Owner: E9/M1/M2 maintainers. Resolve before P1; consume their species mapping
  and count-versus-density convention instead of inventing local contracts.
- [ ] What complete supported strategy/distribution/configuration matrix is
  approved, including latent heat and turbulent-DNS? Owner: scientific
  reviewers. Seed from all P1–P5 existing suites; freeze each family before its
  migration. Exclusions require rationale, not merely legacy fixture usage.
- [ ] Which minimal temporary seams keep pre-M4 legacy consumers working?
  Owner: M2/M3/M4 maintainers. Prefer existing seams; any addition requires
  named consumers, no competing state and M6 deletion ownership.
- [ ] Do audited native/legacy normalization, binning or mutation results
  disagree with the approved M1 physical contract? Owner: scientific reviewer.
  Audit non-unit volumes/unequal weights first. Isolate confirmed discrepancies
  for review rather than blessing both branches or introducing new physics.
  Unresolved differences block the affected phase.
- [ ] Who signs M3 scientific completion/validation and accepts M4 handoff?
  Owner: E9 maintainer. Name reviewers before implementation; v0.3.0 does not
  imply a fixed calendar deadline.

Settled by #1602: single-box CPU, process-owned physics, unchanged GPU
boundaries, BOTH wall-loss suites, tests with implementation, serial track
gates and M6-last removal are not open choices. Record evidence-backed answers
here as review resolves questions.
