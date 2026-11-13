## Governance

### Decision Process

- Schema ownership decisions from E2-F1 are binding inputs for E2-F2 through
  E2-F9 unless later child plans explicitly reopen them.
- Any public API compatibility change must document migration behavior,
  deprecation risk, and tests that prove old call sites still work or fail with
  clear errors.
- Numerical recommendations from E2-F6 and E2-F7 should distinguish measured
  evidence, assumptions, and downstream hypotheses.

### Review Requirements

- Container changes require review for shape conventions, dtype choices,
  validation messages, and copy/round-trip semantics.
- GPU changes require review for Warp availability gating, CPU-device test
  coverage, and scalar compatibility.
- Documentation changes require review for consistency with shipped behavior,
  not aspirational roadmap claims.

### Ownership

- Child feature implementers own co-located tests and documentation updates for
  their track.
- The epic owner owns cross-track consistency, dependency sequencing, and final
  handoff quality.
- Downstream roadmap owners should consume E2 outputs only after E2-F9 publishes
  the consolidated foundation docs.
