# Open Questions

- [ ] Which final M1–M4 revisions freeze accessor spelling, species mapping,
  normalization, preset inventory and nucleation environment ownership?
  - Owner: upstream reviewers; resolve before P1. Recommend consuming their
    approved contracts verbatim rather than documenting speculative APIs.
- [ ] Who approves the exhaustive supported-example/snippet inventory and any
  historical-only exceptions, including older tutorial filenames?
  - Owner: E9 documentation maintainer; P1 gate. Recommend retaining all
    currently published supported scientific content with native equivalents;
    renames require navigation updates, not silent removal.
- [ ] What environment/resources and execution budget are required for every
  full simulation notebook and optional scientific dependency?
  - Owner: example maintainers; inventory at P1, resolve before owning phase.
    Recommend full published-workload execution plus bounded regression
    fixtures; reduced fixtures alone do not authorize handoff.
- [ ] Which stable numerical outputs and tolerances should be recorded for
  examples currently checked only by plots or narrative?
  - Owner: scientific reviewers; resolve per family before migration using
    M3/M4 independent references. Do not loosen tolerances to absorb drift.
- [ ] Who signs the final M5 evidence and M6 removal authorization?
  - Owner: E9 maintainer; assign before P7. No fixed calendar date supplied.
    Recommend explicit named acceptance of inventory, execution and docs gates.

Resolved by issue #1602: maintenance-only scope, strict validated serial chain,
legacy deletion last in M6, CPU single-box execution, retained explicit transfer
helpers, no automatic resident rebind and no permanent replacement facade.
