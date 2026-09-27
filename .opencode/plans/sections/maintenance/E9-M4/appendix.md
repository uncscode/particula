# Appendix: Audit and Handoff Support

## Evidence map

| Boundary | Current evidence anchor | Owning phase |
|---|---|---|
| Full cycle per equal sequence substep | `runnable.py:177–218` | P1 |
| Legacy scientific runnable state access | `particle_process.py:502–557,600–657,691–743` | P1 |
| Two legacy dilution gas groups | `dilution.py:310–428` | P2 |
| Restore and re-raise dilution failures | `dilution.py:504–585` | P2 |
| Facade topology and duplicate environment | `particle_process.py:209–306` | P3 |
| Current-gas sequential commit loop | `particle_process.py:375–455` | P3 |
| CPU dispatch carriers | `execution/adapters/condensation.py`, `coagulation.py` | P4 |
| Validation policy and excluded wall-loss suite | `.opencode/guides/testing_guide.md`, `pyproject.toml:95–110` | All |

Line references describe drafting checkout, not immutable implementation ranges.
Source context: issue #1602; E9 outcomes/guardrails; E9-M1 success criteria;
E9-M2 ownership requirements; E9-M3 scientific phases and closeout ledger.

## Ledgers to maintain during implementation

- Consumer ledger: path/symbol, old state authority, approved native authority,
  units/shape, phase, test node and removal disposition.
- Behavioral ledger: original assertion, new fixture/node, independent oracle,
  tolerance and failure boundary. No enduring scientific assertion is silently
  deleted because its setup uses a facade.
- Compatibility ledger: seam, remaining consumers, reason, owner, M6 removal
  gate. Native calls must not reconstruct facades. Empty ledger is valid only
  after an explicit audit, not by assumption.
- Validation ledger: revision/date, environment/Warp availability, exact command,
  exit/result, literal evidence pointer, skips and reviewer decision. Initial
  state: implementation commands not run; no results claimed.

M5 receives tested native invocation patterns and environment-signature choices.
M6 receives residual legacy consumers and temporary seams, not an instruction
to delete retained CPU↔Warp transfer helpers or scientific regression coverage.
