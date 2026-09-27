# Appendix: Evidence and Migration Ledgers

## Observations, not passing test evidence

Scientific paths below are under `particula/dynamics/`.

| Source | Observation | Phase |
|---|---|---|
| `condensation/condensation_strategies.py:287–364` | Native process-owned configuration coexists with legacy strategy lookup | P1 |
| `condensation/tests/condensation_strategies_test.py:121–136` | Native fixture setup derives from legacy conversion | P1–P2 |
| `coagulation/coagulation_strategy/coagulation_strategy_abc.py:49–112` | Native single-box helpers expose radius, mass, volume and raw concentration | P3 |
| Same ABC, native binning and step | Weighted-inventory audit needed, not a new algorithm | P4 |
| `wall_loss/wall_loss_strategies.py:292–339` | Legacy updates include explicit concentration-volume round trip | P5 |
| `pyproject.toml:95–98`; testing guide Wall Loss Coverage | Adjacent wall-loss tests excluded recursively | P5–P6 |

## Implementation ledger

For each enduring assertion record old test/node, physical invariant, native
replacement test/node, units/normalization, species/distribution layout,
tolerance rationale, phase and result/revision. Mark facade-only assertions
for M6 review, not silent deletion. Incomplete rows block the relevant gate;
importing `ParticleData` alone does not prove facade-free setup.

For each temporary seam record symbol/path, current consumer, retention
reason, native successor and removal owner E9-M6. M4 integration is distinct
from M6 deletion. Protected CPU-to-Warp transfers are not obsolete bridges.

## Closeout packet

Record date/revision, accepted M1/M2 evidence, exact commands/literal results,
optional runtime/device skips, both wall-loss results, ledgers and reviewer
approval. Initial status: no implementation evidence yet.

References: https://github.com/uncscode/particula/issues/1602; E9 scope;
E9-M1 guidelines; E9-M2 phase details. Those upstream contracts remain authority.
