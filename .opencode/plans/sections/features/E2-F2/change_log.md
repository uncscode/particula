# Change Log

## 2026-07-03

- Drafted feature plan E2-F2 for issue #1172 feature E2-F2.
- Added three implementation phases:
  - E2-F2-P1: fields and validation with unit tests;
  - E2-F2-P2: dataclass exports and copy semantics with tests;
  - E2-F2-P3: documentation of environment-state read/mutation boundaries.
- Incorporated parent epic E2 context, E2-F1 dependency, and codebase research
  covering `GasData`, `ParticleData`, validation patterns, tests, and docs.

## 2026-07-03

- Completeness review clarified the final phase as a development-doc update gate
  so the feature explicitly satisfies the plan final-phase documentation policy.

## 2026-07-03

- Completeness review aligned `success_criteria` with the accepted canonical
  schema: `temperature` and `pressure` stay `(n_boxes,)`,
  `saturation_ratio` stays `(n_boxes, n_species)`, and supersaturation remains
  explicitly allowed.
