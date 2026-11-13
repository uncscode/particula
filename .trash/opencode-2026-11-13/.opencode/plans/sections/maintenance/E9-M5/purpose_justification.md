# Purpose and Justification

E9-M5 implements T5 of issue #1602 under parent E9: migrate the supported
learning and simulation surface to the v0.3.0 flat `Aerosol` holding
`ParticleData`, `GasData`, and `EnvironmentData`. This is maintenance, not a
new feature or diagnostics track. Implementation starts only after E9-M4 is
completed and validated; E9-M6 removes legacy APIs only after this track passes.

Adding one quick start while existing tutorials still construct facades would
leave users unable to reproduce scientific examples after removal. Mechanical
getter replacement also risks double volume normalization, species permutation,
or confusing per-particle mass with population inventory. M5 therefore owns a
complete supported-example inventory, scientific regression-backed migrations,
executed notebook pairs, API/migration guidance, and advanced state replacement
examples. It consumes, rather than redefines, the M1–M4 contracts.

The reliability promise is usable native workflows with preserved scientific
outcomes, runnable composition, CPU single-box admission, explicit CPU/Warp
transfers and unchanged resident ownership. Historical legacy snippets must be
clearly labeled unsupported before/after history, never supported execution.
Drafting this plan does not assert that any migration or validation has run.
