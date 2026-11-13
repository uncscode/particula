# Construction migration

Replace facade builders with native particle initializers or `ParticleDataBuilder`,
full ordered `GasData` (or `GasDataBuilder`), and `EnvironmentData`. Assemble all
three using the keyword-only native `Aerosol` constructor or its builder.
Physics belongs to processes rather than particle/gas facades.

Follow [native construction and replacement](../../data-model/construction.md)
for the executable initializer and replacement examples, units, and distribution
rules. [Historical access replacements](containers.md) cover removed methods;
[dynamics migration](dynamics.md) covers process configuration.
