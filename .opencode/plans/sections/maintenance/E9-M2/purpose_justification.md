# Purpose and Justification

M2 supplies the data-native construction boundary needed by the v0.3 migration:
Aerosol holds ParticleData, GasData and EnvironmentData directly. Replacements
validate the full candidate triple before publication. Useful initializers
preserve defaults and scientific quantities without constructing facades.

Independent radius/PDF arithmetic, resolved V=.25/1/4 inventories, full gas
ordering and identity/nonmutation tests guard against normalization errors and
partial replacement. Algorithms remain process-owned. Legacy consumers use an
explicit temporary module until M3–M5 migration; M6 owns removal.

M1 is merged and its handoff is approved. This is an implementation record,
not a release claim. M3 remains gated by completed final validation and
maintainer M2 handoff acceptance.
