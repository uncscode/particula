# Purpose and Justification

E9-M1 is T1 of parent E9, the v0.3.0 data-native migration in
[issue #1602](https://github.com/uncscode/particula/issues/1602). It establishes
the scientific and ownership contracts that construction and later consumers
must use. This is a Draft plan; no implementation or validation is complete.

Legacy facades provide behavior, not just storage. In particular,
`ParticleRepresentation.get_concentration()` divides stored concentration by
volume (`particula/particles/representation.py:566–581`). Meanwhile,
`ParticleData` documents particle-resolved counts and binned number density.
Blind replacement with raw fields can introduce missing or double volume
normalization. Equal array widths also do not prove species identity.

T1 therefore freezes ordered gas/particle/environment/configuration alignment,
including nonpartitioning gas, distinguishes per-particle mass from population
inventory, and specifies identity-preserving whole-container replacement.
Only demonstrated gaps receive data-native helpers and adjacent tests; no new
facade, physics owner, or duplicate authoritative state is introduced.

Neglecting this work risks incorrect conservation, silently reordered or lost
gas species, partial replacement, and stale prepared GPU bindings. M2 must not
start until M1 is completed and validated. M2 owns aggregate construction and
replacement implementation; M3–M6 migrate consumers and delete legacy APIs last.
