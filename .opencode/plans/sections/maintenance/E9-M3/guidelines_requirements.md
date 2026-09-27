# Guidelines and Requirements

## Functional requirements

1. Start only after M2 is completed and validated with M1 approved mapping and
   distribution-specific normalization. Drafted upstream plans are not gate
   evidence. Block rather than invent unresolved scientific contracts.
2. Native execution must not construct a facade or obtain strategies from
   state. Activity, surface, vapor pressure, distribution selection, collision
   mechanisms, geometry and random controls belong to processes. Preserve
   ordered partitioning/nonpartitioning gas in one `GasData`: no silent mask
   compaction, name sorting or alignment inferred from equal widths.
3. Preserve kg per particle/species, gas kg/m^3, radius m, volume m^3, K, Pa
   and seconds. Use M1 physical number density c: c=w/V for weight/count
   storage or c=stored density for density storage. Never divide twice.
   Population mass density is sum_i(m_i,s*c_i), not per-particle total mass.
   Extensive mapped particle-plus-gas inventory is
   V*(sum_i(m_i,s*c_i)+gas_concentration_s). PDF modes use their approved
   integration measure rather than an unweighted sum.
4. Cover V=0.25/1/4 m^3, unequal weights and multiple species. Assert each
   gas/particle output separately. Preserve skipped/nonpartitioning lanes.
   Treat `update_gases=False` as its specified reservoir behavior, not a
   closed-system conservation claim.
5. Preserve current formulas, limiting/clipping, stepping/theta modes,
   latent-heat behavior and stochastic controls. Coagulation conserves weighted
   mass under documented discretization tolerance, not particle number. Wall
   loss is a sink: compare remaining plus independently accounted removed
   inventory rather than asserting constant remaining particle mass.
6. Require n_boxes=1 before CPU scientific writes, including directly mutated
   containers. Preflight malformed shape/species/configuration before mutation;
   snapshot inputs on rejection. Preserve return identity and documented
   failure semantics; do not promise new global rollback or silently impose
   GPU slot semantics on CPU behavior.
7. Native fixtures are facade-free; retained compatibility has named consumers
   and M6 deletion ownership. CPU replacement never automatically rebinds
   prepared GPU/resident state. Keep transfer/export/ownership boundaries.

## Quality bars

- Each modified function ships adjacent `*_test.py` tests in the same PR.
- Independent analytical/NumPy oracles, explicit scientific tolerances and a
  legacy-assertion-to-native-assertion ledger. Legacy parity alone is not an
  enduring oracle; do not delete behavior because its fixture used a facade.
- Typed APIs, Google-style units/shape/mutation docstrings, 80-column Ruff
  conventions and required repository lint/type checks. No new dependencies.

## Constraints

Serial P1 through P6 and completed-and-validated M1→M2→M3→M4→M5→M6.
No lowered thresholds/tolerances. Focused checks are coverage-disabled
assertions; comprehensive evidence uses the untargeted repository runner with
configured full-package coverage and the normal threshold.
