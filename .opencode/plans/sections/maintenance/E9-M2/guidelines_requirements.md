# Guidelines and Requirements

## Functional Requirements

1. Require the completed-and-validated M1 handoff: approved species map,
   normalization ledger, minimal API spelling, copy/view rules and reusable
   read-only validators. Unresolved M1 decisions block implementation.
2. `Aerosol` directly holds exactly the three data containers as authoritative
   state. Whole-container reads return the held object; valid construction and
   replacement retain supplied identity. Do not put scientific strategies or
   cached competing gas/environment state on the aggregate.
3. Individual replacement checks the candidate with the other two held
   containers. Validate the entire proposed triple before coordinated
   publication; do not implement the operation as sequential public setters.
   Same-object assignments still obey M1 validation rules.
4. Rejected construction/replacement must not mutate candidates. Rejected
   replacement preserves every old reference, array and metadata value.
   Validation is read-only: no coercing reconstruction of held containers,
   mutation of names, partial reference swaps, or silent repair/reordering.
5. Enforce the M1 ordered alignment of gas names/molar masses/flags,
   particle mass/density lanes, environment lanes and declared process mapping.
   Keep nonpartitioning gas; equal widths alone do not prove semantic order.
   Unnamed-array permutations cannot be detected without declared metadata;
   do not claim otherwise or store process configuration on the aggregate.
6. Preserve M1 distribution-specific concentration and volume conventions.
   Builders normalize physical inputs only as explicitly documented; aggregate
   access/replacement does not normalize, copy or recompute candidate arrays.
   Presets preserve physical distributions, per-species masses and inventories,
   not facade-specific implementation details or strategy-bearing outputs.
7. Reuse native builders and existing distribution utilities. Keep construction
   allocation/defaulting distinct from identity-preserving aggregate wrapping.
   Document defaults, units, shapes, random-state policy and rejection behavior.
8. Direct array mutation remains possible and subject to process revalidation.
   Container batch support is not multi-box CPU execution support. CPU container
   replacement neither rebinds nor refreshes prepared GPU/resident state;
   callers explicitly recreate/reprepare through the existing boundaries.

## Quality Bars

- Co-located `*_test.py` behavioral tests accompany every function change.
  Migrate enduring assertions instead of deleting legacy-fixture tests.
- Use independent numerical expectations, explicit tolerances, volumes 0.25
  and 4 m^3 and distinguishable species sentinel values.
- Public APIs have types, unit/shape/ownership docstrings and 80-column style;
  retain required Ruff/mypy checks and unchanged coverage policy.
- All temporary compatibility is reviewable and removal-owned by E9-M6;
  no permanent facade, new authoritative backing store or hidden GPU work.

## Constraints

Strict serial completed-and-validated M1 → M2 → M3 → M4 → M5 → M6. Internally
P1–P6 also execute serially. Keep each phase one bounded PR, approximately the
template's 100 production-line increment where practical; seek re-scoping
approval if inventory reveals a larger requirement. No new external dependency
or deferred failing-test waiver is authorized to bypass a gate.
