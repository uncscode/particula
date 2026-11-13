# Dependencies

## Parent and Track Relationships

- **Parent:** E6, GPU Process Completeness.
- **Issue track:** T1, mapped authoritatively to E6-F1.
- **Inbound feature dependencies:** None. E6-F1 may proceed in parallel with
  E6-F3 and E6-F5.
- **Outbound dependencies:** E6-F2 requires this CPU reference for direct GPU
  dilution parity. E6-F9 requires E6-F1 through E6-F8 for integrated validation
  and closeout.
- **Sibling context:** E6-F3/F4 cover wall loss; E6-F5/F6 cover fixed-slot
  management; E6-F7/F8 cover nucleation. Their behavior must not be folded into
  this feature.

## Internal Code Dependencies

- Existing free functions in `particula/dynamics/dilution.py`.
- `Aerosol`, `ParticleRepresentation`, `GasSpecies`, `RunnableABC`, and
  `RunnableSequence` public contracts.
- Existing concentration accessors/setters and representation-volume semantics.
- Validation conventions in `particula.util.validate_inputs`.
- Existing wall-loss runnable tests as process/substep prior art.
- Shipped E5 is contextual prior art for downstream GPU work but is not an
  implementation prerequisite for this CPU-only feature.

## External Dependencies

- NumPy, already a core dependency, for scalar/array numerical behavior.
- Python 3.12+ and the existing pytest/Ruff/mypy development toolchain.
- No new package, service, Warp, or CUDA dependency is introduced by E6-F1.
  Warp CPU and optional CUDA evidence apply to downstream E6-F2, not this plan.

## Downstream Contract to Preserve

E6-F2 must be able to reuse the equation, units, validation ordering, no-op
semantics, protected-field invariants, and deterministic fixtures unchanged.
Any later change to that contract requires coordinated review of E6-F2 parity
tests and E6-F9 integrated documentation.
