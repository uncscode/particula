## Dependency Map

### Primary Ordering

1. **E2-F1 is foundational.** Schema ownership and shape conventions should be
   reviewed before implementation-heavy child plans finalize APIs.
2. **E2-F2 depends on E2-F1.** `EnvironmentData` should implement the approved
   environment ownership and validation rules.
3. **E2-F3 depends on E2-F2.** Warp environment transfers require a stable CPU
   container schema.
4. **E2-F4 can proceed after E2-F1,** but should coordinate with E2-F2/F3 on
   saturation and vapor-pressure ownership so gas and environment boundaries do
   not conflict.
5. **E2-F5 depends on E2-F2 and E2-F3.** Kernel migration needs normalized
   environment arrays and transfer helpers.
6. **E2-F6 can run after E2-F1, and E2-F7 should stage behind E2-F2 plus the
   `fp64` baseline from E2-F6.** E2-F7 may start stress-case design in parallel,
   but its recommendation phase should not finalize before environment-shape and
   precision-baseline inputs are written down.
7. **E2-F8 depends on E2-F1 and should incorporate findings from E2-F5.** It
   documents boundaries exposed by the migration path without redefining schema
   ownership established earlier in the epic.
8. **E2-F9 is last.** It consolidates shipped behavior, limitations, examples,
   and downstream roadmap guidance after E2-F2 through E2-F5 and E2-F8 settle
   the user-visible contracts.

### Cross-Cutting Dependencies

- `particula/particles/particle_data.py` and `particula/gas/gas_data.py` define
  existing CPU container patterns that all child tracks should reuse.
- `particula/gpu/warp_types.py` and `particula/gpu/conversion.py` are shared by
  E2-F3, E2-F4, and E2-F5.
- GPU condensation and coagulation kernels are shared compatibility surfaces for
  E2-F5 and evidence sources for E2-F7.
- Documentation updates should reference `docs/Features/particle-data-migration.md`
  and `docs/Features/Roadmap/data-oriented-gpu.md`.
