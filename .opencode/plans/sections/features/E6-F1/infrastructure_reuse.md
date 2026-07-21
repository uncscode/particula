# Infrastructure Reuse

| Existing infrastructure | Reuse in E6-F1 |
|---|---|
| `particula/dynamics/dilution.py` | Retain `get_volume_dilution_coefficient()` and `get_dilution_rate()` as the equation-level source of truth; add validated process semantics without duplicating formulas. |
| `particula/dynamics/tests/dilution_test.py` | Extend scalar/array equation tests with physical validation, no-op, and regression cases. |
| `particula/dynamics/particle_process.py` | Follow `WallLoss`, `MassCondensation`, and `Coagulation` conventions for `RunnableABC`, `execute()`, `rate()`, substeps, in-place aerosol mutation, and `|` composition. |
| `particula/particles/representation.py` | Use `get_concentration()` and the representation's volume-aware concentration storage; never alter mass, charge, density, distribution, or volume. |
| `particula/gas/species.py` | Use `get_concentration()` and `set_concentration()` for scalar or multi-species gas concentrations while retaining names, molar masses, vapor-pressure strategies, and partitioning flags. |
| `particula/util/validate_inputs.py` | Reuse repository validators where their scalar/array contracts fit; add explicit shape/type checks only where necessary. |
| `particula/dynamics/__init__.py` | Publish strategy and runnable alongside existing dilution helpers and process exports. |
| `particula/runnable.py` | Inherit `RunnableABC` and verify composition through `RunnableSequence`. |
| `particula/dynamics/tests/wall_loss_runnable_test.py` | Reuse fixture and assertion patterns for substeps, clamping/no-op behavior, identity, and runnable composition. |

No new external runtime dependency is needed. NumPy remains the CPU numerical
implementation. Existing builders/factories are not required unless repository
API review identifies a concrete construction use case; the issue only requires
a strategy and runnable reference.
