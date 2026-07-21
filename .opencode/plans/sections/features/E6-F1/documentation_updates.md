# Documentation Updates

- Update the dilution API documentation to distinguish equation helpers,
  strategy behavior, and the `Dilution` runnable.
- State SI units for volume, inlet flow, coefficient, elapsed time, particle
  number concentration, and gas mass concentration.
- Document the canonical finite-step equation selected in P1, substep behavior,
  input validation, exact no-ops, nonnegative-output policy, and mutation
  ordering.
- Explicitly list invariants: particle mass, charge, density, distribution and
  volume do not change; gas metadata and atmospheric temperature/pressure do
  not change.
- Add a runnable CPU example under `docs/Examples/` that demonstrates particle
  and gas dilution and can be executed during validation.
- Update relevant `docs/Examples/index.md` and feature/API indexes if present.
- Cross-reference `docs/Features/Roadmap/data-oriented-gpu.md`, parent E6, and
  E6-F2 as the downstream direct GPU parity feature.
- Clearly state unsupported behavior: no inlet composition/source, no
  multi-box transport, no direct GPU kernel, no backend selection, and no
  performance claim.
- Keep public class/function docstrings Google-style and include the governing
  chemical-reactor reference already cited by the equation helpers after its
  citation is verified.

Documentation completion requires an executable example, valid internal links,
and exact agreement between documented symbols and `particula.dynamics`
exports.
