# Appendix

## Primary References

- `docs/Features/Roadmap/data-oriented-gpu.md` — epic scope and exit bar.
- `docs/Features/Roadmap/condensation-stiffness-study.md` — four-substep selection.
- `particula/gpu/kernels/condensation.py` — current production GPU path.
- `particula/gpu/kernels/environment.py` — normalized environment contract.
- `particula/gpu/warp_types.py` and `particula/gpu/conversion.py` — schemas and ownership.
- `particula/dynamics/condensation/condensation_strategies.py` — CPU composition and latent reference.
- `particula/dynamics/condensation/mass_transfer.py` — thermal resistance and energy equations.
- `particula/gpu/kernels/tests/_condensation_test_support.py` — fixed-substep prototype.

## Key Design Decisions

1. Numeric Warp-safe model selection replaces Python strategy objects on-device.
2. Vapor pressure remains GPU helper state and is refreshed per substep.
3. The public path uses exactly four substeps and stable-shape scratch.
4. Gas mutation lands only with inventory bounds and conservation tests.
5. Multi-box GPU parity compares each box with an independent one-box CPU run.

## Rejected Alternatives

- Adaptive stepping: rejected for variable control flow and graph instability.
- Hidden host refresh: rejected because it violates explicit transfer ownership.
- One monolithic feature: rejected because it obscures the required dependency gates.
