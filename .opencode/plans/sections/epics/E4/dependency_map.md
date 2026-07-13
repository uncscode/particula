# Dependency Map

## Inbound

- Shipped E1/E2/E3 GPU container, conversion, environment, and kernel foundations.
- `particula/gpu/warp_types.py` fixed-shape schemas.
- `particula/gpu/kernels/environment.py` input normalization contract.
- CPU condensation, activity, surface, vapor-pressure, and latent-heat references.
- The selected `fixed_count_substeps_4` stiffness-study result.

## Outbound

- Trustworthy gas-particle coupled GPU simulations and future high-level integration.
- GPU support documentation and direct-kernel examples.

## Sequencing

```text
E1/E2/E3 -> E4-F1 -> E4-F2 --\
                    E4-F3 ---+-> E4-F4 -> E4-F5 -> E4-F6 -> E4-F7
```

- E4-F2 and E4-F3 are the only parallel branch.
- E4-F4 requires E4-F1, E4-F2, and E4-F3.
- E4-F5 requires stable integration and latent-transfer semantics.
- E4-F6 requires all production physics tracks; E4-F7 requires accepted evidence.
