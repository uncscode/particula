---
title: Migration Troubleshooting
---

# Migration Troubleshooting

## Shape mismatches when creating data containers

`ParticleData` expects `(n_boxes, n_particles, n_species)` for masses and
`(n_boxes, n_particles)` for concentration and charge. Use `np.newaxis` or
`np.tile` to add the batch dimension.

## Deprecation logs

The facades log at INFO level to avoid `-Werror` failures. To reduce noise,
prefer `ParticleData` and `GasData` directly or wrap with `from_data` methods.

## Single-box vs multi-box data

Legacy facades assume a single box. The audited CPU baseline is:

- Condensation public `ParticleData` and `GasData` paths accept only
  `n_boxes == 1` and reject `n_boxes != 1`.
- CPU coagulation `ParticleData` paths accept only `n_boxes == 1`; multi-box
  inputs raise a clear `ValueError` instead of falling back to box `0`.

For the support contract and caller-managed per-box loop guidance, see
[Dynamics migration](dynamics.md#using-particledata-and-gasdata-in-dynamics).

When legacy-shaped arrays are needed, index the first box explicitly:

```python
radii_single_box = particle_data.radii[0]
concentration_single_box = gas_data.concentration[0]
```

## `condensation_step_gpu` rejects my environment inputs

`condensation_step_gpu(...)` validates environment inputs before Warp launch.
Check the following first:

- Do not mix direct `temperature` and `pressure` arguments with `environment=`
  in the same call.
- If `environment` is omitted, supply both direct thermodynamic inputs as
  scalars, `(n_boxes,)` Warp arrays, or a supported hybrid.
- Ensure direct or environment-owned `temperature` and `pressure` values are
  positive and finite.
- Ensure direct arrays match the particle and gas box count and live on the
  same Warp device.

For a CPU-owned source of truth, keep using `EnvironmentData` and convert it
only at the explicit `to_warp_environment_data()` boundary.

## Direct-condensation troubleshooting and reproduction

Keep restored ordered gas names and thermodynamics-sidecar species order aligned
with `gas.molar_mass`, including a valid water-species index. Particle and gas
layouts retain their leading `(n_boxes, ...)` dimension, but sidecars have
field-appropriate shapes: species configuration uses `(n_species,)`, scratch
property fields use `(n_boxes,)`, and transfer fields use their required
per-particle or per-species transfer shapes. Scalar indices, including the
water-species index, remain scalar. Supplied scratch, latent-heat, and energy
sidecars must be active-device `wp.float64`.

Use either `environment=` or both direct positive finite temperature/pressure
inputs, with direct arrays on the active device. P2 inventory limiting bounds
applied transfers rather than proving parity. Synchronize explicitly before host
observation of caller-owned energy output. Warp `device="cpu"` is the baseline
when installed; CUDA is optional/local and skips cleanly when CUDA is
unavailable.

For the single canonical command matrix, see the
[GPU condensation command matrix](../data-containers-and-gpu-foundations.md#focused-reproduction-commands).
