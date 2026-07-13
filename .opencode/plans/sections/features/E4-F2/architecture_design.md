# Architecture Design

## High-Level Design

```text
E4-F1 temperature + numeric vapor-pressure config
  -> refreshed gas.vapor_pressure[n_boxes, n_species]
particle masses + density + molar mass + numeric E4-F2 config
  -> on-device composition denominators
  -> ideal/kappa activity and effective surface tension
  -> Kelvin radius/term
  -> activity * pure vapor pressure * Kelvin term
  -> pressure delta -> existing mass-transfer path
```

Formula helpers belong in focused GPU property modules; orchestration,
preflight validation, and launch order remain in
`particula/gpu/kernels/condensation.py`. Species position is identity. Model
modes are int32 and physical parameters/results are fp64 fixed-shape arrays.

## Data / API / Workflow Changes
- **Data model:** No CPU or Warp container schema changes. Configuration remains
  caller-owned sidecar data; E4-F1 owns the refreshed vapor-pressure buffer.
- **API surface:** Preserve positional per-species `surface_tension`; add
  keyword-only numeric activity/surface configuration. Unsupported mode values,
  invalid water indices, shape/dtype/device mismatches, and non-finite or
  negative physical parameters fail before allocation, launch, or mutation.
- **Workflow hooks:** The primitive remains directly importable through
  `particula.gpu.kernels`, callable repeatedly by E4-F3, and feeds E4-F4.

## Security & Compliance

There is no permission or network change. Robustness requires bounded Kelvin
exponents, finite/nonnegative state checks, no Python strings or strategy
objects in Warp containers, no hidden host transfer, and deterministic failure
before state mutation.
