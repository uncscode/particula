---
title: Native migration troubleshooting
---

# Native migration troubleshooting

## Explicit population kind is required

Set `distribution_type` through a native initializer/builder or explicit
`ParticleData` construction. Choose the physical measure deliberately:

- `discrete`: bin number density, m⁻³, at exactly unit volume.
- `continuous_pdf`: radius `dN/dr`, m⁻⁴, with a positive increasing radius grid
  and exactly unit volume. Integrate over radius; do not sum nodes.
- `particle_resolved`: represented counts at positive physical volume.
  Density input must be multiplied by volume once; density output divides once.

An untagged legacy carrier is not native. Facade wrapping or stripping a tag is
not a workaround. For initializers, `concentration_units="count"` preserves
weights; explicit density units convert to counts using represented volume.

## Replacement rejects a shape or layout

Particle masses use `(B,N,P)`, counts/weights and charge `(B,N)`, density `(P,)`
and volume `(B,)`. Full gas concentration and environment saturation ratios use
`(B,G)`. Temperature and pressure use `(B,)`; molar mass and gas flags use `(G,)`.
Particle and gas species widths may differ. Keep every gas lane, including inert
species.

Individual assignments validate against held peers. To change a gas width and
its matching environment width together, use `aerosol.replace_data` with all
three candidate containers. Rejection preserves held identities and candidate
state. Construction/replacement validates structure and representation, not all
later physical requirements or an unnamed permutation of chemistry. See the
[executable replacement checks](containers.md#executable-access-and-replacement-walkthrough).

## A process rejects state after successful construction

Raw arrays are writable; direct edits may introduce negative/nonfinite values.
Processes revalidate their own physical domain and full expected gas order.
Reordering or replacing gas does not update old process configuration. Recreate
its explicit mapping only when the new order is intentional. Do not catch the
error and retry with a facade.

Resolved coagulation requires equal positive active counts. Fractional counts
are valid storage but that does not imply every process supports arbitrary
weights. Binned coagulation also retains method-specific composition limits.
Consult the process documentation instead of changing conservation tolerances.

## Multi-box storage is not general CPU execution

Native CPU runnables remain single-box. Multi-box arrays must not be passed to a
single-box process expecting implicit iteration. Application-owned per-box
execution requires explicit independent state/configuration and result handling;
there is no undocumented helper that slices/reassembles a complete simulation.
See [dynamics migration](dynamics.md).

## A GPU binding rejects state

Native condensation and nucleation bindings pin the exact carriers, arrays,
devices, population metadata and complete gas-name provenance. Prepare them
explicitly after conversion. Replacing CPU containers does not change existing
device state; replacing Warp storage or restarting a resident session needs fresh
explicit preparation. Radius-PDF particle upload is unsupported.

Direct condensation requires `thermodynamics=`. Supply either `environment=` or
both direct temperature/pressure inputs, never both forms. Direct arrays must
have shape `(B,)`, positive finite values and the active device. Configuration,
scratch and diagnostic arrays must match their documented per-material,
full-gas or per-box shapes and dtypes; equal widths are not a species map.

## Restore lost names or derived state

`WarpGasData` does not own gas names. Pass the saved full ordered names to
`from_warp_gas_data`; otherwise placeholders are produced. CPU `GasData` has no
vapor-pressure field, so this device-only derived state is dropped on restore.
Normal condensation refreshes it from current temperature and thermodynamics.
Synchronize before host observation, or use the conversion helper's documented
synchronization mode.

Checkpoint CPU inspection carriers are also lossy. Explicit restart consumes
canonical schema-v5 bytes on an exactly compatible device; it does not rebuild
continuation from inspection data or silently migrate an old schema.

## Reproduce the supported path

Run the published scripts and their owning `examples_tests` suites from the
selected E9-native checkout. Warp CPU is the installed-Warp baseline for
uncaptured examples. Optional CUDA tests skip when unavailable; native capture
has no CPU fallback. A no-Warp result must report that no kernel ran, not claim
physics parity. See [focused reproduction commands](../data-containers-and-gpu-foundations.md#focused-reproduction-commands).
