# Condensation Configuration

### GPU thermodynamics and condensation refresh

Import the supported direct step only with
`from particula.gpu.kernels import condensation_step_gpu`.
Import its required `ThermodynamicsConfig` from the concrete module only:

```python
from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
```

`ThermodynamicsConfig` is caller-owned, device-local process configuration.
Its Warp arrays have no species names: species identity is positional, and
`molar_mass_reference` must exactly equal ordered `gas.molar_mass` on the
active device. `WarpGasData.vapor_pressure`, in contrast, is a mutable,
derived GPU helper buffer rather than authoritative `GasData` state. It is
omitted by `from_warp_gas_data()`.

The fixed configuration and refresh contracts are:

| Field | Required Warp dtype and shape | Meaning |
| --- | --- | --- |
| `modes` | `wp.int32`, `(n_species,)` | Per-species model code. |
| `parameters` | `wp.float64`, `(n_species, 4)` | Per-species model parameters. |
| `molar_mass_reference` | `wp.float64`, `(n_species,)` | Ordered compatibility fingerprint for `gas.molar_mass`. |
| Refresh temperature | `wp.float64`, `(n_boxes,)` | Device-local temperature at the standalone refresh boundary. |
| `vapor_pressure` output | `wp.float64`, `(n_boxes, n_species)` | Derived pressure matrix in Pa. |

Only two vapor-pressure model codes are shipped: constant `wp.int32(0)`, which reads
`parameters[:, 0]` in Pa, and canonical Buck `wp.int32(1)`, which evaluates
the water/ice equations and ignores all four reserved parameter slots.

`condensation_step_gpu()` requires keyword-only `thermodynamics=` while
preserving the positional `mass_transfer` slot. Its accepted temperature
sources are a direct scalar, a direct same-device Warp array with shape
`(n_boxes,)`, hybrid direct scalar/Warp-array inputs, or keyword-only
`environment=` when both direct values are `None`. NumPy arrays and Python
lists are not accepted as direct arrays, and the step does not perform hidden
transfers. Temperature and pressure must be positive finite values. Direct and
environment temperature/pressure arrays are active-device `(n_boxes,)` inputs;
non-`wp.float64` temperature arrays are normalized into a device-local fp64
buffer. This normalization does not make every environment input fp64-only.

On every successful condensation call, the step validates aggregate caller
state and optional sidecars first. It then casts a non-`wp.float64` direct or
environment temperature array into a device-local float64 buffer when needed
and performs exactly four equal `time_step / 4.0` substeps. Each substep
overwrites `vapor_pressure`, prepares environment properties, gates disabled
partitioning entries and inactive particle slots, and creates a raw proposal.
Sequence validation finalizes that proposal against particle and gas inventory limits, then
derives the matching concentration-weighted opposite gas delta. The step
validates that delta and all pending commit values before mutating particle
masses, accumulating the finalized transfer in the returned total-transfer
buffer, and mutating `gas.concentration`. Raw proposal work storage is
intermediate state, not returned transfer. Later proposals read the coupled gas
concentration; vapor-pressure refresh does not. A failed aggregate preflight
leaves the vapor-pressure output buffer unchanged. Import the
standalone `refresh_vapor_pressure_gpu` only from
`particula.gpu.kernels.thermodynamics`, never `particula.gpu.kernels`.

### Condensation activity and surface sidecars

The optional `CondensationActivitySurfaceConfig` is a direct-kernel-only,
caller-owned configuration sidecar for `condensation_step_gpu()`. Activity mode
`0` is ideal and mode `1` is kappa; surface-tension mode `0` is static and mode
`1` is composition-weighted. Activity applies only to `water_species_index`;
all non-water species use unit activity. Static tension uses the current
species' supplied value, while composition-weighted tension computes one
particle-wide value from the per-species tensions.

For native calls, let `Sp` be particle-material width and `Sg` full gas width.
`kappas` and `molar_mass_reference` are same-device `wp.float64` arrays shaped
`(Sp,)`. The water selector addresses a particle-material lane. Kappas are
finite and nonnegative; reference molar masses are finite and positive.
Every mapped particle reference must exactly equal its corresponding gas
molar mass through the explicit species map; unmapped particle references
remain particle-order data. Only the legacy equal-width path uses positional
whole-array equality with `gas.molar_mass`. The frozen dataclass prevents
rebinding, not array mutation; callers must not mutate storage during launch.

Successful direct calls mutate particle masses and gas concentration in place,
overwrite the derived GPU-only vapor-pressure buffer, and return the accumulated
**inventory-limited, inventory-limited** transfer. A supplied total-transfer buffer
holds that finalized accumulated output and is returned by identity; separate
supplied work storage retains only the final gated raw proposal and is never the
returned transfer. The two-item return assignment is
`particles_out, mass_transfer = condensation_step_gpu(...)`;
`energy_transfer` is caller-owned output, not a third return value. Disabled
`(box, species)` entries in the
per-box `(n_boxes, n_species)` partitioning mask and inactive particle slots
are zeroed before sequence validation finalization or reductions. CPU↔Warp conversion expands
the CPU shared species mask to this per-box layout. `to_per_box_partitioning()`
is the public migration helper for legacy one-dimensional masks. CPU restoration
rejects masks that differ by box and rejects zero-box mirrors because no box can
authoritatively supply the shared CPU mask.
Sequence validation demand, release, scale, and accumulator sidecars are validated and used for
finalization; aggregate invalid state, metadata, or ownership fails with
`ValueError` before mutable physics or caller-state mutation. Preflight may run
read-only validation kernels. A later failure caused by a fresh raw proposal
does not roll back completed substeps, although it occurs before sequence validation mutation in
its failing substep. Unsupported activity or surface
strategies are not silently copied or approximated; passing
`activity_surface=None` retains legacy unit activity and static tension.

`CondensationScratchBuffers` is a concrete-module-only optional sidecar at
`particula.gpu.kernels.condensation`; it is not another step entry point.
Native buffers use independent particle and gas widths:

| Field family | Shape | Lane authority |
| --- | --- | --- |
| Particle mass, accumulated transfer, raw/finalized scratch transfer | `(B, N, Sp)` | Particle-material order |
| Gas concentration, vapor pressure, energy output | `(B, Sg)` | Full gas order |
| sequence validation demand, release, scale and gas accumulator | `(B, Sg)` | Full gas order |
| Latent heat and deferred thermal work | `(Sg,)` | Full gas order |
| Thermodynamic model codes/reference masses | `(Sg,)` | Full gas order |
| Activity kappas/reference masses | `(Sp,)` | Particle-material order |
| Scratch environment properties | `(B,)` | Box order |

Numeric transfer/property/thermal storage is same-device `wp.float64`;
thermodynamic model codes are `wp.int32`. `B` is box count and `N` is fixed
particle capacity. Every supplied field is validated for its applicable shape,
dtype and device. Independently omitted optional scratch fields use fallback
allocations; supplied fields retain identity but mutable work/output storage is
written on success. An explicit binding connects the two species orders; equal
array widths alone do not establish chemical alignment. No implicit transfer,
container replacement or automatic rebinding follows from these shapes.

The direct condensation parity suite exercises all four activity/surface pairs
(ideal/static, ideal/composition-weighted, kappa/static, and
kappa/composition-weighted) against an independent NumPy `float64` reference.
It uses deterministic one-box and multi-box inputs, including pure and mixed
particle compositions, a clamp-to-zero evaporation case, and constant plus
both Buck vapor-pressure branches. Parity checks cover raw transfer, final
particle mass, refreshed vapor pressure, finalized totals, and coupled gas
concentration. Tighter conservation checks separately verify the weighted
particle transfer equals the opposing gas change. These checks remain a bounded
direct-step contract, not a general CPU-strategy parity claim.

### Latent-rate correction and energy diagnostic

The direct step accepts optional keyword-only caller-owned active-device
`wp.float64` `latent_heat` with shape `(Sg,)` in full gas order. For species `i`, its
shipped latent-rate correction is
`dm_i/dt = isothermal_rate_i / correction_i`, where
`isothermal_rate_i = k_i * Delta_p_i * M_i / (R * T)`,
`R_specific_i = R / M_i`, and
`correction_i = thermal_factor_i / (R_specific_i * T)`. The source thermal
factor is
`thermal_factor_i = (D_i * L_i * p_surface_i / (T * k_thermal)) * (L_i / (T * R_specific_i) - 1) + R_specific_i * T`.
Here `p_surface_i` is the activity- and Kelvin-adjusted surface vapor pressure.
An omitted sidecar, or exactly `L_i = 0`, follows the exact isothermal branch.

Optional keyword-only `energy_transfer` is a caller-owned active-device
`wp.float64`, `(B, Sg)`, write-only output in full gas order. It requires valid
`latent_heat`, is overwritten only after successful preflight, and is not a
third return value. Its signed whole-call diagnostic, shipped in issue #1272,
is `Q[b,g] = sum_slots(count[b,n] * Delta m_applied[b,n,p(g)]) * L[g]`
for native resolved storage: `Delta m`
is applied transfer in kg, `L` is latent heat in J/kg, and `Q` is J. Thus
condensation is positive and evaporation is negative. The strict #1272
energy-regression tolerance is `rtol=1e-12, atol=1e-18`.

`thermal_work` has the same validated sidecar shape but remains deferred and
unused: the step neither reads nor writes it. It does not evolve temperature or
add a `Runnable`, adaptive substeps, graph capture/replay, or autodiff.

Direct callers use `to_warp_*` and `from_warp_*` for explicit CPU↔Warp transfer.
The separate [prepared runnable path](../prepared.md) validates native `Aerosol`
configuration and owns upload and resident resources. CUDA preflight reads one-element
device validation flags with `.numpy()`; those status reads synchronize but do
not transfer or mutate caller-owned simulation buffers. Callers requiring retry
or rollback must snapshot and restore particle masses, gas concentration,
derived vapor pressure, and caller-owned output/work buffers. Deterministic
fp64 tests compare an independent NumPy reference with explicit parity
`rtol`/`atol`, then apply tighter separate ownership invariants. Warp `cpu` is
the baseline whenever Warp is installed; CUDA is optional, availability-guarded,
and separately marked. Warp-dependent test imports are lazy, so marker-
deselected collection does not require Warp. Focused direct-kernel and CPU
integration smoke commands are:

```bash
pytest particula/gpu/kernels/tests/condensation_test.py --collect-only -q -m "not warp"
pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror
pytest particula/integration_tests/condensation_particle_resolved_test.py -q -Werror
```

Run CUDA-marked kernel cases separately only on CUDA-capable hosts; unavailable
Warp or CUDA remains an expected guarded skip, not evidence of a fallback path.


Native mapped condensation uses each vapor species' molar mass for its
transport mean free path, matching the native CPU strategy. Particle mobility
uses air properties. The low-level raw proposal kernel retains its explicitly
supplied mean-free-path semantics.
