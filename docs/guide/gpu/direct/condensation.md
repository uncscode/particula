# Condensation

### Low-level Warp GPU condensation

#### Tagged native populations and species binding

Tagged discrete/resolved GPU calls require `species_binding=`. Create it
explicitly with `prepare_condensation_species_map` from
`particula.gpu.species_mapping`, using the exact Warp carriers, actual complete
gas-name order retained at transfer, and the process's
`CondensationSpeciesMap`. The binding owns fresh device lane maps and pins
array identities, layouts, devices and population metadata. Keep its map
contents unchanged; new storage, names, mapping or metadata requires fresh
preparation. Full-gas partitioning remains live device state.

| Native quantity | Species order / units |
|---|---|
| Particle masses and work/total transfer | Full material order; kg per particle, `(boxes, slots, materials)` |
| Activity kappas, reference molar masses and surface tension | Full material order; mapped reference masses must match the declared gas species |
| Thermodynamics, accommodation, diffusion and latent/thermal arrays | Full gas order, including unused/nonpartitioning gas species |
| Demand/release/scale scratch and energy output | Full gas width, `(boxes, gas_species)` |
| Gas inventory coupling | kg/m³: transfer weighted by N/V for resolved counts, stored density for PMF |
| Native energy output | Extensive J: applied transfer weighted by counts and mapped latent heat |

The direct and private prepared routes use the same four equal substeps. Each
refreshes full-gas vapor pressure, evaluates full particle composition, gates
mapped proposals, bounds evaporation and gas-limits uptake, then couples the
representable particle change to gas. Whole-call transfer and energy outputs
are cleared at entry and accumulated from the applied substeps. PMF requires
V=1; resolved volumes need not be one. PDF execution rejects explicitly.

Resolved N/V must be finite at admission. Evaporation contributes only its
representable release to the uptake budget. Before either inventory update,
rounded positive additions are reconciled against available gas; a bounded
downward correction suppresses uptake if it cannot fit, and a nonfinite
reduction suppresses that species' transfer. Fully evaporated resolved slots
have their counts and charge cleared after whole-call gas and latent-energy
accounting. Empty gas storage with an empty map is a no-op that clears transfer
outputs and launches no gas-indexed work.

Setup validates configuration, physical state and nonoverlapping writers;
validation may launch read-only device scans and read private status flags.
Enqueue uses retained references without allocation, upload, readback,
synchronization or rebinding. No rollback is promised after a writer launches.
The standalone private inventory-finalization test helper remains legacy-only;
it is not a mapped native entry point. Untagged legacy direct/prepared callers
retain the same-width contract described below. Native transfer does not
establish resident/checkpoint support by itself. See the separately exercised
[resident walkthrough](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/resident/gpu_resident_multi_timestep.py) for explicit
exact-device restart and fresh species bindings.

Native scientific evidence is in
`particula/gpu/kernels/tests/condensation_native_test.py`: independent scalar
four-substep equations, separate particle/gas changes, V=.25/1/4 conservation,
unequal counts, unequal species widths, latent energy, activity/surface modes,
rejection snapshots and prepared execution. Warp CPU is required when Warp is
installed; CUDA is optional and skips explicitly when unavailable.

#### Temporary untagged direct path

This is a direct low-level Warp path, separate from the CPU strategies and the
`MassCondensation` runnable described above. Its canonical step import is:

```python
from particula.gpu.kernels import condensation_step_gpu
```

When reusable scratch is needed, import `CondensationScratchBuffers` only from
`particula.gpu.kernels.condensation`. That concrete-module-only sidecar is not
a second step entry point. Its transfer fields have shape
`(n_boxes, n_particles, n_species)` and its property fields have shape
`(n_boxes,)`. Each supplied field must be active-device, stable-shape
`wp.float64`; fields may be omitted independently and use fallback allocations.
Supplied buffers preserve identity and callers must keep them alive and
unmodified through launch completion.

Every successful call executes exactly four `time_step / 4.0` substeps. Each
substep optionally refreshes composition-weighted surface tension, overwrites
`gas.vapor_pressure`, refreshes environment properties, produces a raw transfer
proposal, finalizes that proposal against available inventory, applies the
finalized transfer to particle mass, accumulates it, and mutates gas
concentration by the matching particle-concentration-weighted amount. Later
proposals therefore see gas concentration coupled by earlier substeps.
Vapor-pressure refresh is temperature-driven and does not read gas
concentration. The resolved total transfer buffer is cleared once after
preflight, accumulates inventory-limited transfer, and is returned by identity when
supplied. Work storage keeps only the final raw proposal. Production
calculations make no hidden data CPU↔Warp transfers. CUDA validation reads small
device status flags with `.numpy()`, which synchronizes execution but does not
transfer or mutate caller-owned simulation buffers.

Invalid primary inventory finalization state or supplied inventory finalization sidecars fail during aggregate
preflight, before launches or caller-buffer mutation. A non-finite fresh raw
proposal fails before inventory finalization finalization for that substep; completed earlier
substeps remain applied. The failed cycle may already have written raw work,
`gas.vapor_pressure`, and supplied dynamic-viscosity, mean-free-path, or
composition-weighted surface-tension workspaces, but not inventory finalization inventories or
energy output. Retain or restore caller-owned snapshots when a failed call must
be retried from its original state.

This direct low-level GPU step uses an optional keyword-only caller-owned
active-device `wp.float64` `latent_heat` sidecar with shape `(n_species,)` in
each fixed substep. Its shipped rate is
`dm_i/dt = isothermal_rate_i / correction_i`, where
`isothermal_rate_i = k_i * Delta_p_i * M_i / (R * T)`,
`R_specific_i = R / M_i`, and
`correction_i = thermal_factor_i / (R_specific_i * T)`. The source thermal
resistance expression is
`thermal_factor_i = (D_i * L_i * p_surface_i / (T * k_thermal)) * (L_i / (T * R_specific_i) - 1) + R_specific_i * T`,
using the activity- and Kelvin-adjusted `p_surface_i`. Omitting latent heat, or
setting a species entry to exactly zero, takes that species through the exact
isothermal branch.

Issue #1272 also ships optional keyword-only caller-owned active-device
`wp.float64` `energy_transfer`, shape `(n_boxes, n_species)`, as write-only
diagnostic output. It requires valid latent heat, is overwritten only after
successful preflight, and is not a third return item. Its signed whole-call
identity is `Q[box, species] = sum_particles(Delta m_finalized) * L[species]`:
finalized `Delta m` is kg, `L` is J/kg, and `Q` is J, so condensation is
positive and evaporation negative (`kg * J/kg = J`). The #1272 diagnostic uses strict
`rtol=1e-12, atol=1e-18` evidence. `thermal_work` is validated but remains
deferred and unused.

This is diagnostic bookkeeping, not temperature feedback, gas mutation or
gas/full-system conservation. The direct step couples finalized particle and
gas transfer through its stated inventory-limited direct coupling, but does not add strategy/
`Runnable` integration, adaptive stepping, graph capture/replay, autodiff
guarantees, or complete condensation parity cross-device certification.

The [GPU condensation parity walkthrough](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_condensation_parity_walkthrough.py)
separately reports physics, conservation, and energy evidence for this
fixed-four-substep low-level direct-kernel comparison. Its
[condensation parity walkthrough ownership record](../../../contribute/validation/gpu/condensation-parity.md)
keeps deferred capabilities explicit. Warp CPU is the installed-Warp baseline;
CUDA is optional additive evidence. The caller-owned, write-only
`energy_transfer` output is a diagnostic, not a return value and not temperature
feedback.

Focused fixed-four-substep direct-kernel evidence commands are:
`python docs/examples/gpu/direct/gpu_condensation_parity_walkthrough.py` and
`pytest examples_tests/gpu_condensation_parity_walkthrough_test.py -q -Werror`.

#### Direct-condensation evidence

| Evidence | Discoverable wrapper | Bounded result and backend |
| --- | --- | --- |
| Independent parity | `particula/gpu/kernels/tests/condensation_test.py` | An independent NumPy fixed-four-substep oracle separately compares final particle masses and gas concentrations for one/multi-box and multi-species cases. Warp CPU is the normal backend when Warp is installed; CUDA is optional additive evidence. |
| Conservation and contracts | `particula/gpu/kernels/tests/condensation_test.py` | Separate concentration-weighted particle-plus-gas inventory checks use `rtol=1e-12`, distinct from independent parity. The wrapper also covers in-place mutation and caller-owned output contracts. |
| Capture limit | `particula/gpu/kernels/tests/condensation_graph_capture_test.py` | Warp CPU capture is capability-skipped because Warp capture requires CUDA. On CUDA, only the public-step host-validation readback inside capture is strict-xfailed; setup and normal calls must still pass. This records an unsupported capability, not replay support. |
| Raw-rate autodiff | `particula/gpu/kernels/tests/condensation_autodiff_test.py` | A one-box fp64 out-of-place `condensation_mass_transfer_kernel` Tape derivative is compared with centered finite differences only in the interior, where inventory finalization positivity and inventory boundaries are inactive. Warp CPU is the normal backend; CUDA is optional. |

This matrix is evidence for `condensation_step_gpu` (proposal gating--thermal correction) or the raw
`condensation_mass_transfer_kernel` (validation) only. It establishes no CPU strategy
or `Runnable` equivalence, performance target, adaptive stepping, new API,
general graph replay, or general differentiability claim. For focused,
warning-clean coverage, run:

```bash
pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_graph_capture_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_autodiff_test.py -q -Werror

# Optional CUDA selections; unavailable CUDA must skip cleanly.
pytest particula/gpu/kernels/tests/condensation_test.py -q -m "warp and cuda" -Werror
pytest particula/gpu/kernels/tests/condensation_graph_capture_test.py -q -m "warp and cuda" -Werror
pytest particula/gpu/kernels/tests/condensation_autodiff_test.py -q -m "warp and cuda" -Werror
```

Warp-backed tests may skip when Warp is missing. CUDA evidence is optional when
CUDA is unavailable; a skip is not GPU execution. thermal correction's CPU skip and narrowly
scoped CUDA strict expected failure document the public-step host-readback
limitation rather than a successful capture replay.

## Backend-selected condensation

Selected condensation is a bounded concrete adapter layer over the existing
native CPU `MassCondensation` runnable and the direct
`condensation_step_gpu` boundary. It is not new physics, a new public runnable,
a scheduler, a resident session, or automatic backend selection. Ownership and
selection boundaries are described in the
[execution-selection guide](../execution.md#execution-selection)
and its [GPU thermodynamics and condensation refresh record](condensation-configuration.md#gpu-thermodynamics-and-condensation-refresh).

| Selection/backend case | Contract |
| --- | --- |
| CPU semantic catalogue | All 36 equal-step/staggered, latent, activity, and surface semantic combinations are declared, including `NONREPRESENTABLE` categories. This is catalogue support only. |
| Selected CPU execution | The concrete adapter accepts declared catalogue metadata but does not bind it to the supplied runnable; it is isothermal only (`latent_heat=False`) and calls the caller-owned `MassCondensation` once. Equal-step or staggered semantics remain CPU-side. |
| Selected Warp execution | Exactly 8 declared profiles: equal-step, latent heat enabled or disabled, ideal or kappa activity, and static or composition-weighted surface tension. Profile preflight requires enabled latent-heat storage, checks activity/surface mode selectors, and validates their particle-width arrays. Native species bindings validate exact carrier ownership separately. |
| Unavailable selected Warp mapping | Staggered/Gauss-Seidel and every `NONREPRESENTABLE`/BAT activity or surface mapping reject during capability-profile preflight before lazy kernel resolution, native dispatch, or adapter-driven writes; no conversion, approximation, or fallback occurs. |

Warp delegation executes exactly four equal `time_step / 4.0` substeps. Each
substep overwrites and refreshes temperature-driven vapor pressure, then applies
inventory finalization inventory-limited particle/gas coupling. It mutates particle masses, gas
concentration, and derived GPU vapor pressure in place. The caller owns supplied
finalized total-transfer output, scratch/work storage, `latent_heat`, write-only
`energy_transfer`, and validated but deferred `thermal_work` sidecars. Omitted
optional scratch or output buffers may use direct-kernel fallback allocation;
validation may perform permitted device scans or status readbacks. Adapter
dispatch itself performs no hidden transfer, restore, retry, or CPU fallback,
and callers synchronize before host observation.

Bounded integration fixtures use an independent fixed-four-substep inventory finalization oracle
that separately compares particle mass and gas concentration at
`rtol=1e-10, atol=1e-30`. Inventory is checked separately. Warp CPU is the
installed-Warp baseline, and CUDA rows are optional guarded evidence. This does
not claim general
CPU-strategy parity, performance, adaptive stepping, graph capture/replay,
autodiff, multi-GPU, resizing, or full scheduler support. The
[Epic G record](../../../Features/Roadmap/data-oriented-gpu.md#epic-g-backend-selection-and-gpu-resident-simulation)
and resident guides define their separate scheduling and lifecycle scope.

