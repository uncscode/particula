# Coagulation

### GPU direct-kernel foundations and limitations

This direct-kernel-only path is separate from CPU strategies, builders,
factories, and `Runnable` APIs. Import its direct step and immutable host
configuration with:

```python
from particula.gpu.kernels import coagulation_step_gpu
from particula.gpu.kernels.coagulation import CoagulationMechanismConfig
```

It accepts only explicitly tagged `distribution_type="particle_resolved"`
populations. Direct and prepared entry points reject untagged, discrete, and
continuous-PDF populations before particle-array access; no population meaning
is inferred. The identifiers are
`brownian`, `charged_hard_sphere`, `sedimentation_sp2016`, and
`turbulent_shear_st1956`; requested tuples normalize to Brownian, charged,
sedimentation, then turbulent order. Executable masks are singleton masks `1`,
`2`, `4`, `8`; two-term masks `3`, `5`, `6`, `9`, `10`, `12`; and four-term
mask `15`. Three-term masks `7`, `11`, `13`, and `14` are deferred and fail
closed. Mask `7` rejects before particle metadata or enabled-term validation;
masks `11`, `13`, and `14` validate metadata and enabled terms before failing.

Every turbulent mask requires keyword-only positive finite
`turbulent_dissipation` and `fluid_density`: either scalars or same-device
`wp.float64` arrays with shape `(n_boxes,)`. Non-turbulent masks ignore these
inputs. Particle data, including same-device fp64 `charge`, supplied collision
pair/count outputs, and persistent RNG state are caller-owned Warp resources.
Accepted collisions mutate particle mass, concentration, and charge in place;
supplied outputs retain identity, persistent RNG advances in place and resets
only with `initialize_rng=True`, and omitted outputs/state are call-local. The
exact return tuple is `(particles, collision_pairs, n_collisions)`; RNG state is
not returned.

For native `ParticleData` transferred to Warp, every supported mechanism
requires **equal positive active counts within each box**.
Counts may be fractional and may differ between boxes; zero-count slots are
inactive. Every native schedule includes the common count: its trial budget
scales as `K_majorant * pair_count * count * dt / volume`. The original
per-particle species masses and raw counts remain in their original storage.
Accepted merges conserve count-weighted species mass and charge; donor fields
are cleared and recipient counts remain unchanged. Untagged legacy inputs keep
their separately tested, mechanism-specific behavior.

Native admission rejects PMF/PDF populations, unequal active counts, malformed
fp64 primary storage, invalid physical state, and overlapping writable and
protected arrays. A supplied volume must match the physical volume on the
particle container. CPU-to-Warp and reverse transfers retain raw counts and
representation metadata. Existing charged/sedimentation active-count limits
remain mechanism-specific; the native tag does not add new mechanism support.

The private prepared call freezes the native scheduling mode and device
references. Enqueue uses current counts from the pinned device arrays and
performs no allocation, readback, synchronization, reset or host configuration
resolution. Callers keep arrays alive and state compatible between enqueues;
new bindings or semantics require preparation again. This is direct/prepared
science evidence. Native resident continuation is illustrated in the
[resident walkthrough](https://github.com/uncscode/particula/blob/main/docs/Examples/gpu_resident_multi_timestep.py); it requires
fresh exact-device containers and bindings after explicit restart.

Native regression coverage uses resolved volumes 0.25/1/4 m³, common counts
0.25/1/4, independent species/charge inventories and independent one-pair
stochastic expectations for all eleven supported masks on Warp CPU and optional
CUDA. Run it with:

```bash
pytest particula/gpu/kernels/tests/coagulation_native_test.py -q --no-cov
```

Malformed configuration and unsupported distributions fail before particle
access. Same-device preflight validates caller state, but a later runtime
failure has no rollback guarantee. Transfers are explicit: there is no automatic
transfer, CPU fallback, or high-level `Runnable` integration. Warp CPU is the
installed-Warp baseline. CUDA is optional/additive, never mandatory, and guarded
suites skip cleanly when it is unavailable. This path provides no performance,
broad CPU/GPU parity, graph-capture, broad autodiff, DNS/general-turbulence
support, unsupported-mechanism, or unsupported-distribution claim. See the
[canonical ownership contract](../../data-model/index.md)
and [evidence/command record](../../../contribute/validation/gpu/coagulation.md).

Focused baseline commands:

```bash
pytest particula/gpu/kernels/tests/coagulation_validation_test.py -q -m "warp and gpu_parity" -Werror
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and stochastic and not cuda" -Werror
```

**Optional/local CUDA evidence:**

```bash
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -m "warp and cuda" -Werror
```

This optional/local command is additive and skips cleanly when CUDA is
unavailable.

### Concrete selected-Brownian adapter

The concrete selected Brownian coagulation adapter is a separate selected route beside the
multi-mechanism direct-kernel contract above. It accepts an exact
`BrownianCoagulationConfig` marker only. The CPU reference route retains a
caller-owned `Aerosol` and exact `Coagulation` runnable. The resident-Warp
route retains caller-owned `WarpParticleData`. It supports Brownian
`particle_resolved` dispatch only. CPU and Warp stochastic trajectories have
no seed-by-seed parity claim.

Import the concrete-only marker, state carriers, and adapters from their
concrete module:

```python
from particula.execution.adapters.coagulation import (
    BrownianCoagulationConfig,
    CPUCoagulationExecutionAdapter,
    CPUCoagulationExecutionState,
    CPUCoagulationState,
    WarpBrownianCoagulationExecutionAdapter,
    WarpBrownianCoagulationExecutionState,
    WarpBrownianCoagulationState,
)
```

Neither `particula.execution` nor the top-level `particula` package exports
these concrete names. The exact marker fails closed before selected dispatch
for charged, sedimentation, turbulent shear, combined, or unknown mechanisms,
non-`particle_resolved` requests, and marker subclasses. These are
selection-owned capability errors; native direct-kernel physical, device, and
schema errors retain their native behavior.

The Warp state accepts exactly one thermo form: provide both `temperature` and
`pressure` with `environment=None`, or provide `temperature=None` and
`pressure=None` with a `WarpEnvironmentData`. Optional `volume` is valid in
either form. Mixing an environment with either direct thermo input fails
adapter-owned form validation.

The caller retains the aerosol or resident particles, environment or direct
thermo/volume values, optional collision outputs, and required persistent
`(n_boxes,)` `wp.uint32` RNG sidecar by identity. Supplied diagnostic outputs
have an adapter identity guarantee; omitted direct-kernel diagnostics remain
native call-local convenience results. Diagnostics do not imply a host
readback. Seed the sidecar on the first selected Warp dispatch with
`initialize_rng=True`; ordinary later dispatches reuse that identical sidecar
with `initialize_rng=False`. Only `initialize_rng=True` resets it, even when
`rng_seed` repeats.

Callers control conversion, restoration, device placement, and
`wp.synchronize()` before observing or resolving device state. The adapter
does not upload, restore, synchronize, select a fallback, reseed per step,
retry, create a session or scheduler, or roll back state after launch. Resident sessions
resident-session lifecycle, resident scheduling scheduling, and resident RNG and restart checkpoint/restart
stream policy remain explicitly deferred.

For a runnable selected-Brownian introduction, see the
[GPU coagulation example](https://github.com/Gorkowski/particula/blob/main/docs/Examples/gpu_coagulation_direct.py). It
keeps explicit conversion, synchronization, restoration, and caller-owned
collision/RNG sidecars; it is not a CPU fallback or a high-level `Runnable`
workflow. The direct-kernel evidence/command record above remains the complete
approved-mask evidence boundary.

