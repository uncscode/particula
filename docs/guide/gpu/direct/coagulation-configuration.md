# Coagulation Configuration

### GPU coagulation configuration and sidecar ownership

Import the supported direct step and its host-side configuration separately:

```python
from particula.gpu.kernels import coagulation_step_gpu
from particula.gpu.kernels.coagulation import CoagulationMechanismConfig
```

Pass immutable `mechanism_config=CoagulationMechanismConfig(...)` as keyword-only
host metadata. The frozen dataclass is not re-exported by
`particula.gpu.kernels`, is not device-resident simulation state, and neither
transfers nor owns Warp state. In contrast,
`WarpParticleData`, supplied `collision_pairs`, supplied `n_collisions`, and
supplied `rng_states` are caller-owned, same-device Warp resources. Omitted
collision outputs and RNG state use call-local convenience allocation. A
supplied RNG sidecar is reused and changes in place; it resets only with
`initialize_rng=True`. Configuration and caller-owned sidecars have no implicit
CPU↔GPU transfer.

The optional `charged_model` selects `hard_sphere` (the default),
`dyachkov2007`, `gatti2008`, `gopalakrishnan2012`, or `chahl2019`.
A nondefault selection requires the `charged_hard_sphere` mechanism identifier,
retained for compatibility. Unknown models reject before particle access.
These fits describe complete charged Brownian rates. For that physical model,
select the charged mechanism alone; adding the ordinary Brownian mechanism
would count both complete rates. Existing explicitly additive calls retain
their original semantics.

Only `distribution_type="particle_resolved"` is accepted. `"discrete"` and
`"continuous_pdf"` raise `ValueError`; they do not fall back or convert.
Malformed configuration and an unsupported distribution fail before particle
access. Mask `7` fails at capability preflight before particle metadata or
enabled-term validation. Masks `11`, `13`, and `14` validate particle metadata
and enabled terms before they fail closed. These read-only preflight boundaries
do not make a later runtime failure transactional: callers retain their mutable
state, and no rollback is guaranteed after such a failure.

| Identifier | Bit | Status |
| --- | --- | --- |
| `brownian` | `1` | Executable singleton |
| `charged_hard_sphere` | `2` | Executable singleton |
| `sedimentation_sp2016` | `4` | Executable singleton |
| `turbulent_shear_st1956` | `8` | Executable singleton |
| `brownian + charged_hard_sphere` | `3` | Executable two-way mask |
| `brownian + sedimentation_sp2016` | `5` | Executable two-way mask |
| `charged_hard_sphere + sedimentation_sp2016` | `6` | Executable two-way mask |
| `brownian + turbulent_shear_st1956` | `9` | Executable two-way mask |
| `charged_hard_sphere + turbulent_shear_st1956` | `10` | Executable two-way mask |
| `sedimentation_sp2016 + turbulent_shear_st1956` | `12` | Executable two-way mask |
| `brownian + charged_hard_sphere + sedimentation_sp2016 + turbulent_shear_st1956` | `15` | Executable four-way mask |
| `brownian + charged_hard_sphere + sedimentation_sp2016` | `7` | Rejected three-way mask |
| `brownian + charged_hard_sphere + turbulent_shear_st1956` | `11` | Rejected three-way mask |
| `brownian + sedimentation_sp2016 + turbulent_shear_st1956` | `13` | Rejected three-way mask |
| `charged_hard_sphere + sedimentation_sp2016 + turbulent_shear_st1956` | `14` | Rejected three-way mask |

The executable singleton masks are `1`, `2`, `4`, and `8`; executable two-term
masks are `3`, `5`, `6`, `9`, `10`, and `12`; and executable four-term mask is
`15`. Deferred masks are exactly `7`, `11`, `13`, and `14`.

Requested tuples normalize to the canonical Brownian, charged, sedimentation,
then turbulent order. Exact SP2016 sedimentation is selected with
`CoagulationMechanismConfig(("sedimentation_sp2016",))`. Its shipped pair rate
is `K = π (r_i + r_j)^2 |v_i - v_j|` in m³/s, where settling velocities use
Stokes settling with Cunningham slip correction (Seinfeld & Pandis, 2016,
Eq. 13A.4). Collision efficiency is fixed at 1; no public efficiency argument
exists. This mode uses fp64 particle mass state, validates finite particle
charge, finite/nonnegative mass and concentration, plus finite/positive density
before mutable work, and requires equal positive particle concentrations within
each box for representable inventory-preserving merges. Non-finite charge is
rejected during host-side preflight before caller-owned outputs, persistent RNG,
or particle state can be mutated.

The exact turbulent-shear singleton is selected with
`CoagulationMechanismConfig(("turbulent_shear_st1956",))`; approved mixed masks
that include this term use the same inputs. Keyword-only `turbulent_dissipation`
is in `m^2/s^3` and `fluid_density` is in `kg/m^3`. Both are positive finite
Python or NumPy floating scalars, or active-device `wp.float64` Warp arrays
shaped `(n_boxes,)`; supplied arrays retain identity and scalars use private
device-local broadcast/property storage. They are required for structurally
valid turbulent requests, including rejected three-way requests during
enabled-term validation, and ignored by non-turbulent masks. They are not
inferred from a container: NumPy arrays, Python lists, and hidden host-to-device
conversion are not supported as array inputs.

Supply temperature and pressure either as direct scalars, supported-float Warp
arrays of shape `(n_boxes,)` on the particle device, or same-device
`WarpEnvironmentData` with both direct inputs set to `None`. These thermodynamic
inputs are not required to be fp64. Caller-owned particle, collision-output, and
RNG resources retain their documented dtypes and identity. Helper-owned,
call-local fp64 work buffers hold per-particle properties as needed; scalar
turbulence inputs use private device-local `(n_boxes,)` broadcast/property
storage, while valid supplied arrays retain identity. Exact SP2016 sedimentation
uses settling storage only for that mechanism; ST1956 has no dedicated
settling-velocity buffer.
It is a direct-kernel path only: it establishes neither CPU-strategy parity nor
general accuracy or performance claims.

Approved masks sanitize and sum fp64 component rates. The safe summed
component-majorant bound satisfies `sum_m K_m(i, j) <= sum_m M_m`; component
maxima can occur on different pairs, so that bound can be conservative.
Production selection obtains its majorant by an exact compact-active scan of the
summed pair rate. One shared active set, candidate stream, acceptance stream,
collision-buffer set, per-box RNG stream, and apply pass serve each call;
mechanisms are not sequential steps. Invalid, nonfinite, or nonpositive terms do
not add to totals, and invalid candidates do not mutate particle or
collision-output state. Persistent RNG state can advance before an invalid-rate
rejection. This documents bounded implementation scope, not throughput or
scaling evidence.
Read-only preflight can use a private device-status buffer and a bounded
synchronization/readback to report invalid caller state without copying,
mutating, or CPU-falling-back caller simulation state.

The return tuple is exactly `(particles, collision_pairs, n_collisions)`; RNG
state is not returned. Accepted collisions mutate caller-owned particle mass,
concentration, and charge in place. Supplied collision buffers are returned by
identity. Supplied `rng_states` are caller-owned same-device Warp state, are
reused and advanced in place, and reset only when `initialize_rng=True`
explicitly opts in; omitted buffers and RNG state are call-local conveniences.
Sedimentation-specific read-only preflight fails before output allocation, RNG
initialization, or particle mutation, but does not make unrelated later runtime
failures atomic. Transfers are explicit. Warp CPU is the baseline when Warp is
installed. CUDA is optional additive evidence and guarded suites skip cleanly
when unavailable. The direct call itself provides no automatic transfer or
CPU fallback.
Brownian and charged models are also available through the separate
[prepared runnable interface](../prepared.md); sedimentation and turbulence
remain direct-only.

Future mechanisms must provide a stable identifier; required host and device
inputs; property preparation; a sanitized additive pair-rate term; a proven-safe
additive majorant; a capability-table row; shared-dispatcher integration; and
co-located `*_test.py` coverage. They must not add an independent sampling
loop, acceptance pass, collision buffer, or RNG stream. This boundary excludes
binned/discrete/continuous-PDF GPU coagulation, high-level `Runnable`
integration, graph-capture claims, alternate mechanisms, DNS or general
turbulence, CPU fallback, hidden transfer, and performance or broad-accuracy
claims. See the [validation/evidence record](../../../contribute/validation/gpu/coagulation.md)
for the focused fixed-mask and stochastic command record.

