# Dilution Strategy System

## Complete direct-process context

CPU dilution remains the semantic source. The direct GPU step is one call in the
illustrative [explicit-transfer sequence](https://github.com/Gorkowski/particula/blob/main/docs/examples/gpu/direct/gpu_complete_process_sequence.py), backed by [direct-process validation evidence](https://github.com/Gorkowski/particula/blob/main/particula/gpu/tests/process_sequence_test.py) and its [example regression](https://github.com/Gorkowski/particula/blob/main/examples_tests/gpu_complete_process_sequence_example_test.py). It does not create public coordination, hidden transfer, CPU fallback, or a resident loop.

The CPU dilution system provides a bounded chamber-loss process for a single
`Aerosol`. It is the completed CPU contract from parent **GPU process completeness**. **GPU dilution** is a
downstream consumer of this CPU reference only; this guide does not deliver
GPU dilution or CPU/GPU parity.

## Public API and equation

Construct the public strategy and runnable through `particula.dynamics`:

```python
import numpy as np
import particula as par

particles = par.particles.create_resolved_particle_data(
    masses=[1e-18, 2e-18], density=1000.0,
    concentration=[4.0, 8.0], volume=2.0,
)
gas = par.gas.GasData(
    name=["vapor", "inert"], molar_mass=np.array([0.1, 0.04]),
    concentration=np.array([[3e-6, 5e-6]]),
    partitioning=np.array([True, False]),
)
environment = par.gas.EnvironmentData(
    temperature=[298.15], pressure=[101325.0], saturation_ratio=[[1.0, 0.0]],
)
aerosol = par.Aerosol(particles=particles, gas=gas, environment=environment)

strategy = par.dynamics.DilutionStrategy(coefficient=0.1)
dilution = par.dynamics.Dilution(strategy)
result = dilution.execute(aerosol, time_step=10.0, sub_steps=2)
assert result is aerosol
```

Execution mutates the supplied `Aerosol` and returns that same `Aerosol`
object. For chamber volume `V` [m³] and dilution flow `Q` [m³/s], the scalar
coefficient is `alpha = Q / V` [s⁻¹]. Every supported concentration follows
the exact exponential update

`c_new = c * exp(-alpha * time_step)`.

Here `time_step` is in s. Native resolved particle storage holds counts;
PMF storage holds bin number density (m⁻³), and PDF storage holds `dn/dr`
(m⁻⁴). All receive the same multiplicative decay at fixed volume. All lanes of
the unified `GasData.concentration` (kg/m³) decay, regardless of partitioning.
The native runnable supports one box; storage shapes do not imply multi-box
transport support.

## Execution, validation, and recovery

`sub_steps` must be a positive, non-boolean Python or NumPy integer. For the
concrete exact-exponential strategy, the runnable performs one total-duration
operation, independent of `sub_steps`, so the whole call is retry-safe. Custom
compatible strategies retain equal `time_step / sub_steps` slice delegation.
The coefficient and duration must be finite, nonnegative real scalars;
complex values are rejected, and booleans and non-scalars are rejected.

A zero coefficient or zero duration is an exact no-op, but supported state is
still preflighted even for a no-op. Concrete CPU dilution supports exactly one
particle box and validates state
before mutation. If validation fails, state is unchanged; if an unexpected
CPU write failure occurs, it restores attempted concentration writes from
snapshots. Held containers and destination-array identities are retained.
These atomicity and retry-safe failure guarantees let callers correct input and
retry safely. Extreme finite decay may underflow to zero, but outputs remain
finite and nonnegative.

Dilution changes only particle number concentration and gas mass
concentrations. It protects particle distribution/mass, charge, density,
representation volume, gas names, molar masses, partitioning metadata, and
environment temperature, pressure and saturation ratios.

## Boundaries and example

This CPU scalar contract excludes inlet composition or source terms, multi-box
particle data and transport, GPU, Warp, CUDA, alternate-backend implementation or parity,
backend selection, and performance claims.

`dilute_aerosol` and `get_dilution_step` are concrete-module-only helpers;
they are not public `particula.dynamics` imports. Use the public strategy and
runnable above for supported execution.

The deterministic, hardware-free example uses only this public API:

- Source: <https://github.com/Gorkowski/particula/blob/main/docs/examples/processes/dilution/cpu_dilution.py>
- Run: `python docs/examples/processes/dilution/cpu_dilution.py`

Focused regression commands are:

```bash
pytest examples_tests/dilution_example_test.py -q -Werror
pytest particula/dynamics/tests/dilution_test.py -q -Werror
pytest particula/dynamics/tests/dilution_runnable_test.py -q -Werror
pytest particula/dynamics/tests/dilution_exports_test.py -q -Werror
```
