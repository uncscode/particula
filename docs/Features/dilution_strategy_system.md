# Dilution Strategy System

The CPU dilution system provides a bounded chamber-loss process for a single
`Aerosol`. It is the completed CPU contract from parent **E6**. **E6-F2** is a
downstream consumer of this CPU reference only; this guide does not deliver
E6-F2 or CPU/GPU parity.

## Public API and equation

Construct the public strategy and runnable through `particula.dynamics`:

```python
import particula as par

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

Here `time_step` is in [s], particle number concentration is in [1/m³], and
gas mass concentration is in [kg/m³]. The runnable operates on CPU scalar
state: elementwise particle concentration, scalar partitioning gas
concentration, and multi-species gas-only concentration are supported. This
does not imply multi-box transport support.

## Execution, validation, and recovery

`sub_steps` must be a positive, non-boolean Python or NumPy integer. The
runnable delegates equal `time_step / sub_steps` slices, whose product gives
the same whole-step exact exponential factor. The coefficient and duration
must be finite, nonnegative scalars; booleans and non-scalars are rejected.

A zero coefficient or zero duration is an exact no-op, but supported state is
still preflighted even for a no-op. Concrete CPU dilution validates state
before mutation. If validation fails, state is unchanged; if an unexpected
setter failure occurs, already-written concentration state is rolled back.
These atomicity and retry-safe failure guarantees let callers correct input and
retry safely. Extreme finite decay may underflow to zero, but outputs remain
finite and nonnegative.

Dilution changes only particle number concentration and gas mass
concentrations. It protects particle distribution/mass, charge, density,
representation volume, gas names, molar masses, partitioning metadata, and
atmospheric temperature and pressure.

## Boundaries and example

This CPU scalar contract excludes inlet composition or source terms, multi-box
transport, GPU, Warp, CUDA, alternate-backend implementation or parity,
backend selection, and performance claims.

`dilute_aerosol` and `get_dilution_step` are concrete-module-only helpers;
they are not public `particula.dynamics` imports. Use the public strategy and
runnable above for supported execution.

The deterministic, hardware-free example uses only this public API:

- Source: <https://github.com/Gorkowski/particula/blob/main/docs/Examples/cpu_dilution.py>
- Run: `python docs/Examples/cpu_dilution.py`

Focused regression commands are:

```bash
pytest particula/tests/dilution_docs_test.py -q -Werror
pytest particula/dynamics/tests/dilution_test.py -q -Werror
pytest particula/dynamics/tests/dilution_runnable_test.py -q -Werror
pytest particula/dynamics/tests/dilution_exports_test.py -q -Werror
```
