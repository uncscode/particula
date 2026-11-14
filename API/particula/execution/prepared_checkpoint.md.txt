# particula.execution.prepared_checkpoint

*module*

Prepared configuration around canonical resident continuation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_checkpoint.py#L1-L1)

## particula.execution.prepared_checkpoint._host_state

*function*

```python
def _host_state(resident, primary) -> Aerosol
```

Materialize validation state from canonical bytes, never inspection.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_checkpoint.py#L40-L77)

## particula.execution.prepared_checkpoint._preflight

*function*

```python
def _preflight(processes, saved, backend, device)
```

Reject schema/configuration/device drift before any allocation/upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_checkpoint.py#L80-L132)

## particula.execution.prepared_checkpoint._restore_after_binding

*function*

```python
def _restore_after_binding(owner, saved)
```

Undo setup-only refresh/reset writes using canonical bytes once more.

Recompilation reconstructs private scratch and constant bindings. Some
direct adapters initialize shared registry buffers and thermodynamics at
setup, so restore their exact checkpoint contents after compilation. Keep
staging alive until the dedicated stream finishes every copy.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_checkpoint.py#L166-L198)

## particula.execution.prepared_checkpoint._validate_auxiliary

*function*

```python
def _validate_auxiliary(saved, plans)
```

Validate the optional last-step energy before setup or upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_checkpoint.py#L135-L163)

## particula.execution.prepared_checkpoint.checkpoint

*function*

```python
def checkpoint(owner: Any) -> SimulationCheckpoint
```

Observe errors before saving canonical bytes and last-step results.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_checkpoint.py#L14-L37)

## particula.execution.prepared_checkpoint.resume_simulation

*function*

```python
def resume_simulation(processes: tuple[Any, ...], saved: SimulationCheckpoint, *, backend: Backend = Backend.WARP, device: str | None = None)
```

Restore owned arrays and counters on the exact checkpoint device.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_checkpoint.py#L201-L230)
