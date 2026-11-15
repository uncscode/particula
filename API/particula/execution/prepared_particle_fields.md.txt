# particula.execution.prepared_particle_fields

*module*

Compose native event decisions with registered particle-field ownership.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_particle_fields.py#L1-L1)

## particula.execution.prepared_particle_fields._wrap

*function*

```python
def _wrap(context: BuiltinPreparationContext, plan, operation, before, fields)
```

Prepare a single family's event launches against fixed buffers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_particle_fields.py#L123-L251)

## particula.execution.prepared_particle_fields.attach_transport

*function*

```python
def attach_transport(prepared, context: BuiltinPreparationContext)
```

Retain sparse overlap records from the native resampling sweep.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_particle_fields.py#L45-L61)

## particula.execution.prepared_particle_fields.validate_particle_fields

*function*

```python
def validate_particle_fields(aerosol, plans)
```

Reject undefined particle events before uploading any state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_particle_fields.py#L17-L42)

## particula.execution.prepared_particle_fields.wrap_particle_fields

*function*

```python
def wrap_particle_fields(context: BuiltinPreparationContext, plans, operations)
```

Retain lifecycle operations without per-step allocations or readbacks.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_particle_fields.py#L64-L120)
