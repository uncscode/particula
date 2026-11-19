# particula.execution.prepared_subcycling

*module*

Storage reservations and device-resident undo for adaptive prepared steps.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_subcycling.py#L1-L1)

## particula.execution.prepared_subcycling._bound_children

*function*

```python
def _bound_children(item, wp)
```

Visit the immutable carriers and closures used by native adapters.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_subcycling.py#L78-L92)

## particula.execution.prepared_subcycling.reserve_prepared

*function*

```python
def reserve_prepared(aerosol, plans)
```

Check host shape bounds before upload, including dependent processes.

The direct sampler and primary layouts have explicit dimensions. Composed
adapters reserve space for their optional work arrays even when individual
branches alias them. The retained-array inventory is checked again at bind.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_subcycling.py#L15-L75)

## particula.execution.prepared_subcycling.rollback_buffers

*function*

```python
def rollback_buffers(owner, call)
```

Retain one device backup per unique bound array.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_subcycling.py#L95-L135)
