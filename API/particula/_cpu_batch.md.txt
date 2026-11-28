# particula._cpu_batch

*module*

Private CPU batch staging shared by native runnable implementations.

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L1-L1)

## particula._cpu_batch.arrays

*function*

```python
def arrays(aerosol)
```

List stored numeric arrays in a stable ownership order.

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L35-L55)

## particula._cpu_batch.box_view

*function*

```python
def box_view(aerosol, box)
```

Expose one staged box through existing native container types.

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L87-L115)

## particula._cpu_batch.detached

*function*

```python
def detached(aerosol)
```

Stage independent containers preserving population metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L77-L84)

## particula._cpu_batch.duration

*function*

```python
def duration(time_step, sub_steps)
```

Validate scalar controls before allocating or evaluating state.

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L15-L32)

## particula._cpu_batch.publish

*function*

```python
def publish(aerosol, candidate, writers)
```

Publish selected fields by identity with rollback on write failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L118-L130)

## particula._cpu_batch.validate

*function*

```python
def validate(aerosol, writers=())
```

Validate original storage before detached copies can hide defects.

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L58-L74)
