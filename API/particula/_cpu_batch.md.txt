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

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L34-L52)

## particula._cpu_batch.box_view

*function*

```python
def box_view(aerosol, box)
```

Expose one staged box through existing native container types.

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L111-L138)

## particula._cpu_batch.detached

*function*

```python
def detached(aerosol)
```

Stage independent containers preserving population metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L102-L108)

## particula._cpu_batch.duration

*function*

```python
def duration(time_step, sub_steps)
```

Validate scalar controls before allocating or evaluating state.

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L14-L31)

## particula._cpu_batch.publish

*function*

```python
def publish(aerosol, candidate, writers)
```

Publish selected fields by identity with rollback on write failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L141-L153)

## particula._cpu_batch.validate

*function*

```python
def validate(aerosol, writers=())
```

Validate original storage before detached copies can hide defects.

[View source](https://github.com/uncscode/particula/blob/main/particula/_cpu_batch.py#L55-L99)
