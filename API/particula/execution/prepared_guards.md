# particula.execution.prepared_guards

*module*

Persistent device failure admission for prepared scientific sequences.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_guards.py#L1-L1)

## particula.execution.prepared_guards._GAS_CONSTANT

*attribute* · *module attribute*

```python
_GAS_CONSTANT = wp.constant(wp.float64(GAS_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_guards.py#L9-L9)

## particula.execution.prepared_guards.equal_active_counts

*function*

```python
def equal_active_counts(counts: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Reject an evolving unequal-weight cohort before any collision writer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_guards.py#L12-L31)

## particula.execution.prepared_guards.latch_nucleation_failure

*function*

```python
def latch_nucleation_failure(pipeline: wp.array(dtype=wp.int32), p1: wp.array(dtype=wp.int32), p2: wp.array(dtype=wp.int32), p3: wp.array(dtype=wp.int32), molecule: wp.array(dtype=wp.int32), p4: wp.array(dtype=wp.int32), resampling_invalid: wp.array(dtype=wp.int32), resampling_failed: wp.array(dtype=wp.int32), scaling: wp.array(dtype=wp.int32), final: wp.array(dtype=wp.int32), commit: wp.array(dtype=wp.int32), saved: wp.array(dtype=wp.int32), failure: wp.array(dtype=wp.int32))
```

Retain the first source failure and its phase diagnostics forever.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_guards.py#L34-L68)

## particula.execution.prepared_guards.refresh_saturation

*function*

```python
def refresh_saturation(concentration: wp.array2d(dtype=wp.float64), molar_mass: wp.array(dtype=wp.float64), vapor_pressure: wp.array2d(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), saturation: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Refresh full-gas ratios only while the sequence remains valid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_guards.py#L71-L90)
