# particula.execution.prepared_transfer

*module*

An explicit, synchronous host boundary within a prepared sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_transfer.py#L1-L1)

## particula.execution.prepared_transfer.compile_transfer

*function*

```python
def compile_transfer(plan, owner)
```

Retain upload staging and execute the explicit CPU callback.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_transfer.py#L33-L91)

## particula.execution.prepared_transfer.download_state

*function*

```python
def download_state(owner)
```

Download full state after the caller has drained the owner stream.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_transfer.py#L9-L30)
