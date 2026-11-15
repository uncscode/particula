# particula.execution.prepared_extensions

*module*

Connect explicit public GPU adapters to the resident preparation boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_extensions.py#L1-L1)

## particula.execution.prepared_extensions.clock_operation

*function*

```python
def clock_operation(owner)
```

Retain one guarded device-clock advance after the complete sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_extensions.py#L161-L179)

## particula.execution.prepared_extensions.compile_extension

*function*

```python
def compile_extension(plan, owner)
```

Pass a narrow resource context, never the owner, to user preparation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_extensions.py#L66-L158)

## particula.execution.prepared_extensions.plan_extension

*function*

```python
def plan_extension(adapter, process, aerosol)
```

Freeze validated extension configuration before device allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_extensions.py#L13-L63)
