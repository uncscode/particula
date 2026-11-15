# particula.execution.prepared_cohort

*module*

Prepare and checkpoint original-seed lineage tracking on resident arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_cohort.py#L1-L1)

## particula.execution.prepared_cohort.compile_cohort

*function*

```python
def compile_cohort(_plan, _owner)
```

The observer is attached around physical operations after compilation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_cohort.py#L24-L26)

## particula.execution.prepared_cohort.plan_cohort

*function*

```python
def plan_cohort(process, aerosol)
```

Freeze the initial labeling rule and first-passage threshold.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_cohort.py#L6-L21)

## particula.execution.prepared_cohort.validate_cohort_checkpoint

*function*

```python
def validate_cohort_checkpoint(saved)
```

Check outcome partition and unresolved labels before any upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_cohort.py#L150-L173)

## particula.execution.prepared_cohort.validate_cohort_plans

*function*

```python
def validate_cohort_plans(plans)
```

Reject untracked removal, resampling and marker-material sources.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_cohort.py#L29-L62)

## particula.execution.prepared_cohort.wrap_cohort_operations

*function*

```python
def wrap_cohort_operations(owner, plans, operations)
```

Initialize labels and observe after each ordered physical process.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_cohort.py#L65-L147)
