# particula.execution.prepared_atmospheric

*module*

Compile explicit ion collision laws into the existing prepared sampler.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_atmospheric.py#L1-L1)

## particula.execution.prepared_atmospheric.configure_atmospheric

*function*

```python
def configure_atmospheric(plan, context: BuiltinPreparationContext, prepared)
```

Retain derived classification and rate scratch; no checkpoint payload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_atmospheric.py#L33-L127)

## particula.execution.prepared_atmospheric.plan_atmospheric

*function*

```python
def plan_atmospheric(process, _aerosol)
```

Freeze the complete physical collision configuration for restart.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_atmospheric.py#L9-L30)

## particula.execution.prepared_atmospheric.validate_atmospheric_companions

*function*

```python
def validate_atmospheric_companions(plans)
```

Reject conflicting source/cleanup/exclusion material identities.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_atmospheric.py#L130-L158)
