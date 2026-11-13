# particula.execution.validation

*module*

Shared fail-closed validation for execution declaration carriers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/validation.py#L1-L1)

## particula.execution.validation._NAME_PATTERN

*attribute* · *module attribute*

```python
_NAME_PATTERN = re.compile('^[a-z][a-z0-9_]*$')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/validation.py#L16-L16)

## particula.execution.validation._validate_device

*function*

```python
def _validate_device(device: object, field_name: str) -> None
```

Validate a device value and its backend/native fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/validation.py#L72-L84)

## particula.execution.validation._validate_process

*function*

```python
def _validate_process(process: object, field_name: str) -> None
```

Validate a process value and its identifier-style name.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/validation.py#L87-L94)

## particula.execution.validation._validate_requirements

*function*

```python
def _validate_requirements(requirements: object, field_name: str) -> None
```

Validate an exact immutable capability-requirements declaration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/validation.py#L97-L113)

## particula.execution.validation.validate_capability_matrix

*function*

```python
def validate_capability_matrix(matrix: object, field_name: str) -> None
```

Deeply validate a capability matrix and every declaration it retains.

**Parameters**

- `matrix` (`object`) — Candidate capability matrix carrier.
- `field_name` (`str`) — Qualified public name used in error messages.

**Raises**

- `TypeError` — If a carrier or nested declaration has an invalid type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/validation.py#L41-L69)

## particula.execution.validation.validate_execution_request

*function*

```python
def validate_execution_request(request: object, field_name: str) -> None
```

Deeply validate an execution request and its nested declarations.

**Parameters**

- `request` (`object`) — Candidate execution request carrier.
- `field_name` (`str`) — Qualified public name used in error messages.

**Raises**

- `TypeError` — If a carrier or nested declaration has an invalid type.
- `ValueError` — If a nested value violates its closed declaration contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/validation.py#L19-L38)
