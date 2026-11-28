# particula.data.profile

*module*

Portable measurement profiles and source-located diagnostics.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L1-L1)

## particula.data.profile.ProfileSource

*attribute* · *module attribute*

```python
ProfileSource = str | PathLike[str] | dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L16-L16)

## particula.data.profile._BUILTINS

*attribute* · *module attribute*

```python
_BUILTINS = ('cpc-3010', 'smps-aim')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L18-L18)

## particula.data.profile._RESERVED

*attribute* · *module attribute*

```python
_RESERVED = {'scan_id', 'source_sha256', 'time_utc', 'source_line_start', 'source_line_end', 'repaired', 'grid_id', 'vendor_status', 'vendor_flags', 'scan_duration_s', 'retrace_duration_s'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L19-L31)

## particula.data.profile.ProfileError

*class*

```python
class ProfileError(ValueError)
```

Bases: `ValueError`

A machine-readable diagnostic for a profile or its source binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L34-L67)

### particula.data.profile.ProfileError.code

*attribute* · *instance attribute*

```python
code = code
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L64-L64)

### particula.data.profile.ProfileError.expected

*attribute* · *instance attribute*

```python
expected = expected
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L61-L61)

### particula.data.profile.ProfileError.message

*attribute* · *instance attribute*

```python
message = message
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L66-L66)

### particula.data.profile.ProfileError.observed

*attribute* · *instance attribute*

```python
observed = observed
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L60-L60)

### particula.data.profile.ProfileError.path

*attribute* · *instance attribute*

```python
path = path
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L65-L65)

### particula.data.profile.ProfileError.severity

*attribute* · *instance attribute*

```python
severity = 'error'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L59-L59)

### particula.data.profile.ProfileError.source_line_end

*attribute* · *instance attribute*

```python
source_line_end = source_line_end
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L63-L63)

### particula.data.profile.ProfileError.source_line_start

*attribute* · *instance attribute*

```python
source_line_start = source_line_start
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L62-L62)

### particula.data.profile.ProfileError.__init__

*method*

```python
def __init__(code: str, path: str, message: str, *, observed: object = None, expected: object = None, source_line_start: int | None = None, source_line_end: int | None = None) -> None
```

Describe a profile error and its location.

**Parameters**

- `code` (`str`) — Machine-readable diagnostic identifier.
- `path` (`str`) — JSON Pointer within the profile, or empty for its root.
- `message` (`str`) — Human-readable explanation.
- `observed` (`object`) (default: `None`) — Value that failed validation, when available.
- `expected` (`object`) (default: `None`) — Constraint or expected value, when available.
- `source_line_start` (`int | None`) (default: `None`) — First physical source line, counted from one.
- `source_line_end` (`int | None`) (default: `None`) — Last physical source line, inclusive.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L37-L67)

## particula.data.profile.ValidationResult

*class* · *dataclass*

```python
class ValidationResult
```

Result of checking one profile and optionally its source header.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L70-L79)

### particula.data.profile.ValidationResult.errors

*attribute* · *class attribute* · *instance attribute*

```python
errors: tuple[ProfileError, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L74-L74)

### particula.data.profile.ValidationResult.__init__

*method*

```python
def __init__(errors: tuple[ProfileError, ...] = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L0-L0)

### particula.data.profile.ValidationResult.raise_for_errors

*method*

```python
def raise_for_errors() -> None
```

Raise the first ProfileError if validation recorded any errors.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L76-L79)

## particula.data.profile._Loader

*class*

```python
class _Loader(yaml.SafeLoader)
```

Bases: `yaml.SafeLoader`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L93-L95)

### particula.data.profile._Loader.construct_mapping

*method*

```python
def construct_mapping(node, deep=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L94-L95)

## particula.data.profile._decode

*function*

```python
def _decode(profile)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L146-L162)

## particula.data.profile._pairs

*function*

```python
def _pairs(pairs)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L82-L90)

## particula.data.profile._resource

*function*

```python
def _resource(name)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L114-L119)

## particula.data.profile._validate_array

*function*

```python
def _validate_array(field, index)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L165-L189)

## particula.data.profile._validator

*function* · *cached*

```python
def _validator()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L122-L124)

## particula.data.profile.canonical

*function*

```python
def canonical(value: object) -> bytes
```

Serialize JSON values deterministically for provenance.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L98-L106)

## particula.data.profile.describe_profile

*function*

```python
def describe_profile(profile: ProfileSource | None = None) -> dict
```

Describe the profile format or resolve a particular profile.

**Parameters**

- `profile` (`ProfileSource | None`) (default: `None`) — Built-in ID, mapping, or path to a JSON or YAML file. Omit to retrieve the installed JSON Schema.

**Returns**

- (`dict`) — A fresh dictionary containing the schema or the validated profile,
- (`dict`) — including its declared quantities, units and clock interpretation.

**Raises**

- `ProfileError` — The supplied profile cannot be loaded or validated.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L127-L143)

## particula.data.profile.digest

*function*

```python
def digest(value: object) -> str
```

Return the SHA-256 of a normalized definition.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L109-L111)

## particula.data.profile.load_profile

*function*

```python
def load_profile(profile: ProfileSource) -> dict
```

Load a profile and validate its schema, names, axes and time zone.

**Parameters**

- `profile` (`ProfileSource`) — Built-in ID, mapping, or path to a JSON or YAML file.

**Returns**

- (`dict`) — Validated profile independent of the caller's input mapping.

**Raises**

- `ProfileError` — The input cannot be decoded or violates a profile
constraint. The diagnostic identifies the location when available.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L192-L252)

## particula.data.profile.validate_profile

*function*

```python
def validate_profile(profile: ProfileSource, *, source: str | PathLike[str] | None = None) -> ValidationResult
```

Check a profile and optionally bind it to the first source layout.

**Parameters**

- `profile` (`ProfileSource`) — Built-in ID, mapping, or path to a JSON or YAML file.
- `source` (`str | PathLike[str] | None`) (default: `None`) — Optional delimited export for checking column selectors, coordinate definitions and preamble requirements.

**Returns**

- (`ValidationResult`) — Validation result with the first profile diagnostic, or an empty
- (`ValidationResult`) — error tuple on success.

**Raises**

- `OSError` — The supplied source cannot be opened or read.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L255-L280)
