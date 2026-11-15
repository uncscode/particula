# particula.extension_fields

*module*

Explicit descriptors and storage for persistent user simulation fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L1-L1)

## particula.extension_fields._DTYPES

*attribute* · *module attribute*

```python
_DTYPES = {np.dtype('float64'), np.dtype('int32'), np.dtype('int64')}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L15-L15)

## particula.extension_fields._LAYOUTS

*attribute* · *module attribute*

```python
_LAYOUTS = {('box'), ('box', 'particle'), ('box', 'gas_species'), ('box', 'particle_species'), ('box', 'particle', 'particle_species')}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L8-L14)

## particula.extension_fields.ExtensionFields

*class*

```python
class ExtensionFields
```

Own named numeric arrays with explicit shape and dtype admission.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L94-L274)

### particula.extension_fields.ExtensionFields._arrays

*attribute* · *instance attribute*

```python
_arrays: dict[str, np.ndarray] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L101-L101)

### particula.extension_fields.ExtensionFields._dimensions

*attribute* · *instance attribute*

```python
_dimensions = dict(dimensions)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L99-L99)

### particula.extension_fields.ExtensionFields._ledgers

*attribute* · *instance attribute*

```python
_ledgers: dict[str, np.ndarray] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L102-L102)

### particula.extension_fields.ExtensionFields._specs

*attribute* · *instance attribute*

```python
_specs: dict[str, FieldSpec] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L100-L100)

### particula.extension_fields.ExtensionFields.specs

*property*

```python
specs
```

Return a read-only view of registered descriptors.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L105-L107)

### particula.extension_fields.ExtensionFields.__getitem__

*method*

```python
def __getitem__(name)
```

Return a held numeric field by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L109-L111)

### particula.extension_fields.ExtensionFields.__init__

*method*

```python
def __init__(dimensions)
```

Bind named axis sizes from an owning aerosol schema.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L97-L102)

### particula.extension_fields.ExtensionFields._activate_particles

*method*

```python
def _activate_particles(before_counts, after_counts)
```

Initialize newly active slots and record their additive inventory.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L237-L249)

### particula.extension_fields.ExtensionFields._resample_particles

*method*

```python
def _resample_particles(plan)
```

Transport tracers using the native remap's exact count overlaps.

Called on private source-transaction state after native plan validation.
Changing representation adds no source or removal inventory.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L214-L235)

### particula.extension_fields.ExtensionFields.add

*method*

```python
def add(spec, values=None)
```

Register owned storage without replacing an existing field.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L113-L138)

### particula.extension_fields.ExtensionFields.box_view

*method*

```python
def box_view(box)
```

Retain writable slices of staged storage for one box.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L152-L163)

### particula.extension_fields.ExtensionFields.copy

*method*

```python
def copy()
```

Detach every field while retaining immutable descriptors.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L140-L146)

### particula.extension_fields.ExtensionFields.inventory_changes

*method*

```python
def inventory_changes(name)
```

Return detached cumulative source and removal inventories per box.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L255-L260)

### particula.extension_fields.ExtensionFields.items

*method*

```python
def items()
```

Iterate registered names and arrays in registration order.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L148-L150)

### particula.extension_fields.ExtensionFields.merge_particles

*method*

```python
def merge_particles(donor, recipient)
```

Apply admitted single-box resolved pairs to additive quantities.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L199-L212)

### particula.extension_fields.ExtensionFields.particle_names

*method*

```python
def particle_names()
```

Return names whose values follow resolved particle slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L185-L191)

### particula.extension_fields.ExtensionFields.require_additive

*method*

```python
def require_additive()
```

Reject undefined merger rules before selecting collisions.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L193-L197)

### particula.extension_fields.ExtensionFields.retire_particles

*method*

```python
def retire_particles(before_counts, after_counts)
```

Account count-only sinks, then clear retired slot values.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L262-L274)

### particula.extension_fields.ExtensionFields.transaction_arrays

*method*

```python
def transaction_arrays()
```

Expose internal inventory arrays to transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L251-L253)

### particula.extension_fields.ExtensionFields.validate_layout

*method*

```python
def validate_layout(dimensions, population, active)
```

Validate shape, population semantics and active-field values.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L165-L183)

## particula.extension_fields.FieldSpec

*class* · *dataclass*

```python
class FieldSpec
```

Describe persistent state; units are metadata, not a conversion rule.

Particle policies are passive or additive per physical particle. Births
use initial_value, while retired slots use zero. Bounds apply to active
state; finite values are always required.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L18-L91)

### particula.extension_fields.FieldSpec.axes

*attribute* · *instance attribute*

```python
axes: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L28-L28)

### particula.extension_fields.FieldSpec.dtype

*attribute* · *class attribute* · *instance attribute*

```python
dtype: str = 'float64'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L30-L30)

### particula.extension_fields.FieldSpec.initial_value

*attribute* · *class attribute* · *instance attribute*

```python
initial_value: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L31-L31)

### particula.extension_fields.FieldSpec.maximum

*attribute* · *class attribute* · *instance attribute*

```python
maximum: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L33-L33)

### particula.extension_fields.FieldSpec.minimum

*attribute* · *class attribute* · *instance attribute*

```python
minimum: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L32-L32)

### particula.extension_fields.FieldSpec.name

*attribute* · *instance attribute*

```python
name: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L27-L27)

### particula.extension_fields.FieldSpec.particle_policy

*attribute* · *class attribute* · *instance attribute*

```python
particle_policy: str = 'passive'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L34-L34)

### particula.extension_fields.FieldSpec.units

*attribute* · *instance attribute*

```python
units: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L29-L29)

### particula.extension_fields.FieldSpec.version

*attribute* · *class attribute* · *instance attribute*

```python
version: int = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L35-L35)

### particula.extension_fields.FieldSpec.__init__

*method*

```python
def __init__(name: str, axes: tuple[str, ...], units: str, dtype: str = 'float64', initial_value: float = 0.0, minimum: float | None = None, maximum: float | None = None, particle_policy: str = 'passive', version: int = 1) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L0-L0)

### particula.extension_fields.FieldSpec.__post_init__

*method*

```python
def __post_init__()
```

Reject ambiguous schemas before allocating storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L37-L82)

### particula.extension_fields.FieldSpec.validate

*method*

```python
def validate(values)
```

Check finite values and declared physical bounds.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L84-L91)

## particula.extension_fields.field_dimensions

*function*

```python
def field_dimensions(particles, gas)
```

Resolve independent particle-material and gas-species widths.

[View source](https://github.com/uncscode/particula/blob/main/particula/extension_fields.py#L277-L284)
