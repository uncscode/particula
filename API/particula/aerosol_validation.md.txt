# particula.aerosol_validation

*module*

Read-only structural alignment of CPU particle, gas, and environment data.

Import this concrete boundary directly; it does not admit a physical process.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_validation.py#L1-L1)

## particula.aerosol_validation._gas_species_count

*function*

```python
def _gas_species_count(names: object) -> int
```

Count the nonempty, unique gas names without changing their order.

**Parameters**

- `names` (`object`) — Stored gas-name metadata to inspect.

**Returns**

- (`int`) — Number of ordered gas species.

**Raises**

- `ValueError` — If names are missing, blank, or duplicated.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_validation.py#L30-L51)

## particula.aerosol_validation._require_containers

*function*

```python
def _require_containers(particles: object, gas: object, environment: object) -> None
```

Check all top-level types before inspecting any stored fields.

**Parameters**

- `particles` (`object`) — Candidate particle container.
- `gas` (`object`) — Candidate gas container.
- `environment` (`object`) — Candidate environment container.

**Raises**

- `TypeError` — If an input is not its required CPU container.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_validation.py#L54-L72)

## particula.aerosol_validation._require_shape

*function*

```python
def _require_shape(value: object, field: str, shape: tuple[int, ...]) -> None
```

Require a raw NumPy array with the indicated rank and exact shape.

**Parameters**

- `value` (`object`) — Stored field to inspect without conversion.
- `field` (`str`) — Name to include in validation errors.
- `shape` (`tuple[int, ...]`) — Required array dimensions.

**Raises**

- `ValueError` — If the field is not an array of the required shape.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_validation.py#L13-L27)

## particula.aerosol_validation.validate_aerosol_structure

*function*

```python
def validate_aerosol_structure(particles: ParticleData, gas: GasData, environment: EnvironmentData) -> None
```

Check raw CPU container dimensions and full ordered gas-name metadata.

This check does not inspect array values, partitioning eligibility,
distribution interpretation, or chemical provenance. In particular, equal
ratio and gas widths do not prove that their species order agrees.

**Parameters**

- `particles` (`ParticleData`) — Particle storage with nonempty box axis.
- `gas` (`GasData`) — Ordered gas storage, including nonpartitioning species.
- `environment` (`EnvironmentData`) — Per-box state with ratio lanes for every gas species.

**Returns**

- (`None`) — None when the stored schemas and shared dimensions agree.

**Raises**

- `TypeError` — If any top-level input is not its required CPU container.
- `ValueError` — If stored fields or cross-container shapes are malformed.

[View source](https://github.com/uncscode/particula/blob/main/particula/aerosol_validation.py#L75-L146)
