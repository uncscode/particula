# particula.dynamics.chemistry.ncar

*module*

Import the closed single-gas-phase subset of NCAR 1.3.0 mechanisms.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/ncar.py#L1-L1)

## particula.dynamics.chemistry.ncar._at

*function*

```python
def _at(location)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/ncar.py#L20-L25)

## particula.dynamics.chemistry.ncar._catalog

*function*

```python
def _catalog(records, phases)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/ncar.py#L44-L70)

## particula.dynamics.chemistry.ncar._entries

*function*

```python
def _entries(records, location, label='name') -> Iterator[dict]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/ncar.py#L28-L41)

## particula.dynamics.chemistry.ncar._reaction

*function*

```python
def _reaction(item, catalog, phase)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/ncar.py#L92-L125)

## particula.dynamics.chemistry.ncar._side

*function*

```python
def _side(records, side, catalog)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/ncar.py#L73-L89)

## particula.dynamics.chemistry.ncar.convert_ncar

*function*

```python
def convert_ncar(source, destination)
```

Validate and atomically write native JSON or YAML at destination.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/ncar.py#L155-L170)

## particula.dynamics.chemistry.ncar.import_ncar

*function*

```python
def import_ncar(path) -> ReactionMechanism
```

Read an NCAR 1.3.0 JSON or YAML file into a native gas mechanism.

Gas concentrations are mol/m³ of air. Optional source molecular weights
(kg/mol) are retained for validation when binding to simulation species.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/ncar.py#L128-L152)

## particula.dynamics.chemistry.ncar.main

*function*

```python
def main()
```

Convert a supported NCAR file to inspectable native JSON or YAML.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/ncar.py#L173-L182)
