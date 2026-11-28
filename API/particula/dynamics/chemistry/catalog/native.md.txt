# particula.dynamics.chemistry.catalog.native

*module*

Compile selected closed gas channels to Particula's native schema.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/native.py#L1-L1)

## particula.dynamics.chemistry.catalog.native.ALIASES

*attribute* · *module attribute*

```python
ALIASES = {'HO': 'OH', 'O3P': 'O', 'O1D_CB4': 'O1D', 'HCHO': 'CH2O'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/native.py#L50-L50)

## particula.dynamics.chemistry.catalog.native.FORMULAS

*attribute* · *module attribute*

```python
FORMULAS = {name: name for name in ('O', 'O1D', 'O2', 'O3', 'H', 'H2', 'OH', 'HO2', 'H2O', 'H2O2', 'N', 'N2', 'NO', 'NO2', 'NO3', 'N2O', 'N2O5', 'HONO', 'HNO3', 'CO', 'CO2', 'CH4', 'SO', 'SO2', 'SO3', 'H2SO4', 'H2S', 'CH3O2', 'CH3OOH', 'CH2O', 'HCHO', 'C2H6', 'C2H4')}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/native.py#L11-L48)

## particula.dynamics.chemistry.catalog.native._elements

*function*

```python
def _elements(name)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/native.py#L53-L57)

## particula.dynamics.chemistry.catalog.native._normalize

*function*

```python
def _normalize(side, declared)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/native.py#L60-L68)

## particula.dynamics.chemistry.catalog.native._reservoirs

*function*

```python
def _reservoirs(record, lhs, rhs, close_oxygen)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/native.py#L71-L91)

## particula.dynamics.chemistry.catalog.native.compile_channel

*function*

```python
def compile_channel(record, *, close_oxygen=False)
```

Return a native reaction and reviewed atomic compositions.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/native.py#L94-L140)

## particula.dynamics.chemistry.catalog.native.native_mechanism

*function*

```python
def native_mechanism(records, *, close_oxygen=False)
```

Build one validated closed mechanism from a pinned selection.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/native.py#L156-L187)

## particula.dynamics.chemistry.catalog.native.selected_channel

*function*

```python
def selected_channel(record, *, close_oxygen=False)
```

Read a released native reaction or convert a locally imported record.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/native.py#L143-L153)
