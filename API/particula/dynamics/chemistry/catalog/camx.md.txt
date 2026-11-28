# particula.dynamics.chemistry.catalog.camx

*module*

Read CAMx numbered rate decks and PMCAMx-SR precursor yield tables.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/camx.py#L1-L1)

## particula.dynamics.chemistry.catalog.camx.TITRATION_DECKS

*attribute* · *module attribute*

```python
TITRATION_DECKS = {'camx': {'8d267b966a4cf3d9be236f4bbed5fdd52bd3cfe3e463f97c454aa7a9807bfe25'}, 'pmcamx': {'d0a18f9beff7538e20238220e7f57dbc3a49e7ceb134001db6d49b6b4cd1fec9'}}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/camx.py#L17-L24)

## particula.dynamics.chemistry.catalog.camx._deck_entry

*function*

```python
def _deck_entry(raw, line_num, pinned, units, photo, listings, tables)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/camx.py#L117-L159)

## particula.dynamics.chemistry.catalog.camx._deck_header

*function*

```python
def _deck_header(lines)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/camx.py#L58-L89)

## particula.dynamics.chemistry.catalog.camx._deck_rate

*function*

```python
def _deck_rate(kind, values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/camx.py#L92-L100)

## particula.dynamics.chemistry.catalog.camx._deck_units

*function*

```python
def _deck_units(pinned, header)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/camx.py#L103-L114)

## particula.dynamics.chemistry.catalog.camx._soap_aging

*function*

```python
def _soap_aging(text)
```

Read the literal bin gain terms in the published SOAP aging loops.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/camx.py#L285-L340)

## particula.dynamics.chemistry.catalog.camx._soap_tables

*function*

```python
def _soap_tables(text)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/camx.py#L185-L216)

## particula.dynamics.chemistry.catalog.camx.equations

*function*

```python
def equations(text)
```

Read a CAMx mech.* stoichiometry listing and continuation lines.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/camx.py#L27-L55)

## particula.dynamics.chemistry.catalog.camx.rate_deck

*function*

```python
def rate_deck(text, source, digest, listings, tables)
```

Join numbered rate rows to an optional matching mechanism listing.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/camx.py#L162-L182)

## particula.dynamics.chemistry.catalog.camx.soap

*function*

```python
def soap(text, digest=None)
```

Inventory PMCAMx-SR SOAP precursor rates and paired NOx yield tables.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/camx.py#L219-L282)
