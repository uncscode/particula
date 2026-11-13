# particula.util.chemical.chemical_search

*module*

Helper utilities to resolve chemical identifiers to a canonical form.

The functions rely on the optional ``chemicals.identifiers`` back-end for
fast, exact look-ups and fall back to fuzzy matching the PubChem name
index when that package is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/chemical/chemical_search.py#L1-L1)

## particula.util.chemical.chemical_search._pubchem_db

*attribute* · *module attribute*

```python
_pubchem_db = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/util/chemical/chemical_search.py#L18-L18)

## particula.util.chemical.chemical_search.get_chemical_search

*function*

```python
def get_chemical_search(identifier: str) -> str
```

Resolve a chemical identifier to a canonical name or CAS number.

The resolution proceeds in two stages:

1. Exact search through ``chemicals.identifiers.CAS_from_any``
   (returns immediately when successful).
2. Case-insensitive fuzzy matching against the PubChem name index that
   ships with *thermo* (Levenshtein distance via ``difflib``).

**Parameters**

- `identifier` (`str`) — Arbitrary chemical name, formula, InChI, or CAS registry number.

**Returns**

- (`str`) — The resolved identifier string when a match is found, otherwise ``None``.

**Examples**

``` py title="Exact CAS lookup"
from particula.util.materials.chemical_search import get_chemical_search
assert get_chemical_search("64-17-5") == "64-17-5"      # ethanol
```
``` py title="Fuzzy text match"
hit = get_chemical_search("soodim chlorid")             # typo
# hit -> "7647-14-5"  (sodium chloride)
```

> **References**
>
> - Wilson, N., et al., "Open-source tools for chemical identifiers,"
>   *J. Chem. Inf. Model.* **60** (2020) 833–839.
> - "Chemical identifier,"
>   [Wikipedia](https://en.wikipedia.org/wiki/Chemical_identifier).

[View source](https://github.com/uncscode/particula/blob/main/particula/util/chemical/chemical_search.py#L24-L83)
