# particula.util.chemical.chemical_properties

*module*

Get selected thermodynamic properties of a chemical at STP.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/chemical/chemical_properties.py#L1-L1)

## particula.util.chemical.chemical_properties.get_chemical_stp_properties

*function*

```python
def get_chemical_stp_properties(identifier: str) -> Dict[str, float]
```

Return selected thermodynamic properties of *identifier* at STP.

Parameters
----------
identifier
    Any string accepted by ``thermo.Chemical`` (name, formula, CAS).

Returns:
-------
dict
    {
        "molar_mass": kg mol⁻¹,
        "density": kg m⁻³,
        "surface_tension": N m⁻¹,
        "pure_vapor_pressure": Pa,
    }

[View source](https://github.com/uncscode/particula/blob/main/particula/util/chemical/chemical_properties.py#L8-L46)
