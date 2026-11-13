# particula.util.refractive_index_mixing

*module*

Calculate the effective refractive index of a two-solute mixture, using
volume-weighted molar refraction.

To be removed, likely particula_beta only. -kyle

[View source](https://github.com/uncscode/particula/blob/main/particula/util/refractive_index_mixing.py#L1-L1)

## particula.util.refractive_index_mixing.get_effective_refractive_index

*function*

```python
def get_effective_refractive_index(m_zero: Union[float, complex], m_one: Union[float, complex], volume_zero: float, volume_one: float) -> Union[float, complex]
```

Calculate the effective refractive index of a two-solute mixture.

The calculation uses volume-weighted molar refraction, described by:
- r_eff = (v0 / (v0 + v1)) * ((m0 - 1) / (m0 + 2)) +
          (v1 / (v0 + v1)) * ((m1 - 1) / (m1 + 2))
    - r_eff is the effective molar refraction,
    - m0, m1 are the refractive indices of each solute,
    - v0, v1 are the volumes of each solute.

Then the resulting refractive index is:
- n_eff = (2 × r_eff + 1) / (1 - r_eff).

**Parameters**

- `m_zero` (`Union[float, complex]`) — Refractive index of solute 0 (float or complex).
- `m_one` (`Union[float, complex]`) — Refractive index of solute 1 (float or complex).
- `volume_zero` (`float`) — Volume of solute 0.
- `volume_one` (`float`) — Volume of solute 1.

**Returns**

- (`Union[float, complex]`) — Effective refractive index of the mixture (float or complex).

**Examples**

``` py title="Example"
import particula as par
n_mix = par.get_effective_refractive_index(1.33, 1.50, 2.0, 1.0)
print(n_mix)
# Output: ~1.382
```

> **References**
>
> - Y. Liu & P. H. Daum, "Relationship of refractive index to mass
>   density and self-consistency mixing rules for multicomponent
>   mixtures like ambient aerosols," Journal of Aerosol Science,
>   vol. 39(11), pp. 974–986, 2008.
>   DOI: 10.1016/j.jaerosci.2008.06.006
> - Wikipedia contributors, "Refractive index," Wikipedia.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/refractive_index_mixing.py#L10-L61)
