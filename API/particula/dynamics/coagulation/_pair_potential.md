# particula.dynamics.coagulation._pair_potential

*module*

Nonretarded dispersion and dielectric response of two compact spheres.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/_pair_potential.py#L1-L1)

## particula.dynamics.coagulation._pair_potential._multipole_orders

*function* · *cached*

```python
def _multipole_orders(order)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/_pair_potential.py#L26-L30)

## particula.dynamics.coagulation._pair_potential.dielectric_response

*function*

```python
def dielectric_response(x, fraction, relative_permittivity, order)
```

Return induced energy coefficients for q1², q1*q2 and q2².

Lengths are normalized by contact distance R=a+b. Add x*q1*q2
for the bare monopoles; multiply the result by 1/(4*pi*eps0*R)
with charges in C to obtain energy in J. The exterior is vacuum.
Free surface charge is uniform on each sphere; bound charge responds
self-consistently. Both polar axes point toward the partner centre.

Each degree l reflects a regular incident potential with coefficient
-l*(eps-1)/(l*(eps+1)+1). Translation of an axial multipole k to
the other centre supplies binomial(l+k,l). Energy is half the sum
of free charges times the induced potentials at their centres.

Reference: Bichoutskaia et al. (2010), doi:10.1063/1.3457157.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/_pair_potential.py#L33-L76)

## particula.dynamics.coagulation._pair_potential.hamaker_energy

*function*

```python
def hamaker_energy(x, fraction)
```

Return U/A for centre-distance ratio x=(a+b)/s, 0 <= x < 1.

The symmetric radius fraction a/(a+b) may refer to either sphere.
Hamaker (1937), doi:10.1016/S0031-8914(37)80203-7.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/_pair_potential.py#L10-L23)
