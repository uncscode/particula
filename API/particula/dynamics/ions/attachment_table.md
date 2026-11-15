# particula.dynamics.ions.attachment_table

*module*

Condition the published LF2013 polynomials for portable fp64 evaluation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/attachment_table.py#L1-L1)

## particula.dynamics.ions.attachment_table.centered_attachment_tables

*function* · *cached*

```python
def centered_attachment_tables()
```

Return immutable coefficients in a centered logarithmic-radius basis.

Shape is (2, 201, 28): negative/positive ion polarity, aerosol charges
-100..100, and 24 polynomial coefficients followed by log-radius center,
log-radius half-width, minimum radius and maximum radius [m].

The published polynomial is in log10(radius [m]). Direct fp64 evaluation
suffers cancellation. Decimal arithmetic at 70 digits changes its basis
to x=(log10(radius)-center)/half_width, with x in [-1, 1]. This is an
algebraic transformation of the published decimal coefficients, not a new
fit, interpolation or extrapolation. Arrays can be uploaded once to Warp.
Source: https://doi.org/10.1080/02786826.2013.783684, tables S1 and S2.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/attachment_table.py#L10-L52)
