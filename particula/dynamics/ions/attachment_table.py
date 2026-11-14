"""Condition the published LF2013 polynomials for portable fp64 evaluation."""

from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def centered_attachment_tables():
    """Return immutable coefficients in a centered logarithmic-radius basis.

    Shape is (2, 201, 28): negative/positive ion polarity, aerosol charges
    -100..100, and 24 polynomial coefficients followed by log-radius center,
    log-radius half-width, minimum radius and maximum radius [m].

    The published polynomial is in log10(radius [m]). Direct fp64 evaluation
    suffers cancellation. Decimal arithmetic at 70 digits changes its basis
    to x=(log10(radius)-center)/half_width, with x in [-1, 1]. This is an
    algebraic transformation of the published decimal coefficients, not a new
    fit, interpolation or extrapolation. Arrays can be uploaded once to Warp.
    Source: https://doi.org/10.1080/02786826.2013.783684, tables S1 and S2.
    """
    directory = (
        Path(__file__).resolve().parents[2] / "util" / "lf2013_coagulation"
    )
    result = np.empty((2, 201, 28), dtype=np.float64)
    with localcontext() as context:
        context.prec = 70
        for polarity, name in enumerate(("S1.txt", "S2.txt")):
            rows = (directory / name).read_text().splitlines()[1:]
            for index, row in enumerate(rows):
                values = [Decimal(value) for value in row.split()]
                coefficients = values[1:-4]
                low, high = values[-2:]
                center = (low.log10() + high.log10()) / 2
                scale = (high.log10() - low.log10()) / 2
                transformed = [coefficients[-1]]
                for coefficient in coefficients[-2::-1]:
                    shifted = [Decimal(0)] * (len(transformed) + 1)
                    for power, value in enumerate(transformed):
                        shifted[power] += value * center
                        shifted[power + 1] += value * scale
                    shifted[0] += coefficient
                    transformed = shifted
                result[polarity, index] = [
                    float(value)
                    for value in (*transformed, center, scale, low, high)
                ]
    result.flags.writeable = False
    return result
