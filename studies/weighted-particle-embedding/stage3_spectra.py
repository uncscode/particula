"""Reproducible dry-radius distributions for the operator stress tests."""


def dry_radius(u, family, where):
    """Return dry radius in metres for each deterministic particle quantile."""
    if family == 901001:
        nm = 65 + 20 * u
    elif family == 901002:
        nm = 10 + 25 * u**1.5
    elif family == 901003:
        nm = where(
            u < 0.75, 11 + 17 * (u / 0.75) ** 1.5, 70 + 75 * ((u - 0.75) / 0.25)
        )
    elif family == 901004:
        nm = where(
            u < 0.995,
            20 + 45 * (u / 0.995) ** 1.8,
            220 + 280 * ((u - 0.995) / 0.005),
        )
    elif family == 901005:
        nm = 17 + 208 * u**2.2
    else:
        radius = 20e-9 + 95e-9 * u**1.8
        return (
            where(u > 0.98, radius * 2.3, radius) if family % 3 == 2 else radius
        )
    return nm * 1e-9


def radius_domain(mass, density):
    """Enclose every initial particle on the reference volume grid."""
    import numpy as np

    conversion = 3 / (4 * np.pi * density)
    rmin = (float(mass.min()) * conversion) ** (1 / 3)
    rmax = (float(mass.max()) * conversion) ** (1 / 3)
    return min(15e-9, 0.85 * rmin), max(1.8e-6, 4 * rmax)
