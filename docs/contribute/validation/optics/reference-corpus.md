# PyMieScatt reference corpus

Development reference only. No external Mie package is required by Particula
or its tests. These fixtures are raw reference observations, not a claim that
every PyMieScatt branch is a correct physical oracle.

The pinned source is MIT licensed, copyright (c) 2017 Benjamin Sumlin:
https://github.com/bsumlin/PyMieScatt/blob/1e79410c21fc9ef36088dd06fee83a31aa42b0dc/LICENSE.md

## Regeneration

In a disposable environment with the versions recorded in the JSON manifest,
download only `PyMieScatt/Mie.py` at the recorded commit. Save the following
generator as a temporary Python file and run it with two arguments: the local
reference source and the output JSON path. Do not install PyMieScatt in the
project environment. The manifest records both source and generator hashes.

```python
"""Generate raw reference data; run outside the Particula environment/package."""
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np

source = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('reference_mie', source)
mie = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mie)
angles = np.array([0, 1e-6, 1e-4, 1e-2, 0.1, np.pi/6, np.pi/3,
                   np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi-1e-4, np.pi])
indices = [1+0j, 1.33+0j, 1.33+1e-8j, 1.5+0.01j,
           1.77+0.63j, 0.8+0.1j, 0.2+3j, 2+5j]
sizes = [1e-6, 1e-4, .01, .049, .05, .051, .1, .49, .5, .51,
         1., 2., 10., 50., 100., 1000., 3000., 10000.]
def pair(z):
    return [float(np.real(z)), float(np.imag(z))]
cases = []
for m in indices:
    for x in sizes:
        # Nonzero contrast is required by PyMieScatt's g and ratio divisions.
        # Index-matched cases are independent zero-limit tests, not raw goldens.
        if m == 1:
            continue
        diameter = 500.0*x/np.pi
        q = mie.MieQ(m, 500., diameter, nMedium=1.)
        a, b = mie.Mie_ab(m, x)
        amplitudes = [mie.MieS1S2(m, x, float(np.cos(t))) for t in angles]
        matrices = [mie.MatrixElements(m, 500., diameter, float(np.cos(t)),
                                       nMedium=1.) for t in angles]
        cases.append(dict(index=pair(m), size_parameter=x,
                          wavelength_nm=500., diameter_nm=diameter,
                          efficiencies=[float(v) for v in q],
                          first_coefficients_a=[pair(v) for v in a[:3]],
                          first_coefficients_b=[pair(v) for v in b[:3]],
                          amplitudes=[[pair(s1),pair(s2)] for s1,s2 in amplitudes],
                          matrix_elements=[[pair(v) for v in row] for row in matrices],
                          efficiency_regime='rayleigh' if np.pi*diameter/500 <= .05 else 'mie',
                          angular_regime='low_frequency' if x < .5 else 'mie'))
result = dict(schema_version=1, source=dict(repository='https://github.com/bsumlin/PyMieScatt',
    revision='1e79410c21fc9ef36088dd06fee83a31aa42b0dc', path='PyMieScatt/Mie.py',
    sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    environment=dict(python=platform.python_version(), numpy=importlib.metadata.version('numpy'),
                     scipy=importlib.metadata.version('scipy')),
    conventions=dict(index='n+ik', medium_index=1., amplitude='raw PyMieScatt MieS1S2',
        matrix='raw MatrixElements: S11,S12,S33,S34',
        efficiencies=['Qext','Qsca','Qabs','g','Qpr','Qback','Qratio'],
        complex_encoding='[real, imaginary]'), angles_radians=angles.tolist(),cases=cases)
out=Path(sys.argv[2])
out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(len(cases), 'cases;', out.stat().st_size, 'bytes;', 'sha256', hashlib.sha256(out.read_bytes()).hexdigest())
```

## Interpretation

- Compare exact Mie regimes separately from small-sphere approximations.
- Below x=0.5, the angular routines use low-frequency coefficients. In an
  exploratory case (m=1.5+0.05i, x=0.49, 90 degrees), their amplitude differs
  from the full coefficient sum by about 0.5 percent.
- Efficiencies and angles have distinct approximation switches; the manifest
  labels their actual branches independently.
- Use nMedium=1 explicitly. Non-unit-medium correctness requires independent
  normalization tests because the pinned high-level routines are inconsistent.
- Cross sections must be independently computed from area times efficiency;
  the pinned small-sphere MieQ path drops asCrossSection.
- Index-matched particles require independent exact-zero tests; PyMieScatt
  divides by scattering in some derived outputs.
- Raw complex Mueller entries are retained, including signed zero, without
  changing sign, normalization, or units.

Population fixtures and independent published/analytic benchmarks are still
pending. This corpus does not by itself qualify a Particula implementation.

## Large-sphere reference discrepancies

The first independent implementation comparison found nine cases outside the
1e-6 relative tolerance: m=1.33 and m=1.33+1e-8i at x=100, 1000, 3000, 10000,
and m=1.5+0.01i at x=100. These exact cases retain their raw fixture values,
but physical acceptance uses an independent direct spherical-Bessel evaluation
with SciPy, rather than the production continued-fraction algorithm. Tests
also assert that the documented raw-reference disagreement remains observable.

For m=1.33, independent/production extinction agrees to about 1e-14:

| x | Independent extinction | Raw PyMieScatt extinction |
| --- | --- | --- |
| 100 | 2.1010895537298273 | 2.101037518280405 |
| 1000 | 2.0165783128478854 | 2.0162568524087 |
| 3000 | 2.008372431870777 | 2.007587213279409 |
| 10000 | 2.00411482223858 | 2.0043977121561216 |

The precise upstream failure mechanism has not yet been isolated. No fixture
values or acceptance tolerances were changed to hide these discrepancies.

## Angular comparison convention

Raw angular comparisons use maximum absolute error normalized by the case's
peak amplitude or Mueller element, with a 1e-6 limit. This implements the
planned scale-aware angular metric: pointwise relative errors are unsuitable
at amplitude nodes and near forward-scattering polarization zeros. This does
not replace independent polarization or energy checks.

At x=50 the raw reference differs slightly from full Bessel coefficients,
including pointwise amplitude differences around 2.4e-6 relative for m=1.33.
All seven x=50 material cases additionally compare against independently
computed SciPy Bessel coefficients and Legendre-polynomial derivatives at
rtol=1e-9, atol=1e-10. Mueller sign conventions are independently checked by
propagating complex Jones electric fields and constructing Stokes vectors.
Angular integration tests recover total scattering and asymmetry through
x=100. Angular qualification at larger size parameters remains pending.

## Population reference cases

`pymiescatt_populations.json` records four raw `Mie_SD` calls from the same
source and environment as the sphere corpus. Regenerate with the source
loader above, m in [1.33+0j, 1.77+0.63j], wavelength=500 nm,
dp=np.array([1.,2.,10.])*500/np.pi nm, ndp=np.array([10.,20.,30.]),
nMedium=1., and each SMPS flag True/False. The JSON records argument values,
entrypoint, output order, source hash and revision. No reference code is
imported by tests.

For SMPS=True the number entries are cm⁻³ bin densities. For SMPS=False they
are a diameter density per cm³ per nm; convert to the native radius PDF by
multiplying by 2e15. Raw bulk optical coefficients are Mm⁻¹ and convert to
m⁻¹ by multiplying by 1e-6. G is dimensionless; Bratio is retained for provenance
but is not a Particula output. Tests compare extinction, scattering,
absorption, backscatter, radiation pressure and G independently.
