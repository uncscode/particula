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
x=100. Larger sizes have independent pointwise and solid-angle
qualification below.

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

## Independent large-sphere angular qualification

`large_angular_test.py` compares NumPy, Warp CPU and native CUDA amplitudes
against independent external spherical Bessel functions and Jacobi-polynomial
angular functions at x=100, 1000, 3000 and 10000. Materials are
m=1.33+1e-8i, 1.5+0.1i and 0.2+3i. Nine angles include both endpoints,
1e-6/1e-4/1e-3 radian forward angles and near-backward scattering. The oracle
includes ten additional series terms to check truncation independently.

The internal logarithmic derivative is computed from ratios of
[SciPy's scaled AMOS Bessel functions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jve.html).
Angular functions use the Legendre derivative identity
P_n'(mu)=(n+1)P_(n-1)^(1,1)(mu)/2, evaluated by
[SciPy's Jacobi polynomial function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_jacobi.html).
These avoid the production continued fraction and angular recurrence.

At m=0.2+3i and x=10000, even scaled AMOS functions underflow at large orders.
For that case, `high_precision_angular.json` stores an independent reference
using upward internal Riccati-Bessel recurrence at 2000 decimal digits,
repeated at 2200 digits. Every logarithmic derivative agrees exactly after
rounding to complex128. External radial and angular functions remain SciPy
double precision; this is not a claim of 2000-digit final amplitudes.
The fixture records versions, precision check, term count and generator hash.
Neither mpmath nor an external Mie solver is a project dependency.

Acceptance is maximum amplitude error divided by the case's peak amplitude
<=1e-8. A second check constrains every amplitude at angles >=0.01 radians
with rtol=1e-6 and atol=1e-8, so weak side/back scattering cannot be hidden
by a large forward peak. All 36 backend/material/size combinations pass on
NumPy, Warp CPU and the RTX 5060. This establishes the listed sample cases,
not an exhaustive proof for every refractive index.

To regenerate the exceptional fixture, install mpmath==1.3.0 in a disposable
external environment alongside the NumPy/SciPy versions recorded in the JSON.
Save and run this generator with its output JSON path as the first argument:

```python
"""Generate one independent absorbing-sphere angular reference.

Run with mpmath==1.3.0 in a disposable environment; not a package dependency.
Internal Riccati-Bessel functions use upward recurrence at 2000/2200 decimal
places, distinct from the production double continued fraction/downward ratio.
The second precision run verifies that cancellation has not exhausted precision.
"""
import hashlib
import json
from pathlib import Path
import sys

import mpmath as mp
import numpy as np
import scipy
from scipy.special import eval_jacobi, eval_legendre, spherical_jn, spherical_yn

x = 10000.0
m = 0.2 + 3j
n = np.arange(1, int(np.ceil(x + 4.05 * np.cbrt(x) + 2)) + 11)
theta = np.array([0, 1e-6, 1e-4, 1e-3, .01, .5, np.pi/2, np.pi-1e-4, np.pi])


def derivatives(precision):
    with mp.workdps(precision):
        z = mp.mpc(2000, 30000)
        previous = mp.sin(z)
        current = mp.sin(z)/z - mp.cos(z)
        result = []
        for order in n:
            order = int(order)
            result.append(complex(previous/current - order/z))
            previous, current = current, (2*order+1)/z*current - previous
        return np.array(result)


d = derivatives(2000)
verified = derivatives(2200)
np.testing.assert_allclose(d, verified, rtol=0, atol=0)
j, jp = spherical_jn(n, x), spherical_jn(n, x, True)
y, yp = spherical_yn(n, x), spherical_yn(n, x, True)
psi, psip = x*j, j+x*jp
xi, xip = x*(j+1j*y), j+1j*y+x*(jp+1j*yp)
a = (m*psip-d*psi)/(m*xip-d*xi)
b = (psip-m*d*psi)/(xip-m*d*xi)
order, mu = n[:, None], np.cos(theta)[None, :]
pi = (order+1)/2*eval_jacobi(order-1, 1, 1, mu)
tau = order*(order+1)*eval_legendre(order, mu)-mu*pi
factor = (2*order+1)/(order*(order+1))
s1 = np.sum(factor*(a[:, None]*pi+b[:, None]*tau), axis=0)
s2 = np.sum(factor*(a[:, None]*tau+b[:, None]*pi), axis=0)
amplitudes = np.stack((s1, s2), axis=-1)
result = {
    'description': 'Independent large absorbing sphere angular oracle',
    'index': [m.real, m.imag], 'size_parameter': x,
    'angles_radians': theta.tolist(),
    'amplitudes': np.stack((amplitudes.real, amplitudes.imag), -1).tolist(),
    'internal_decimal_precision': [2000, 2200],
    'precision_check': 'Every derivative identical after rounding to complex128',
    'series_terms': int(n[-1]),
    'mpmath': mp.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2)+'\n')
print('Saved', sys.argv[1], flush=True)
```

The high-size solid-angle check uses composite 128-point Gauss integration
in scattering angle, including the sin(theta) Jacobian, at x=1000 and 10000.
Two panel densities (ceil(x/40) and ceil(x/20)) independently recover Qsca
and g at rtol=1e-8. A single large Legendre rule in cos(theta) produced a
1.6e-6 relative scattering discrepancy at x=10000; composite integration
resolves that quadrature limitation without relaxing acceptance thresholds.
