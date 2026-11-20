# Independent process reference equations

The original scientific baseline used revision `894f60350` as its branch base.
Measurements below use Python 3.12, float64 CPU calculations; CUDA checks use
an NVIDIA RTX 5060 with driver 615.78.08. They verify implementation and
numerical limits, not agreement with independent laboratory measurements.

## Independent reference definitions

Exact dilution uses the solution `c(t)=c(0) exp(-Qt/V)` in
`particula/dynamics/tests/dilution_test.py`, including distinct gas and particle
updates and preserved per-particle properties.

Condensation uses a monodisperse 1 µm sphere of density 1000 kg/m³, vapor
1e-5 kg/m³, number density 1e9 m⁻³, temperature 298.15 K, pressure 101325 Pa,
diffusivity 2e-5 m²/s and prescribed viscosity 1.8e-5 Pa s. Zero surface vapor
pressure isolates transport. Integrating `dt/dr` independently gives the time
to radius 1.2 µm. In the closed box, gas depletion is determined by species
conservation. With common dilution hazard k, replace elapsed time in the
undiluted solution by `(1-exp(-kt))/k`. Production rate helpers are not used
by this quadrature oracle.

| Case | Duration (s) | Radius error, 16 steps | 32 steps | 64 steps |
| --- | ---: | ---: | ---: | ---: |
| Fixed reservoir | 1.16232227 | 0.00110597 | 0.000553773 | 0.000277082 |
| Closed box | 1.37856539 | 0.000808400 | 0.000405070 | 0.000202747 |
| Dilution before growth | 1.53808988 | 0.000276841 | 0.000134724 | 0.000066437 |
| Dilution after growth | 1.53808988 | 0.00186326 | 0.000929738 | 0.000464395 |

Errors are absolute errors in r/r0. The predeclared finest-grid bound is 0.002;
each halving must reduce error by a factor below 0.6. Closed/diluted material
budgets use rtol 2e-14. Different process orders converge to the same solution;
these cases do not establish accuracy in stiff equilibrium regimes.

For constant-kernel coagulation, initial mass density is exponential,
`n(m,0)=N0/m0 exp(-m/m0)`. The exact solution is
`n(m,t)=N0/[m0(1+tau)^2] exp[-m/(m0(1+tau))]`, with `tau=K*N0*t/2`.
Transform to radius density with the mass/radius Jacobian. Inputs are
N0=1e9 m⁻³, K=1e-15 m³/s, characteristic radius 100 nm, radius domain
1 nm–1 µm, density 1000 kg/m³, duration 2e5 s and 100 time steps.

| Radius nodes | Relative number error | Relative mass error | Integrated spectrum error/N0 |
| ---: | ---: | ---: | ---: |
| 128 | 0.000318886 | 0.000089594 | 0.000163057 |
| 256 | 0.000013571 | 0.000021436 | 0.000124353 |
| 512 | 0.000062065 | 0.000004905 | 0.000117116 |

All finest-grid errors must be below 0.002. The number error is not monotonic:
spatial refinement alone cannot remove time error. Separate existing PMF/PDF
refinement tests check uniform-composition mass quadrature and characterize
persistent variable-composition species defects without relaxing tolerances.

Removal sampling separately varies 128/512 represented slots and 16/64 seeds
at fixed physical population. With hazard 0.7 s⁻¹, the independent survival
probability is exp(-0.7); variance of the ensemble-mean fraction is
p(1-p)/(slots*samples). Five-standard-error mean bounds, predeclared variance
bounds and per-material removed-plus-retained budgets pass all four cases.
This qualifies removal sampling, not arbitrary coagulation sampling workloads.

The existing resolved coagulation pair test uses 4000 trials for each of
interpolated, radius-callable and index-callable kernels. The independent
probability is K*dt/V=0.2 and the mean must lie within five binomial standard
errors. This resolves the earlier same-bin self-pair concern for that regime.
The sampler caps trials at ten times the larger bin population and allows each
donor only one collision per step. Its endpoint estimate is not a proven upper
bound for arbitrary non-monotone custom kernels. Large timesteps, capped
probabilities above one and such unqualified kernels are excluded from this
baseline's kinetic accuracy claim; reduce timestep and establish ensemble
convergence for a chosen workload. These limitations do not invalidate the
separate resolved species conservation tests.
