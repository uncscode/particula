# Native gas chemistry qualification

A single native mechanism drives NumPy and resident CUDA integration for
closed, irreversible gas reactions at prescribed temperature. Three synthetic
experiments isolate different errors: assigning the wrong storage column,
confusing mass with molar concentration, and losing precision in the stiff
solver. The numerical implementation was qualified at revision
`4825b85fbd` with Python 3.12.12, NumPy 2.3.4, SciPy 1.16.2 and Warp
1.11.1 on an NVIDIA GeForce RTX 5060. [Mechanism guide](../../../guide/processes/gas-chemistry.md)

## First-order conversion and external mass

For closed A → B with equal molar masses and $k=0.3$ s⁻¹, A follows
$A_0\exp(-kt)$ and B gains the missing A mass. The regression checks both
species, inert gas, and extensive mass in boxes of 1 and 2 m³. In the
worked example an independent pulse adds A at 4 s and ventilation dilutes
all gases at 0.05 s⁻¹. The pre-pulse reading is at 4 s. The first post-pulse
reading is at 4.5 s. The full 41-point trajectories used 0.25 s process
intervals and were checked against analytical concentrations with
`rtol=2e-9, atol=1e-20 kg/m³` and A+B inventories with
`rtol=1e-12, atol=1e-22 kg`.

| Backend | Maximum absolute A or B error (kg/m³) | Added mass in boxes 1 and 2 (kg) |
| --- | ---: | --- |
| NumPy | 1.7272 × 10⁻²⁰ | 3 × 10⁻⁹, 2 × 10⁻⁹ |
| CUDA | 1.7261 × 10⁻²⁰ | 3 × 10⁻⁹, 2 × 10⁻⁹ |

Restarted and uninterrupted histories and source ledgers matched bit for bit
within each backend. Each run has a prescribed intervention at 4 s. The
experiment snapshots the pre-pulse state, applies the external dose on the
host, and begins the next segment at 4 s. The restart variants resume the
pre-pulse checkpoint and another checkpoint at 4.5 s. Rate evaluations,
trial decisions and linear solves stay on the selected backend.
[Executed example and figure](../../../examples/processes/customization/native_gas_chemistry.py)

## Bimolecular reaction and binding

A + B → C uses molar masses 0.01, 0.02 and 0.03 kg/mol. Initially A and B have
equal *molar* concentrations (1 and 2 mol/m³ in the respective boxes). At
298 and 318 K the rate coefficient is $2(T/298)\exp[-1000/(RT)]$ m³/mol/s.
The two reactants follow $c_0/(1+kc_0t)$, and the product gains one mole for
each mole of A lost. RK23 and ROS3 match the analytical concentrations with
relative tolerance 10⁻⁷. The inert gas retains its exact value. These tests
run on NumPy, Warp CPU and CUDA. Loading JSON and YAML into permuted gas
layouts and rebinding produces exactly matching physical concentrations.
Stale layouts, incompatible masses, duplicate bindings and failed
elemental balances have dedicated rejection tests.

## Stiffness and full-precision coefficients

For the Robertson test use normalized initial state (1, 0, 0), concentration
scale 1 mol/m³ and time scale 1 s. Declare equal 0.1 kg/mol molar masses
for the three synthetic species. The test integrates from zero independently
to 10⁻⁴, 10⁻², 1 and 100 s. The reference uses the Stage 2A right-hand side
with independent Radau and BDF integrations; both agree within
`rtol=1e-8, atol=1e-14`. The production solution must agree with each
reference at tight settings within `rtol=2e-6, atol=1e-12`, with a separate
comparison for every species. [Robertson benchmark](https://archimede.uniba.it/~testset/problems/rober.php)
· [KPP ROS3 method](https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html)

| Backend | Relative tolerance | Maximum absolute errors in A, B, C (mol/m³) | Accepted / rejected | Factorizations / solves |
| --- | ---: | --- | ---: | ---: |
| NumPy | 10⁻⁵ | 5.84 × 10⁻⁹, 1.17 × 10⁻¹¹, 5.84 × 10⁻⁹ | 1064 / 11 | 1075 / 9675 |
| CUDA | 10⁻⁵ | 5.84 × 10⁻⁹, 1.17 × 10⁻¹¹, 5.84 × 10⁻⁹ | 1064 / 11 | 1075 / 9675 |
| NumPy | 10⁻⁸ | 2.40 × 10⁻¹³, 8.04 × 10⁻¹⁵, 2.43 × 10⁻¹³ | 13582 / 11 | 13593 / 122337 |
| CUDA | 10⁻⁸ | 2.35 × 10⁻¹³, 8.04 × 10⁻¹⁵, 2.47 × 10⁻¹³ | 13582 / 11 | 13593 / 122337 |

The work counts sum the four independent integrations for the first box.
Absolute tolerance is 10⁻¹⁴ mol/m³. Every accepted endpoint is nonnegative
and normalized total concentration is conserved to relative tolerance 10⁻¹²
in this equal-mass case. Tightening the tolerance reduced error in all three
species. Using full-precision device tableau constants matters: intermediate
rounding of those constants to float32 produced an error floor near
8 × 10⁻⁹ mol/m³ for A and C during development. A dedicated device regression
checks the refined solution. This study establishes no GPU speedup or
large-network scaling claim.

## Reproduce

From the checked-out repository in a CUDA-capable environment:

```bash
python -m pytest particula/dynamics/chemistry/tests examples_tests/native_gas_chemistry_test.py -q -Werror --no-cov
python -m scripts.assess_gas_chemistry
python scripts/docs_examples.py refresh native-gas-chemistry
```

The assessment prints the component errors and work counts. CUDA
qualification requires an actual device; check the device selection and
pytest skip summary. The pulse example saves its figure and numerical data.
