# Reference observations for the SiMBA reproductions

The [published SiMBA cases](https://doi.org/10.1039/D5SC01473E) use
synthetic concentration measurements. Compare the code-generated reference
profiles to the archived CSVs before comparing network rankings. Run from the
Particula checkout with NumPy and SciPy installed.

```bash
curl -fLsS https://codeload.github.com/OptiMaL-PSE-Lab/auto_react_mech_construct/tar.gz/8f0d16357193d79ea73f517540571fc6997f0941 -o /tmp/simba-source.tar.gz
python studies/simba/reference_inputs.py /tmp/simba-source.tar.gz
```

The [audit](reference_inputs.json) records each input checksum and the largest
absolute discrepancy across 30 observation times and all measured species.
All 15 profiles reproduce within 1e-7 in the original concentration units.

Two details affect comparisons with the published cases.

* The hypothetical generator writes a dimerization flux proportional to A
  squared but consumes only one A per B produced. Applying the stated
  stoichiometry of two A per B changes a measured-species prediction by up
  to 2.45 concentration units at the original rates. Preserve that difference
  when comparing a native mass-balanced model with the archived profiles.
* The HMF dataset associated with the paper's empirical rate law is
  `exp_data_fruc_HMF2`. Experiment four's archived trajectory reproduces
  from the initial concentrations [6, 0, 0]. The generating script on the
  pinned revision specifies [4, 0, 1]. The other four experiments reproduce
  using their scripted initial conditions.

[Hypothetical source](https://github.com/OptiMaL-PSE-Lab/auto_react_mech_construct/blob/8f0d16357193d79ea73f517540571fc6997f0941/hypothetical_network.py)
· [Fructose source](https://github.com/OptiMaL-PSE-Lab/auto_react_mech_construct/blob/8f0d16357193d79ea73f517540571fc6997f0941/fructose_to_HMF2.py)

## Ranking the published candidates

Run the following after downloading the pinned tarball above. The script fits
each network listed in Tables 1–3 jointly across five experiments using the
original measurement mask. It minimizes squared concentration errors, then
computes the authors' AIC with a separate residual variance per measured
species. The rates use each paper case's original concentration and time
units. Two bounded least-squares starts per network use the paper's 0–10
interval, independent LSODA integration, and the archived initial conditions
verified above. The upstream repository's default fitting script has a
different upper bound of 3. Each table reports a winner per complexity.
This experiment refits those published finalists.

```bash
OPENBLAS_NUM_THREADS=1 python studies/simba/published_rankings.py /tmp/simba-source.tar.gz /tmp/published_rankings.json
```

| Benchmark | AIC by ascending published iteration | Selected iteration | Paper selection |
|---|---|---:|---:|
| Hypothetical, signed unit changes with mass-action orders preserved | 1110.34, 60.91, −406.84, 81.02 | 3 | 3 |
| Hypothetical, molecular stoichiometry as written | 933.39, 935.38, 937.38, 939.39 | 1 | 3 |
| Aldol | 1759.23, 848.42, −580.36, −578.58 | 3 | 3 |
| Fructose to HMF | −171.45, −165.23, −167.77 | 1 | 2 |

The published hypothetical winner is recoverable with the source generator's
dimer convention. The physically balanced model ranks the shortest candidate
first. The aldol fit selects the paper's three-step pathway; the four-step fit
is within two AIC units. The fructose refit selects sequential dehydration
starting at the first step. Its first intermediate forms water immediately.
The paper's winning alternative delays water formation until the second step.
Its archived source generates measurements directly from a one-step empirical
rate law; the intermediate structures have no measured time series. The
difference in rankings remains a reproducibility result. Numerical optimizer,
kinetic bounds, initial conditions and the published tables' internal
differences limit attribution of the exact score discrepancies.

[Rate fits and per-species RMSE](published_rankings.json) ·
[Predicted concentrations](published_rankings.json.gz), one array per fit
in the same order, with axes time, experiment, measured species.

## Coupled chamber trajectory

The synthetic box starts with three fixed lognormal seed cohorts. A gas
precursor converts to a semivolatile vapor, the vapor exchanges reversibly
with each seed, and its condensed material ages. Deterministic deposition
moves each particle constituent to its own wall inventory. Time-dependent
observations include two speciated gas mass concentrations, suspended organic
mass and three number channels including the two detector tails.

Run the qualification on a CUDA-equipped host from the checkout root.

```bash
PYTHONPATH=. python studies/simba/qualify_trajectory.py /tmp/trajectory_qualification.json
```

An independent SciPy Radau implementation evolves moles per physical
particle and surviving cohort weights. Warp evolves suspended and deposited
molar concentrations per air volume. The two initial vapor conditions give
initial evaporation and condensation, respectively. At 0.025 s, the maximum
difference across the six channels and both experiments was less than
0.002% of each channel's peak. The saved report compares all nine input
derivatives against the Radau reference at three finite-difference
increments, including a one-sided difference at zero initial vapor. It
includes hard-bin predictions alongside the finite-width response. Gas
chemistry, particle aging, gas exchange, volume, seed shape and deposition
all participate in the differentiated trajectory. The wall coefficient is
defined for an inverse-diameter synthetic deposition law at 100 nm.

[CPU and CUDA qualification](trajectory_qualification.json)

The [bounded enumeration record](enumeration.json) checks whether the native
generator can propose each of the eleven tabulated finalists. Starting from
element or conserved-pool vectors, it generates unimolecular and bimolecular
steps, imposes the paper's terminal-product and net-balance restrictions,
and searches up to each reported complexity. All eleven occurred within the
stated budgets. The largest search examined 1,385,979 combinations. The
hidden species budgets are supplied hypotheses. The independent refits above
compare the published finalists, while enumeration also admits additional
networks whose fit was outside this finalist comparison.

```bash
PYTHONPATH=. python studies/simba/enumeration.py /tmp/enumeration.json
PYTHONPATH=. python studies/simba/native_parity.py /tmp/native_parity.json
PYTHONPATH=. python studies/simba/heldout_recovery.py cuda:0 /tmp/recovery.json
```

### Sensitivity of the fructose comparison

The [authors' archived execution log](https://github.com/OptiMaL-PSE-Lab/auto_react_mech_construct/blob/4514fa0/output_fruc_to_hmf2.log)
contains the paper's three AIC scores to rounding. At that same revision
the fitting script declares [4, 0, 1] for experiment four and bounds rates
between 0.0001 and 3. The archived CSV reproduces from [6, 0, 0]. Using the
printed initial condition to fit that CSV raises the best AIC from −171.45
to 914.83. Regenerating experiment four from the printed condition with the
original random seed gives −172.10, −166.02 and −168.76 across iterations.
Reducing the upper rate bound from 10 to 3 on the archived observations also
leaves iteration one first. These checks identify a material input discrepancy
and rule out those individual corrections as explanations of the paper's
ranking. The archived log does not contain the per-iteration parameter vectors
or predictions required to replay the reported scores at fixed parameters.

```bash
python studies/simba/fructose_sensitivity.py /tmp/simba-source.tar.gz /tmp/hmf.json
```

[Matched sensitivity fits](fructose_sensitivity.json) ·
[Contemporaneous estimator](https://github.com/OptiMaL-PSE-Lab/auto_react_mech_construct/blob/4514fa0/parameter_estimation.py)
