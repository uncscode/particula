# Reproduce the condensation comparison

Run from the root of the weighted-particle research checkout. Use Python 3.12 with the local Particula checkout, PyTorch 2.10.0+cu128, Triton 3.6.0, NumPy and SciPy. The measured device was an RTX 5060. The commands below use the existing CUDA interpreter and write to a fresh output directory.

```bash
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
s=studies/weighted-particle-embedding
out=.artifacts/condensation-replay
python "$s/condensation_options.py" "$out"
python "$s/condensation_audit.py" "$out"
```

The first command fits the two positive coefficient models on 24,000 native transition factors with fixed features and reweighted least squares. Training and holdout draws use different seeds. The third fit is one scalar multiplying a difference of physical steps. Its least-squares labels are gas concentrations from 32 vapor substeps per half interval, using two 2,048-host training boxes. Each vapor is scaled by its initial gas concentration. The validation forecasts use independent event seeds but the same synthetic forcing family. The audit recreates both coefficient heads from the fixed training recipe and stores their layer arrays and the step weight in `weights.npz`.

The boxes, particle and species ledgers, 32 vapor lanes, pressure, transport law, collision kernel, birth law, random streams and sampling cadence follow the main 32-vapor experiment. One million initial hosts at 10⁶ cm⁻³ occupy 1 cm³. Each 10 s rollout uses 80 intervals with a vapor half step on either side of physical formation and sampled physical collisions. Approximation errors at equal step size use the original physical implementation. Refined-clock comparisons use the exact two-pass physical implementation with 16 substeps per half interval. The one-seed audit extends the reference to 64 substeps. Relative errors span all observations, individual phases and totals, with the study's 10⁻¹¹ kg/m³ floor. Particle count error is scored relative to the paired reference. The code checks the particle number ledger and each species inventory at all observations.

The independent stage profile times successive *vapor-only* half steps with device events. It excludes collisions, births and diagnostic observations. Whole-forecast timings include these processes, diagnostics and process synchronization. Forecast timing begins after initialization, which excludes fitting, lookup construction and the warm-up. Each method has two warm-up intervals before the timed seed blocks. The audit separately measures table construction and four-vapor inference at the initial population. All coefficients and mass budgets use double precision.

`profile.csv` holds per-half-step milliseconds. The three `trials-*.csv` files hold the paired forecast times, number and phase errors, event counts and inventory checks. `fits.csv` describes the two coefficient fits, `coefficient-probe.csv` times direct and tabulated evaluation of the same fitted head, and `clock-audit.csv` gives the one-seed 64-substep check. `settings.json` and `weights.npz` retain the fit and run parameters. The downloadable figure bundle includes these supplementary results under `condensation/`.

The two-pass kernel is specific to this 32-vapor mass-fraction equilibrium law, 33-component particle layout and fixed material properties. Table methods check the radius domain from 0.25 to 5,000 nm on lookup. Temporal coefficient reuse occurs only across adjacent vapor half steps that have no intervening birth or merger. For corrected steps, gas and all particle masses of a species receive the same extrapolation weight. If any proposed mass would be negative, that entire species retains its conservative two-substep prediction.
