# Reproduce the weighted-particle forecasting study

The [study overview](../../docs/studies/explorations/weighted-particle-embedding/index.md) compares physical and learned aerosol processes on synthetic particle populations. Two timing experiments use different physical baselines. The five-kernel experiment starts with native Particula coagulation. The ten-second process comparison already samples collision pairs before replacing any rate with a neural network. Keep each candidate paired with its own baseline.

## Redraw the figures

The [figure download](../../website/public/study-data/weighted-particle-embedding/figure-data.zip) contains the plotted CSVs, both quantitative plotting scripts, source-table checksums and the small condensation fits. After extracting it, run the following commands in the extracted directory with NumPy and Matplotlib installed. They draw the 15 quantitative figures on CPU.

```bash
python exploration_figures.py --from-values . --output figures
python kernel_transfer_figures.py --from-values . --output figures
```

From the repository root, these commands rebuild the public CSVs and download directly from the [source measurements](measurements.zip), redraw the quantitative figures and render the square graphical abstract. The graphical abstract uses Particula's plotting style and the original matched neutral-particle rows.

```bash
export PYTHONPATH=.
s=studies/weighted-particle-embedding
python "$s/exploration_figures.py"
python "$s/kernel_transfer_figures.py"
python "$s/forecast_toc.py"
```

## Repeat the forecasts

The measurements used Particula at `2db6a77323a7`, Python 3.12, an NVIDIA GeForce RTX 5060, PyTorch 2.10.0+cu128 and Warp 1.11.1. The fused condensation experiment also uses Triton 3.6.0. Use the repository's Particula dependencies plus NumPy, SciPy, Matplotlib, PyTorch and Warp. The historical timings depend on the GPU, precision, warm-up and observation schedule. Use the same device for both methods in a timing pair.

Run the following from the repository root with the simulation environment active. The output directory holds new runs separately from the published measurements.

```bash
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
s=studies/weighted-particle-embedding
out=.artifacts/weighted-particle-replay
mkdir -p "$out"
```

### Pair sampling and kernel transfer

The first command repeats the five-kernel comparison at 4,096 and 32,768 particles. It fits the settling and shear heads, warms each solver, then times the native, sampled physical and sampled neural forecasts in randomized order on eight matched seeds. The next commands check rate errors against independent physical kernels and vary the collision clock for shear.

```bash
python "$s/kernel_transfer.py" "$out/kernels"
python "$s/kernel_transfer_audit.py" "$out/kernels"
python "$s/kernel_transfer_shear_check.py" "$out/kernels"
```

For a short installation check, add `--sizes 256 --laws Brownian --repeats 1` to the first command. The source settings and measured rows for the published comparison are in the `kernel-transfer/` folder of [the source archive](measurements.zip). Each row of `trials.csv` is one complete forecast.

The million-particle comparison used native vapor exchange and one simulated second. These drivers run the full native, collision-off and sampled controls. Set a UTC deadline for the campaign, for example eight hours from launch on Linux. Each driver checks the deadline between configurations. To run each campaign for its own window, set the deadline again before launching it.

```bash
deadline=$(date -u -d '+8 hours' --iso-8601=seconds)
python "$s/stage3_native_large.py" "$out/native-large" --million --until "$deadline"
python "$s/stage3_native_off.py" "$out/native-off" --until "$deadline"
python "$s/stage3_native_sampled.py" "$out" --ablation --repeats 128 --until "$deadline"
```

In the source archive, `stage-3/native-vapor-sampled-collisions.csv` records the number of completed runs for each configuration. Its two neutral million-particle rows supply the graphical abstract. They use the same random pairs. The native full-timing control uses an independent event stream.

### Process costs and reversible condensation

The small one-process substitution experiment runs first. The scale experiment then compares two box volumes at the same number concentration, 10⁶ cm⁻³. Both use 32 vapors and 80 steps of 0.125 s. Sixteen vapor lanes initially favor uptake (including the nonvolatile precursor), eight favor evaporation and eight start near equilibrium. Repeat each box with a finer vapor clock to distinguish a change in rate from a change in integration error.

```bash
python "$s/process_cost.py" "$out/process-cost"
for hosts in 32768 1000000; do
  python "$s/process_scale.py" "$out/scale-$hosts" --hosts "$hosts"
  python "$s/process_scale.py" "$out/scale-$hosts" --hosts "$hosts" --refine-vapor
done
python "$s/condensation_options.py" "$out/condensation"
python "$s/condensation_audit.py" "$out/condensation"
```

The [condensation replay notes](condensation-replay.md) give the training draws, timing boundaries and physical time-step refinements. The fitted rate parameters used by the process comparisons are in [examples/stage-2](examples/stage-2/).

### Particle representation, process clocks and formation

| Published comparison | Simulation and analysis entry points |
| --- | --- |
| Equal dry spectra and bulk composition under a vapor pulse | `dataset.py` generates disjoint train, validation, test and challenge families. `fit_encoder.py` fits the weighted encoder and `sections.py` supplies sectional controls. The [reserved comparison criteria](qualification.md) specify how the two mixtures are scored. |
| Neural rate fitting | `stage2_jump_distance.py` refits the neutral collision term with `--widths 4 --seeds 901 --steps 6000`. `stage2_pair_data.py` and `stage2_fit_pair.py` generate and fit pair-rate targets. `stage2_transfer_data.py` and `stage2_fit_transfer.py` generate vapor histories and fit the transfer heads. |
| Restoring physical steps after a forcing change | `stage3_species_clocks.py reference` generates the two-tolerance physical control. `stage3_live_ode.py` runs the separate forecast policies. `stage3_number_power.py` scores seed blocks against the reference. |
| Particle and vapor scaling | `stage3_one_box.py` generates a single explicit box. `stage3_headroom.py` and `stage3_width_confirmation.py` run the size and vapor ladders. `stage3_moving_grid.py` supplies the independent sectional refinement. |
| Newborn particles crossing 5 and 10 nm | `stage3_birth_growth.py` and `stage3_birth_reference.py` compare native birth and growth against a continuous reference. `stage3_newborn_ion_paths.py` follows tracer cohorts through repeated ion attachment. `stage3_ion_arrivals.py` scores the two arrival thresholds across independent batches. |

The small paired aerosol input and the fitted encoder let you exercise the eight-coordinate forecast on CPU. The first step encodes the initial particles and their inventories. The second applies the saved forecaster to that record.

```bash
enc="$s/examples/direct-eight/joint-cycle-w8-s711-encoder-9000.pt"
python "$s/direct_inference.py" encode "$s/examples/one-vapor-pair.npz" "$enc" "$out/pair.npz"
python "$s/direct_inference.py" forecast "$out/pair.npz" "$enc" "$s/examples/direct-eight/first-full-cycle.pt" "$out/forecast.npz"
```

To run the single-box scaling model at one million particles and 32 vapors, use the same initial concentration and family as the capacity ladder.

```bash
python "$s/stage3_one_box.py" "$out/million.json" --n 1000000 --species 32 \
  --density 1e12 --family 101 --seed 212001 --horizon 1 --steps 1 --vmax .02
```

The source archive groups the measurement tables by experiment. Extract it to inspect the original tables and settings, or supply an unpacked directory to `exploration_figures.py --results DIRECTORY` to rebuild the figure download from new measurements. Run the simulation drivers to generate new particle trajectories, restart files and training histories for follow-up analysis.
