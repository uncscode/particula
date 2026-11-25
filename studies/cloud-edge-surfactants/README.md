# Reproduce the cloud-edge surfactant study

The [study article](../../docs/studies/literature/cloud-edge-surfactants/index.md) follows two measured citric-acid mixtures from trapped droplets to competing particles in a rising parcel. All simulations and graphics use the version of Particula in this checkout. From the repository root, activate a Python 3.12+ environment, install the checkout with its scientific and plotting dependencies, and make Poppler's `pdfimages` available.

## Redraw the published figures on a CPU

From the repository root, run:

```bash
python -m pip install -e '.[extra]' Pillow
cd studies/cloud-edge-surfactants
python fetch_inputs.py
python figures/render.py
python figures/organics.py
python figures/toc.py
```

The input step verifies the publisher's article PDF and ZIP archive and reconstructs Figure 5 and the Figure 6b growth band. Its `--cache DIRECTORY` option accepts copies of the two original publisher files under their original names. The [figure guide](figures/README.md) lists each image and the corresponding columns of plotted model results. The full simulation runs, publisher files and freshly extracted measurements are local products. The article, figures, compact plotting tables and replay source are versioned.

## Regenerate the calculations

The CPU equilibrium calculation can run separately with `python replay.py equilibrium`. CUDA stages use an NVIDIA GPU, `nvcc` and `nvidia-smi`. The recorded runs used an RTX 5060, Python 3.12.12 and the `sm_120` compiler target. On another supported GPU, set `PARTICULA_CLOUD_CUDA_ARCH` to its NVCC architecture before running.

From this study directory, these commands regenerate the measurements-based equilibrium curves, the four-treatment eddy calculation, the parcel identity and height profiles, and the co-condensation perturbations. The extraction step reduces the numerical outputs to the values shown in the figures.

```bash
python replay.py figures
python figures/extract.py
python figures/organics.py --extract
python figures/render.py
python figures/organics.py
python figures/toc.py
```

For all numerical comparisons in the article and companion pages, use `python replay.py all` following input retrieval. A stage accepts `--list` to display its constituent commands. All stages write into this directory or its ignored results directories. For an inexpensive test of the compiled solver, run the eight treatment/placement combinations for a single inlet and write the results to a separate directory.

```bash
python cloud/edge_attribution.py main --inlet 0 --out /tmp/cloud-edge-inlet-0
```

| Replay stage | Question |
|---|---|
| `equilibrium` | How do droplet size, adsorption and dilution affect the measured tension ordering and the critical supersaturation? |
| `identity`, `heights` | Which populations activate at +100 m, and how does an exchange with outside air at +50 m change their counts with height? |
| `edge`, `edge-refinement`, `edge-summary` | How does moving the dry inflow change the flux-weighted count in four matched seed treatments? Run the summary after the two resolutions. |
| `regimes`, `fluctuations`, `loading`, `seed-sizes` | How do background, ascent history, seed number and seed size change activation? |
| `organic-limits`, `smoke-mixing` | How much do water interaction of condensed organics and composition differences between smoke particles matter? |
| `edge-adsorption` | How sensitive are the eight original inlet paths to the recipe assumptions? |
| `cpu-check`, `cuda-reference`, `compare-cpu` | How closely do the tagged cloud counts agree with an independent CPU calculation? Run the comparison after both solvers. |

Numerical refinements test time step, particle size resolution and the interpolation of thermodynamics. The small difference between the two surfactant recipes is also sensitive to the relationship between adsorption and surface tension in the actual citric-acid mixtures. [Laboratory observations](https://doi.org/10.5194/acp-25-14301-2025) · [Numerical interpretation](../../docs/studies/literature/cloud-edge-surfactants/reproduction.md)
