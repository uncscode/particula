# Reproduce the Liu terpene evaporation study

Liu et al. measured how organic aerosol from four terpene precursors evaporated in dry and humid chambers. The 220 nm humid β-ocimene particles retained about 41% of their initial apparent dry volume overnight. This study tests two explanations against those observations and transfers each fitted mechanism to a hypothetical 60 nm cohort of the same composition. The observed quantity is the cube of the late-to-initial vacuum aerodynamic diameter ratio. Its interpretation as geometric volume assumes stable effective density. [Published experiment](https://doi.org/10.1021/acsestair.5c00226) · [Study](../../docs/studies/literature/evaporation-and-uptake/liu-terpene/index.md)

## Replay from the digitized measurements

From the repository root, use a Python 3.12+ environment with this Particula checkout, NumPy, pandas, SciPy, Matplotlib and Warp. The original calculation used Particula 0.3.6 at revision `2db6a77323a7ac119f5acce0c180fe6b9e7b555e`, NumPy 2.3.4, SciPy 1.16.2, Matplotlib 3.10.7 and Warp 1.11.1. The finite-chamber fit uses a C++20 compiler. The full replay also needs a CUDA-capable NVIDIA GPU, working driver and `nvcc`. The reference device was an RTX 5060.

```bash
run_dir="$(mktemp -d)/liu"
python studies/liu-terpene/replay.py all "$run_dir"
```

The driver stages the scripts and six digitized/transcribed inputs in a fresh directory, reconstructs the reading bounds, fits the chamber models, screens the two mechanisms and runs the viscosity and cloud checks. All intermediate fits, numerical checks, runtime records and generated graphics go under that directory. Set `NVCC` to select a CUDA compiler or `CUDA_ARCH` to select a target such as `sm_120`; the default target is the local GPU. Each calculation imports Particula from the checkout containing the driver. [Replay driver](replay.py) · [Digitized data](../../docs/data/liu-terpene/digitized.zip)

For the central comparison alone, run the first three stages below. For the chamber reference and its Figure 1, the first two are sufficient on CPU.

```bash
run_dir="$(mktemp -d)/liu"
python studies/liu-terpene/replay.py inputs "$run_dir"
python studies/liu-terpene/replay.py chamber "$run_dir"
python studies/liu-terpene/replay.py mechanisms "$run_dir"
cmp "$run_dir/figure2.csv" docs/data/liu-terpene/figure2.csv
```

The optional `viscosity` (CPU) and `cloud` (CUDA) stages each read the fitted results of `chamber` in the same directory. The cloud input is the size-dependent survival at 200 minutes. A new archive can be selected with `--archive PATH` on the `inputs` or `all` invocation, provided its six input files have the documented layout. Numerical comparisons across different compilers, library releases or devices should allow for floating-point variation.

## Regenerate the graphics from saved measurements and compact figure data

The downloadable Figure 2 table contains the 20-hour envelopes for each precursor, mechanism and starting diameter under the half-bin assumption. Two additional rows retain the full-bin check at 60 nm for ocimene. The combined table has 50 rows. The TOC and Figure 2 both use the half-bin rows. The model table stores fractions of initial volume. The renderer shows percentages. [Figure 2 table](../../docs/data/liu-terpene/figure2.csv) · [Reducer](figure2_table.py)

```bash
run_dir="$(mktemp -d)/liu"
python studies/liu-terpene/replay.py inputs "$run_dir"
PYTHONPATH="$PWD" python studies/liu-terpene/replot.py \
    --measurement-bounds "$run_dir/uncertainty/results/read_bounds.csv" \
    --size-data docs/data/liu-terpene/figure2.csv --out "$run_dir"
```

## Source-to-figure guide

| Result | Input and calculation | Generated files under the run directory |
| --- | --- | --- |
| Measured humid evaporation and its reading precision | Figure 4 marker centers, Figure S4 sampled peaks and printed ages. Reconstruct each volume range from the sampled peak spacing and a shared initial diameter. [Extraction and calibration](sources/extract.py) · [Peak timing](sources/check_sample_times.py) · [Reading bounds](uncertainty/estimate.py) | `uncertainty/results/read_bounds.csv` |
| Figure 1 and the common-size comparison | Reconstruct the published seven-bin fractions, fit cyclic and acyclic aerosols separately, then transfer the cyclic fit with the acyclic size and density. [Chamber](model/chamber.py) · [Published-bin reconstruction](model/published.py) · [Finite vapor](model/finite_transfer.py) · [Density](model/density_check.py) · [Figures](model/figures.py) | `model/results/chamber_comparison.png`, `matched.csv`, `density_sensitivity.csv`, `curves.csv` within `model/results` |
| Overnight timing and size-only bounds | Score native-size overnight predictions from daytime fits. Bound size and time effects between adjacent measured points. [Competing mechanisms](model/mechanisms.py) · [Monotone size transfer](checks/clock_test.py) · [Retained fraction](checks/retained_pool.py) | `model/results/overnight.png`, `model/results/mechanisms.csv`, `checks/marker_bounds.csv` |
| TOC, Figure 2 and the extra-size experiment | Transfer the wet cyclic volatility weights, then vary the added low-volatility fraction or the evolution of diffusivity. Accept each parameter set only if the same initial diameter can reproduce all paired readings. Repeat with a full-bin allowance, alternative observation selections and finite vapor removal. [Transfer model](transport/internal_supply.py) · [CUDA kernel](transport/four_precursors.cu) · [Measurement-based screen](uncertainty/rescreen.py) · [Finite-vapor sensitivity](uncertainty/finite.py) · [TOC and figure renderer](replot.py) | `toc.svg`, `size-test.png`, `figure2.csv`, `uncertainty/results/model_ranges.csv`, `uncertainty/results/finite_vapor.csv` |
| Figure 3 and cloud-density sensitivity | Carry each 200-minute size spectrum into a separate closed parcel. Compare activated fractions over updraft and water-affinity scenarios, refine around the critical size and test evolving effective density. [Parcels](parcel/parcel.py) · [Cutoff refinement](parcel/cutoff.py) · [Density correction](density/cloud_density.py) · [Cloud figure](parcel/graphics.py) | `parcel/cloud-activation.png`, `parcel/cutoff/cutoff-comparisons.csv`, `density/results/cloud_density.csv` |
| Viscosity, tracer mobility and radial structure | Convert published composition-based or coalescence-based viscosities to illustrative tracer diffusivities. Compare the uniform resistance and resolved spherical diffusion models, change the tracer law, and test an initially volatile-rich surface layer. [Forecasts](viscosity/predict.py) · [Whole-particle fractional law](viscosity/fractional/fractional.py) · [Radial model](viscosity/radial/radial.py) · [Layer refinement](viscosity/radial/layer_refine.py) · [Humid ocimene](viscosity/ocimene_size.py) | `viscosity/results/viscosity_forecasts.png`, `viscosity/fractional/results/fractional_comparison.png`, `viscosity/radial/radial.png`, `viscosity/ocimene_size.csv` |

## Inspect the source readings

The source archive contains `sources/observations.csv` (calibrated Figure 4 markers with pixel positions), `sources/fractions.csv` (Tables S4 and S5), `sources/time_verification.csv` (spectral sampling times), `sources/peak_check.csv` (manual peak-center cross-check), `uncertainty/peaks.csv` (native Figure S4 peak coordinates and adjacent samples), and `viscosity/figure7/figure7_viscosity.csv` (composition-based viscosity readings). The overnight points share one initial size per precursor. These same-fill observations constrain the models but cannot estimate between-fill variance.

To inspect the original graphics, run the retrieval script from an empty scratch directory with curl and Poppler available. It downloads the article and supplement, checks the two PDFs against their SHA-256 digests, and extracts the relevant figure images. The native Figure 7 extraction uses Pillow and NumPy. The inputs derive from Sijia Liu and coauthors’ article and supplement under CC BY 4.0. [Retrieval script](sources/obtain_sources.sh) · [Figure 7 calibration](viscosity/figure7/extract_fig7.py) · [Liu et al.](https://doi.org/10.1021/acsestair.5c00226) · [Reuse terms](https://creativecommons.org/licenses/by/4.0/)
