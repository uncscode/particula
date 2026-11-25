# Reproducing the surfactant and cloud-edge comparisons

The measured surface tensions come from the publisher's workbook. The parcel and cloud-edge counts come from simulations. The figure tables contain the model values shown in the figures. You can redraw the paper on a CPU, including the relative activation plot, or repeat the cloud calculations with CUDA. [Study source and figure tables](https://github.com/uncscode/particula/tree/main/studies/cloud-edge-surfactants)

## Recover the measurements and redraw the figures

Start in the repository root with Python 3.12 or later, Particula, NumPy, SciPy, Matplotlib, Pillow and Poppler's `pdfimages` on the path. With the Python environment active, run the following commands.

```bash
cd studies/cloud-edge-surfactants
python fetch_inputs.py
python figures/render.py
python figures/organics.py
python figures/toc.py
```

The first command after changing directory retrieves the article and its supplement from the publisher, checks their SHA-256 digests, then reconstructs two inputs. For Figure 5, it extracts the 55 instrument and recipe records in columns B–D of the workbook's fourth worksheet. Figure 6b supplies a growth-factor band. The extractor reads the colored band at each integer relative humidity from 10% through 94%. The band constrains the range of growth consistent with the published image. Its edges do not represent independent measurements at each humidity. A second run reuses the verified downloads. [Original article](https://doi.org/10.5194/acp-25-14301-2025) · [Publisher's workbook archive](https://acp.copernicus.org/articles/25/14301/2025/acp-25-14301-2025-supplement.zip)

## Repeat the numerical experiments

The replay instructions list the CPU equilibrium calculations and the CUDA runs that produce the population and trajectory endpoints used by the figures. Run them before the figure-data extraction step when changing an assumption. Redrawing from the supplied tables reproduces only the published assumptions. [Replay commands and requirements](https://github.com/uncscode/particula/blob/main/studies/cloud-edge-surfactants/README.md)

The equilibrium calculation combines the water activity of the solution with the Kelvin correction for its curved surface. At each dry diameter, the peak of this Köhler curve gives the critical supersaturation. The relative-change panel on the molecule-budget page reports the percentage change from pure acid at the same dry diameter and model settings. Each adsorption scenario conserves the amount of surfactant on the surface and in solution combined. The mixtures were measured at 295 K. Applying those mixture relations during a cooler ascent is an additional modeling assumption.

The cloud calculation uses Particula parcel thermodynamics in the CPU reference and a study-specific CUDA solver for the shared water, heat and organic budgets. In a closed parcel without the added organic vapor, the maximum discrepancy between the GPU and the 128-class CPU reference was 0.27 droplets/cm³ for an individual target population across the tested clean and background cases. This check tests the parcel integration independently of the assumed adsorption law.

The original cloud runs used Python 3.12.12 and an NVIDIA GeForce RTX 5060. Each modeled parcel starts saturated at 285 K and 90 kPa. Individual dry size classes carry number weights. Every population shares the water and organic vapor in its parcel. Concentrations refer to the parcel volume at the reported height. A droplet counted at +100 m must have exceeded its critical wet radius at each half-second sample over the preceding five seconds.

For the central cloud-edge comparison, each of 128 inlet positions has four seed treatments and two placements of the dry-air intrusion. The 48-class calculations use 1,025 thermodynamic lookup points and three passes to refine the dry sizes at the activation threshold. A matched subset of 16 positions uses 96 classes, 2,049 lookup points and half the time step. Inlet vertical speed sets the weight of each trajectory. These are prescribed streamlines and tested numerical resolutions. The smoke background and the two humidity fields are idealized.

The sign of the small chemistry effect on the placement response changed under refinement. Across the 16 matched positions the estimate moved from −0.0065 to +0.0216 droplets/cm³. The change from imposing fixed tensions remained much larger, at 17.44 and 17.28 droplets/cm³ on those same positions. Comparing these paired results avoids attributing a change in sampled streamlines to a change in numerical resolution. [Cloud-edge results and controls](cloud-edge.md)
