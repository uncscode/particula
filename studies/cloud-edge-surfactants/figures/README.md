# Redraw the cloud-edge figures

Run these commands from the repository root with the study's Python environment. The gallery thumbnail and cloud figures read only the checked-in model coordinates.

```bash
python -m scripts.render_literature_tocs cloud-edge-surfactants
python studies/cloud-edge-surfactants/figures/render.py --figure identity
python studies/cloud-edge-surfactants/figures/render.py --figure edge-attribution
python studies/cloud-edge-surfactants/figures/render.py --figure cloud-edge
```

After the study's input fetch creates local `figure5.csv`, `python studies/cloud-edge-surfactants/figures/render.py` redraws all four report figures. The equilibrium panel reads the measured tensions and reported uncertainties directly from that local extraction of Shahabadi et al. The gallery thumbnail also has a standalone entry point at `figures/toc.py`. All five images are written to `docs/images/cloud-edge-surfactants/`. [Publisher's Figure 5 and data supplement](https://doi.org/10.5194/acp-25-14301-2025)

| Image | Saved graph data in the study's `plot_data/` directory |
| --- | --- |
| `toc.png` | `toc.csv` gives the six displayed flux-weighted counts, rounded to whole droplets per cm³. |
| `equilibrium.png` | `inventory.csv`, `tension-contrast.csv` and `activation-shift.csv` give the modeled inventory bounds, tension differences and critical supersaturation shifts relative to pure acid. The first panel also needs the locally fetched observations. |
| `identity.png` | `identity.csv` gives activated octyl, dodecyl and background droplet counts in the marine and urban calculations. |
| `edge-attribution.png` | `edge-attribution-paths.csv` gives two paired shifts per inlet. `edge-attribution-effects.csv` gives the three contrasts for each dry-inflow placement. |
| `cloud-edge.png` | `cloud-edge.csv` gives activated counts at six sampling heights for each closed or exchanging parcel. |

After a scientific replay, regenerate the compact tables with `python studies/cloud-edge-surfactants/figures/extract.py --source studies/cloud-edge-surfactants`. That command reads `equilibrium_scans.csv`, `budget.csv`, `cloud/results/gpu_identity_populations.csv`, `cloud/results/milestones.csv` and `cloud/results/edge-attribution/main.csv`. Use `--output PATH` to compare regenerated tables in a separate directory.

## Organic-vapor comparison

The optional `organics.png` graph uses the eight rows of `plot_data/organics.csv`. Each point is the change in total activated droplets per cm³ at 100 m relative to a no-gas parcel with the same background and updraft. Both organic runs start with 1 µg/m³ of vapor, with a saturation mass concentration of 10 µg/m³ at 298.15 K and an accommodation coefficient of 0.5. The soluble surrogate allows the organic to partition into all liquid water and supplies one effective osmole per mole. The weakly water-interacting surrogate makes 0.1% of the water available for partitioning and supplies 0.1 effective osmole per mole. Both settings use the same 5% pulse of 90% RH environmental air at 50 m. The two changes in organic behavior were varied together.

From the repository root, redraw the graph with the saved values:

```bash
python studies/cloud-edge-surfactants/figures/organics.py
```

To regenerate the values, run `python replay.py organic-limits` in the study directory after the input fetch. Then run the following from the repository root:

```bash
python studies/cloud-edge-surfactants/figures/organics.py --extract
```

The focused replay stores both treatments and their matched controls in `cloud/results/gpu_organic_limits.csv`. For the original campaign layout, pass `--source PATH` pointing to the archived study directory. Its no-gas controls are in `cloud/results/gpu_factorial.csv`.
