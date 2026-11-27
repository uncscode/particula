# Six-hour multispecies parcel compression

Run from the repository root using the Particula development Python environment with NumPy, SciPy, pandas and Matplotlib. The study uses CPU float64 and a Linux process pool. The published run used Python 3.12.12 and the source in this worktree. The seed list in `run.py` fixes eight calibration histories followed by four held-out histories.

```bash
export PYTHONPATH=. OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
study=studies/parcel-composition/aging
results=$(mktemp -d)
python "$study/run.py" "$results/carriers" --limit 96
python "$study/run.py" "$results/generic" --generic --limit 256
python "$study/run.py" "$results/fine" --seeds 1199 --substeps 4 --limit 96
python "$study/run.py" "$results/half" --seeds 1199 --particles 50000 --limit 96
python "$study/continuum.py" --limit 96
python "$study/continuum.py" --generic --limit 128
python "$study/analyze.py" "$results/carriers"
python "$study/checks.py" "$results/carriers" "$results/generic" "$results/fine" "$results/half"
```

The run directories contain a numerical archive per history with parameters, number counts, reference and reconstructed curves, and size-specific χ. The continuum passes regenerate deterministic trajectories and evaluate the exact largest activation discrepancy between 0.07% and 1.2% supersaturation. Small staged executions can use `run.py --seeds 181 --particles 1800 --limit 10` with a separate output directory. The publication analysis expects the full seed list.

| Saved output | Contents |
| --- | --- |
| `budget.csv`, `continuum_carrier.csv`, `continuum_generic.csv`, `tolerances.csv` | Worst snapshot errors per cap and partition, plus the calibration choice under three sets of tolerances. |
| `selected_scores.csv`, `curves.csv` | Per-snapshot errors and 31 activation settings for the selected carrier-split cap. Activation is a number fraction. Dry absorption is in Mm⁻¹. |
| `profiles.csv`, `shape_scores.csv` | Bin counts, resolved and fitted χ, and errors from reconstructing compositions using those profiles. |
| `refinement.csv` | Changes to resolved predictions and compression scores on a smaller aging step and smaller population. |
| `representatives.csv` | Seed 1441, parcel B, hour zero. Each row contains a zero-based bin, represented number and four species masses in kg **per particle**. The 184 rows represent 100,000 particles. |

The example was selected by largest held-out error at the 31 scored channels. For a row with species masses m and densities ρ, species volumes are m/ρ and dry diameter is (6 Σ m/ρ / π)^(1/3) in metres. Apply the activation and optical operators to each row, then weight by its count. The numerical round-trip in `checks.py` verifies reconstructed activation, inventories and absorption against direct Mie evaluation. A complete scientific interpretation and assumptions appear in the [study page](../../../docs/studies/explorations/parcel-composition/six-hour-aging.md).
