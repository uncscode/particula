# Reproduce the Jeju spectrum comparison

The source measurements are versioned. The chamber fits used Python 3.12.12, NumPy, SciPy, pandas, Matplotlib, Warp 1.11.1 and Particula revision `2db6a77323a7ac119f5acce0c180fe6b9e7b555e`. GPU runs used an NVIDIA GeForce RTX 5060. The standard model uses a 2 s timestep. The refined model uses 1 s and doubles the CCN, salt and ice cohorts while retaining the original Welas bins. [Original chamber inputs and extracted measurements](sources.md)

| Product | What to reproduce |
|---|---|
| Six-run contours | Read Welas bins, chamber pressure and thermometry, available vapor and CPI records. Replay shared and individual whole-record settings on the same measurement bins. Divide bin concentrations by the width in log₁₀ diameter and plot a common number scale. Display signed log differences separately. |
| Record means and minute cuts | Average every valid measured minute within each run for the mean. Select the one-minute windows starting at minutes 12, 20 and 30 for the cuts. Use the same mask and bin edges for all three curves. |
| Scoring perturbations | With the fitted distributions frozen, change the count floor and repeat the comparison for occupied measurements. Check minute selection one minute earlier and later. |
| Source and humidity test | Hold independent run-5 parameters fixed. Cross A2 versus earlier optical initialization with the fitted vertical vapor contrast retained or removed. Evaluate all eight grid/case combinations. |
| Field extraction | Recover the published RF14 figures or use the four small annotated and extracted tables in the data bundle. The bundle contains original figure citations, units and reading limits. |

Replaying the selected physical settings regenerated the pooled fit and six individual fits at each resolution, covering all 14 whole-record outputs. The simulated spectra, vapor and budgets matched the retained runs. The time–size panels, means and minute cuts matched pixel for pixel. The same source recovered the eight new run-5 perturbations. The optimizer reference evaluation was checked separately.

The repository contains the fitted model specification and replay source. Recover original measurements through the cited data providers. Write regenerated spectra and intermediate tables to a local output directory. [Chamber replay and commands](../../../../studies/jeju-chamber/README.md) · [Measurement sources and digitizations](sources.md)

[Six-run results and conclusions](index.md)
