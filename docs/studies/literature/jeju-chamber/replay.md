# Reproduce the Jeju spectrum comparison

Start with the versioned chamber measurements and the archived parameters for the shared and individual fits. The simulations used Python 3.12.12, NumPy, SciPy, pandas, Matplotlib, Warp 1.11.1 and Particula revision `2db6a77323a7ac119f5acce0c180fe6b9e7b555e`. GPU runs used an NVIDIA GeForce RTX 5060. The standard calculation advances particles every 2 s. The refined calculation uses 1 s steps and twice as many size classes for cloud condensation nuclei, salt and ice. Both use the same 70 Welas bins. [Original chamber inputs and extracted measurements](sources.md)

| Product | What to reproduce |
|---|---|
| Six-run contours | Read Welas bins, chamber pressure and thermometry, available vapor and CPI records. Replay shared and individual whole-record settings on the same measurement bins. Divide bin concentrations by the width in log₁₀ diameter and plot a common number scale. Display signed log differences separately. |
| Record means and minute cuts | Average every valid measured minute within each run for the mean. Select the one-minute windows starting at minutes 12, 20 and 30 for the cuts. Use the same mask and bin edges for all three curves. |
| Scoring perturbations | With the fitted distributions frozen, change the count floor and repeat the comparison for occupied measurements. Check minute selection one minute earlier and later. |
| Source and humidity test | Hold independent run-5 parameters fixed. Cross A2 versus earlier optical initialization with the fitted vertical vapor contrast retained or removed. Evaluate all eight grid/case combinations. |
| RF14 aircraft spectra | Use the three RF14 tables in the data bundle to recover the published case-mean size traces and flight context. The fourth table identifies six particles in the Jeju chamber images. The bundle records original sources, units and digitization limits. |

Replaying the shared fit and the six individual fits at both resolutions reproduced all 14 whole-record outputs. Particle spectra, vapor and budgets matched the saved values. The six-run contour plot, record means and one-minute size distributions matched the saved figures pixel for pixel. A separate replay recovered the eight run-5 source and humidity cases and checked the reference score used by the optimizer.

The repository contains the fitted model specification and replay source. Recover original measurements through the cited data providers. Write regenerated spectra and intermediate tables to a local output directory. [Chamber replay and commands](../../../../studies/jeju-chamber/README.md) · [Measurement sources and digitizations](sources.md)

[Six-run results and conclusions](index.md)
