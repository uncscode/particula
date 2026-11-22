Numerical evidence for the polluted Π Chamber droplet-gain study
================================================================

Reference experiment: Yeom et al. (2026), Geophysical Research Letters.
https://doi.org/10.1029/2026GL123212
Accepted manuscript: https://www.osti.gov/servlets/purl/3484638
Author data catalog: https://digitalcommons.mtu.edu/all-datasets/79/

B denotes the seeded steady background. E denotes dry-air entrainment with
continued seeding. Concentrations are particles per cm3, visible liquid is
mg/m3, and optical particle sizes are wet diameters. Cloud starts at 2 µm.
Each file retains the column names and precision of the research calculation.

observations/       Class counts printed in section 3.1 and water/diameter
                    read from Fig. 1. The Fig. 1b spectra were digitized in
                    log diameter, integrated into 112 bins and scaled to the
                    reported haze/cloud counts. lidar_profiles.csv contains
                    251 digitized range-corrected Fig. 2a markers (515 nm).
                    All four cases use the first experimental stage.
transfer-scores.csv, count-bounds.csv, sorting-scores.csv, optics-transfer-paired.csv
                    Clean-to-polluted transfer, count bound, vertical sorting,
                    and range-matched optical changes.
joint/              Pooled four-state bulk/spectrum/lidar fits, multiple ages,
                    dilution boundary, measurement and calibration checks.
exposure/           Source and exposure-time Monte Carlo, transfer between
                    loadings, direct effects, seed-budget and clock tests.
shared_vapor/       Finite vapor, short-history refits, long feed, pulses,
                    independent sampling phases and synthetic recovery.
wall_loss/          Residence-time, sidewall and steady floor-export tests.
maintained/         The highlighted case, finer replay, and reconciled summaries
                    across the two Sobol scrambles and nine configurations.
                    'cloud' excludes haze; 'visible' includes it.
background_transport/  Equal-feed, localized vs distributed backgrounds,
                    tagged cohorts, size-class exports, instrument fit scores,
                    clean floor-flux proxy and numerical checks. Each cohort
                    row contains an ensemble fraction at a given elapsed time.

The four-second-kinetics.npz file is an input to the downloadable CUDA replay.
For a 130 nm NaCl core in 20 °C air, the archived Particula growth rates give
wet-bin endpoints over a four-second fixed-humidity interval. The mass pivot
remap partitions number between neighboring bins, conserving liquid water.
The native kernel solves the local gas budget before applying these transfers.
rh is relative humidity as a fraction. diam is wet diameter in µm. mass and
endpoint are mg/m3 per particle/cm3. dest is the zero-based lower destination
index; weight is the fraction placed in the next bin. seed_bin and seed_weight
initialize the hydrated core. saturation is the saturation vapor concentration
in mg/m3. Settling speed is recomputed from Particula in the replay.

The tables are derived observations and saved simulation results. The ZIP
re-executes the six-hour spin-up, paired six-hour background and treatment,
and final-hour counter predictions of the 'Local dry inlet' row in
maintained/matched-example.csv. The original ensemble sweeps are summarized
in the other tables. Full individual GPU time histories are omitted.
