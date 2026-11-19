# Frozen comparison specification

Selected: Crump & Seinfeld (1981), DOI 10.1016/0021-8502(81)90036-7,
Eq. 13 surface integral and Eq. 17 spherical rate. Primary author thesis copy:
https://thesis.caltech.edu/3500/3/crump-jg_1983.pdf#page=39

Question: How much does summing the zero-settling diffusion limit and the
zero-diffusion settling limit bias one-hour survival relative to their coupled
spherical treatment? This quantifies an approximation, not a novel mechanism.
Baseline beta_add = 6 sqrt(ke D)/(pi R) + 3v/(4R).
Higher fidelity: integrate the surface flux v mu/expm1(pi v mu/(2 sqrt(ke D)))
over mu=-1..1 with prefactor 3/(2R). Independently compare Particula's Debye
implementation and prepared CUDA stochastic survival against this oracle.
Observable: beta in s^-1 and surviving fraction after 3600 s.

R=0.30 m and density=1000 kg/m^3 follow Fig. 2's sphere and water density.
T=298.15 K, P=101325 Pa specify the paper's room-temperature air assumption.
ke=0.36,3.6,36 s^-1; only 36 is explicitly Fig. 2's value. Other rates and the
one-hour decay experiment are analysis choices. Nonvolatile water-density
spheres, fixed size, neutral, well mixed, no source, no ventilation, coagulation,
electrostatics or evaporation. No claim of a complete water aerosol experiment.
Inertia omission limits physical application especially for coarse particles.

Reference scan: 121 diameters 0.03–3 micrometres, refined to 241.
Pilot: ke=3.6, diameter 1 micrometre, 4 independent boxes, 1024 unit particles,
dt=30 s. Campaign: each ke, diameters 0.1,0.3,1,3 micrometres, 16 boxes per
size, 2048 particles, dt=30 s; refine dt=15 s and particle count to 8192.
Separate root seeds for each run; no same-seed convergence claim.
Acceptance: oracle quadrature relative tolerance 1e-10; CPU beta discrepancy
<0.5%; CUDA means within 5 binomial standard errors +0.002 of oracle;
finite nonnegative inventories; mass survivor ratio matches number ratio to
1e-12. Quantify 95% Monte Carlo intervals and refinement shifts. Parameter
sensitivity is not experimental uncertainty. No observational uncertainty is
available for this synthetic comparison. Pilot cost determines scaling.

Stop at 4 h simulation / 5 h daily / 100 MiB packet / 2 GiB archive.
Retain source, compact results, figure, logs and environment. No publication or
issue filing without researcher review. At most one new study, as instructed
by scheduler and user (overrides older Stage 3B three-study language).
