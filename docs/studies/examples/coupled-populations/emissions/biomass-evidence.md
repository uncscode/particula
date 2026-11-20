# Full resident biomass publication

The initial resident campaign ran on an NVIDIA RTX 5060 at revision `9c548d9d617a93cd23a45773426f751e9e6ed9ec`. Later reruns retain their own revisions and observations below.


The 10,000-particle, three-material biomass workload completed on the same
revision and RTX 5060 in **4,148.09 s** with concurrent GPU work. It retained
0.01 s initial adjustment, 600 s preparation and all four independent 3,600 s
branches, each with 10,000 outer intervals of 0.36 s. The user-authorized model
change replaces AO2008/DNS with **ST1956 turbulent shear**, retaining
`epsilon = 0.1 m²/s³` and fluid density `1.225 kg/m³`. This is a changed collision
model, not a CUDA implementation or validation of AO2008.

All four branches passed the unchanged per-species inventory assertion
(`rtol=1e-12`, `atol=1e-30`). Each coupled branch asserted zero physical-state
uploads and downloads between process stages. Adaptive decisions remained in
native CUDA conditional graphs. Physical-state downloads were limited to
declared observations; control metadata was read at completed outer intervals.

| Coupled branch | Accepted condensation steps | Rejected steps | Domain rejections |
| --- | ---: | ---: | ---: |
| Brownian | 31,450 | 7,221 | 0 |
| Brownian + SP2016 settling | 21,902 | 3,612 | 0 |
| ST1956 turbulent shear | 27,602 | 6,101 | 0 |

The refreshed biomass bundle retains the executed source, package hash,
revision, device identity, scientific stdout and all four figures. The final
size-distribution and mixing-state figures explicitly identify turbulent shear.
These are completion and conservation checks for this illustration, not a
collision-model convergence study or experimental validation.


### Biomass rerun with adaptive GPU coagulation

The current [biomass example](biomass_burning_cloud_interactions.py)
completed the same 600 s preparation and four one-hour branches on the RTX 5060.
The original seeds, population of 10,000, event order, condensation tolerances
and 0.36 s outer interval were retained. The sampler can subdivide just the
collision stage when a full interval needs more disjoint pairs than the active
population permits. It recomputes the rate bound after each admitted pass.

The run used revision `d1bdf025f1` and took 2,131 s. The preceding scientific reference used `9c548d9d61`. The table compares final gas-water saturation ratios for one seeded realization.

| Branch | Split intervals out of 10,000 | Most passes in one interval | Mergers | Final saturation ratio | Archived ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Condensation alone | — | — | — | 1.000102995 | 1.000102995 |
| Brownian | 0 | 1 | 240 | 1.000102961 | 1.000102961 |
| Brownian + settling | 179 | 2 | 9,319 | 1.000038092 | 1.000025847 |
| ST1956 shear | 0 | 1 | 448 | 1.000102017 | 1.000102017 |

All four branches passed the per-species gas-plus-particle inventory check (relative tolerance 10⁻¹², absolute tolerance 10⁻³⁰ kg). In each collision branch, the loss of occupied particle slots equaled the number of accepted mergers, and the GPU passed particle and gas state directly between stages. The comparison covers one stochastic realization.
