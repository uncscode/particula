# Connect processes to a research question

These worked simulations combine population construction, thermodynamic inputs,
time evolution, and diagnostics. Choose a question, then inspect the example's
assumptions and validation boundary before adapting its parameters. A successful
run or an attractive size distribution alone does not establish a reliable
material model.

| Question | Example | What it helps you distinguish |
| --- | --- | --- |
| Does a conserved vapor budget guarantee correct growth timing? | [Finite-reservoir growth](adaptive_reservoir_depletion.py) | Independent radius/vapor errors, depletion, and numerical work |
| Does adaptive CUDA meet the same depletion accuracy target? | [CUDA reservoir comparison](adaptive_reservoir_depletion_cuda.py) | Real-device accuracy, explicit downloads, and the limits of a small benchmark |
| Can CUDA resolve stiff evaporation after a humidity reset? | [CUDA wet–dry relaxation](adaptive_cloud_relaxation_cuda.py) | ROS3 convergence, closed-phase budgets, and explicit host control |
| Are wet–dry oscillations physical or numerical? | [Adaptive droplet relaxation](adaptive_cloud_relaxation.py) | Fixed-step error, adaptive tolerance refinement, and independent reference agreement |
| How do collision assumptions change a water-grown mixed population? | [Biomass-burning-inspired aerosol](biomass_burning_cloud_interactions.py) | A common starting state, continued condensation, and alternative collision kernels |
| Can a plausible size spectrum hide a species-inventory failure? | [Organic uptake followed by binned coagulation](organic_partitioning_and_coagulation.py) | A valid closed condensation check versus unsupported fixed-composition coagulation |
| What changes when a vapor reservoir is reset from wet to dry? | [One chamber cycle](cloud_chamber_single_cycle.py) | Water exchange, surviving particles, and the limits of a fixed timestep |
| How do repeated cycles change water content and dry retention? | [Four chamber cycles](cloud_chamber_multi_cycle.py) | Internally mixed versus single-material seeds, threshold definitions, and diagnostic weighting |
| Can a property lookup invalidate an otherwise executable model? | [Respiratory-droplet input audit](cough_droplets_partitioning.py) | Requested chemical names versus resolved identities; the retained results are not valid respiratory-fluid predictions |
| How does imposed cooling affect uptake onto existing seeds? | [Cooling organic aerosol](soot_formation_in_flames.py) | Partitioning and collisions versus soot chemistry, nucleation, and thermal feedback |

For a first coupled calculation, start with the single chamber cycle after the
[process examples](../processes/index.md). Use the multi-cycle example when
comparing composition and removal, and the biomass-inspired example when comparing
collision mechanisms from a common initial state. The organic binned and
respiratory-droplet examples deliberately expose limitations that must be resolved
before drawing their originally suggested research conclusions.

The scripts preserve complete workloads and saved outputs. Where a study needs
a material property, convergence result, or experimental comparison that an
example does not supply, its closing section identifies a concrete next test.

Follow the [time-resolved Π Chamber spectra](pi_chamber_spectra.py) to compare dry and wet starts under continuous NaCl injection.
