# Appendix: Inventory Seed and Evidence Ledger

Discovery on 2026-09-27 found the families below. This is a concrete seed, not
a claim that support classification or migration is complete. P1 expands each
family into one row per source/pair and published snippet, reconciles navigation,
and records unaffected rows too. A new quick start cannot satisfy this ledger.

| Phase | Discovered family / seed paths under docs/Examples | Required disposition |
|---|---|---|
| P1 | Aerosol/Aerosol_Tutorial; Gas_Phase/Notebooks/{Gas_Species,AtmosphereTutorial,Vapor_Pressure} | Native construction and process-owned thermodynamics |
| P1 | Particle_Phase/Notebooks/{Particle_Representation_Tutorial,Distribution_Tutorial,Aerosol_Distributions,Activity_Tutorial,Particle_Surface_Tutorial,Functional/Activity_Functions} | Native particle state; retain functional science |
| P1 | data_containers_and_gpu_foundations.py; Data_Containers/data_containers_and_gpu_foundations.py and index.md | Reconcile canonical entrypoint and landing copy |
| P2 | Dynamics/Condensation/{Condensation_1_Bin,Condensation_2_MassBin,Condensation_3_MassResolved,Condensation_Latent_Heat,Staggered_Condensation_Example} | Migrate all five pairs |
| P2 | Dynamics/Coagulation/Coagulation_1 through Coagulation_5; Charge/*; Functional/* | Inventory every concrete pair; preserve pattern/comparison/functional intent |
| P3 | Chamber_Wall_Loss/Notebooks/{Wall_Loss_Tutorial,Spherical_Wall_Loss_Strategy,Rectangular_Wall_Loss_Strategy,wall_loss_builders_factory,Chamber_Forward_Simulation} | Migrate all five pairs |
| P3 | cpu_dilution.py; Nucleation/cpu_nucleation.py; Nucleation/Notebooks/Custom_Nucleation_Single_Species | Execute scripts and custom pair |
| P4 | Simulations/Notebooks/{Soot_Formation_in_Flames,Organic_Partitioning_and_Coagulation,Cough_Droplets_Partitioning,Cloud_Chamber_Single_Cycle,Cloud_Chamber_Multi_Cycle,Biomass_Burning_Cloud_Interactions} | Migrate all six pairs |
| P4 | Dynamics/Customization/Adding_Particles_During_Simulation; Activity/*; Equilibria/Notebooks/* | Migrate affected construction, explicitly retain unaffected science |
| P5 | gpu_direct_kernels_quick_start.py; gpu_coagulation_direct.py; gpu_condensation_parity_walkthrough.py; gpu_complete_process_sequence.py; Nucleation/gpu_direct_nucleation.py | Preserve direct boundary, explicit transfer and supported imports |
| P5 | gpu_resident_session.py; gpu_resident_multi_timestep.py; gpu_resident_graph_capture.py | Preserve concrete resident lifecycle and CUDA-only capture |
| P6–P7 | All topic indexes, Setup_Particula guidance, feature/API/migration pages and root snippets | Native references, correct links, explicit unsupported history |

Per-row evidence fields: source and notebook paths; published link; support
classification/approver; old/new APIs; owner phase; baseline revision and
scientific quantities/tolerances; regression test; literal sync/execution/build
commands; final revision/date/environment; pass/fail/skip/unavailable; artifact
pointer; remaining blocker. For historical exceptions add location, explicit
non-supported label and reason, with confirmation it is not executable.

References: issue #1602; E9 dependency_map; E9-M1 guidelines_requirements;
E9-M2 success_criteria; E9-M3 scope; E9-M4 phase_details;
`.opencode/guides/{testing_guide,documentation_guide}.md`; `pyproject.toml`.
No execution evidence was produced by drafting. M6 owns legacy deletion;
earlier GPU measurement gaps are not resolved or reclassified here.
