# P1 temporary aggregate consumers

`particula._legacy_aerosol.Aerosol` preserves the old aggregate and its reference/mutation behavior without conversion. No native aggregate is wrapped. Removal owner: E9-M6, after every consumer below migrates and enduring behavioral tests pass.

- `particula/aerosol_builder.py` — M2-P3
- `particula/runnable.py` — M4
- `particula/dynamics/dilution.py` — M4
- `particula/dynamics/particle_process.py` — M4
- `particula/equilibria/equilibria.py` — M4
- `particula/integration_tests/coagulation_integration_test.py` — M4
- `particula/integration_tests/charged_coagulation_comparison_test.py` — M4
- `particula/integration_tests/condensation_latent_heat_conservation_test.py` — M4
- `particula/integration_tests/condensation_particle_resolved_test.py` — M4
- `particula/tests/runnable_test.py` — M4
- `particula/tests/aerosol_builder_test.py` — M2-P3
- `particula/tests/execution_test.py` — M4
- `particula/tests/legacy_aerosol_test.py` — M6 legacy-only removal
- `particula/execution/__init__.py` — M4
- `particula/dynamics/tests/wall_loss_runnable_test.py` — M4
- `particula/dynamics/tests/dilution_runnable_test.py` — M4
- `particula/dynamics/tests/dilution_test.py` — M4
- `particula/dynamics/tests/nucleation_runnable_test.py` — M4
- `particula/dynamics/condensation/tests/condensation_strategies_test.py` — M4
- `particula/execution/adapters/coagulation.py` — M4
- `particula/execution/adapters/condensation.py` — M4
- `particula/execution/tests/coagulation_integration_test.py` — M4
- `particula/execution/tests/condensation_integration_test.py` — M4
- `particula/execution/tests/coagulation_adapter_test.py` — M4
- `particula/execution/tests/condensation_adapter_test.py` — M4
- `docs/Examples/cpu_dilution.py` — M5
- `docs/Examples/Aerosol/Aerosol_Tutorial.ipynb` — M5
- `docs/Examples/Aerosol/Aerosol_Tutorial.py` — M5
- `docs/Examples/Nucleation/cpu_nucleation.py` — M5
- `docs/Examples/Chamber_Wall_Loss/Notebooks/Wall_Loss_Tutorial.ipynb` — M5
- `docs/Examples/Chamber_Wall_Loss/Notebooks/Wall_Loss_Tutorial.py` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_1_PMF_Pattern.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_1_PMF_Pattern.py` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_2_PDF_Pattern.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_2_PDF_Pattern.py` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_3_Particle_Resolved_Pattern.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_3_Particle_Resolved_Pattern.py` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_4_Compared.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_4_Compared.py` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_5_Time_Evolution.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_5_Time_Evolution.py` — M5
- `docs/Examples/Dynamics/Condensation/Staggered_Condensation_Example.ipynb` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_1_Bin.ipynb` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_1_Bin.py` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_2_MassBin.ipynb` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_2_MassBin.py` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_3_MassResolved.ipynb` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_3_MassResolved.py` — M5
- `docs/Examples/Dynamics/Condensation/Staggered_Condensation_Example.py` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py` — M5
- `docs/Examples/Dynamics/Customization/Adding_Particles_During_Simulation.ipynb` — M5
- `docs/Examples/Dynamics/Customization/Adding_Particles_During_Simulation.py` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_1_PMF.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_1_PMF.py` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_2_PDF.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_2_PDF.py` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_3_compared.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_3_compared.py` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_4_ParticleResolved.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_4_ParticleResolved.py` — M5
- `docs/Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.ipynb` — M5
- `docs/Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.py` — M5
- `docs/Examples/Simulations/Notebooks/Biomass_Burning_Cloud_Interactions.ipynb` — M5
- `docs/Examples/Simulations/Notebooks/Biomass_Burning_Cloud_Interactions.py` — M5
- `docs/Examples/Simulations/Notebooks/Cough_Droplets_Partitioning.ipynb` — M5
- `docs/Examples/Simulations/Notebooks/Soot_Formation_in_Flames.ipynb` — M5
- `docs/Examples/Simulations/Notebooks/Soot_Formation_in_Flames.py` — M5
- `docs/Examples/Simulations/Notebooks/Cough_Droplets_Partitioning.py` — M5

AerosolBuilder remains legacy until P3. Native construction does not imply native runnable support.

- `particula/integration_tests/quick_start_test.py` — M5

Scientific strategy normalization remains M3-owned; aggregate-based orchestration and adapter consumers above migrate in M4. Legacy arrays and metadata remain held by reference; replacements remain unchecked legacy assignments. The temporary module neither wraps native Aerosol nor aliases additional authoritative storage.

- `examples_tests/dilution_example_test.py` — M5 removes the temporary import exception.
- `examples_tests/nucleation_example_test.py` — M5 removes the temporary import exception.

## P3 builder consumers

The old builder now lives beside the old aggregate in `particula._legacy_aerosol`. It retains references and its historical strategy/count checks. No native container is converted. These consumers migrate in M5 and the legacy builder is removed in M6.

- `particula/integration_tests/quick_start_test.py` — M5
- `docs/Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.py` — M5
- `docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.py` — M5
- `docs/Examples/Simulations/Notebooks/Organic_Partitioning_and_Coagulation.py` — M5
- `readme.md` — M5
- `docs/index.md` — M5
- `plugins/particula-helper/scripts/run_example.py` — M5
- `particula/tests/legacy_aerosol_builder_test.py` — M6 legacy-only removal

The helper example selects the legacy module for this development checkout and retains the released 0.2.x public builder when that module is absent. Its generated OpenCode copy follows the canonical source. Native builder tests replace the public legacy assertions in P3.

## Final active import inventory (P6)

The earlier lists preserve introducing-phase history. Public `aerosol_builder.py`
and its native tests no longer consume legacy APIs after P3. The authoritative
active list below is derived from current imports/snippets. Every row has M6
removal ownership after its listed migration owner completes.

- `docs/Examples/Aerosol/Aerosol_Tutorial.ipynb` — M5
- `docs/Examples/Aerosol/Aerosol_Tutorial.py` — M5
- `docs/Examples/Chamber_Wall_Loss/Notebooks/Wall_Loss_Tutorial.ipynb` — M5
- `docs/Examples/Chamber_Wall_Loss/Notebooks/Wall_Loss_Tutorial.py` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_1_PMF_Pattern.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_1_PMF_Pattern.py` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_2_PDF_Pattern.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_2_PDF_Pattern.py` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_3_Particle_Resolved_Pattern.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_3_Particle_Resolved_Pattern.py` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_4_Compared.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_4_Compared.py` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_5_Time_Evolution.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Coagulation_5_Time_Evolution.py` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_1_PMF.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_1_PMF.py` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_2_PDF.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_2_PDF.py` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_3_compared.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_3_compared.py` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_4_ParticleResolved.ipynb` — M5
- `docs/Examples/Dynamics/Coagulation/Functional/Coagulation_Basic_4_ParticleResolved.py` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_1_Bin.ipynb` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_1_Bin.py` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_2_MassBin.ipynb` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_2_MassBin.py` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_3_MassResolved.ipynb` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_3_MassResolved.py` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb` — M5
- `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py` — M5
- `docs/Examples/Dynamics/Condensation/Staggered_Condensation_Example.ipynb` — M5
- `docs/Examples/Dynamics/Condensation/Staggered_Condensation_Example.py` — M5
- `docs/Examples/Dynamics/Customization/Adding_Particles_During_Simulation.ipynb` — M5
- `docs/Examples/Dynamics/Customization/Adding_Particles_During_Simulation.py` — M5
- `docs/Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.ipynb` — M5
- `docs/Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.py` — M5
- `docs/Examples/Nucleation/cpu_nucleation.py` — M5
- `docs/Examples/Simulations/Notebooks/Biomass_Burning_Cloud_Interactions.ipynb` — M5
- `docs/Examples/Simulations/Notebooks/Biomass_Burning_Cloud_Interactions.py` — M5
- `docs/Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.ipynb` — M5
- `docs/Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.py` — M5
- `docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb` — M5
- `docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.py` — M5
- `docs/Examples/Simulations/Notebooks/Cough_Droplets_Partitioning.ipynb` — M5
- `docs/Examples/Simulations/Notebooks/Cough_Droplets_Partitioning.py` — M5
- `docs/Examples/Simulations/Notebooks/Organic_Partitioning_and_Coagulation.ipynb` — M5
- `docs/Examples/Simulations/Notebooks/Organic_Partitioning_and_Coagulation.py` — M5
- `docs/Examples/Simulations/Notebooks/Soot_Formation_in_Flames.ipynb` — M5
- `docs/Examples/Simulations/Notebooks/Soot_Formation_in_Flames.py` — M5
- `docs/Examples/cpu_dilution.py` — M5
- `docs/Features/particle-data-migration/construction.md` — M5
- `docs/index.md` — M5
- `examples_tests/dilution_example_test.py` — M5
- `examples_tests/nucleation_example_test.py` — M5
- `particula/dynamics/condensation/tests/condensation_strategies_test.py` — M4
- `particula/dynamics/dilution.py` — M4
- `particula/dynamics/particle_process.py` — M4
- `particula/dynamics/tests/dilution_runnable_test.py` — M4
- `particula/dynamics/tests/dilution_test.py` — M4
- `particula/dynamics/tests/nucleation_runnable_test.py` — M4
- `particula/dynamics/tests/wall_loss_runnable_test.py` — M4
- `particula/equilibria/equilibria.py` — M4
- `particula/execution/__init__.py` — M4
- `particula/execution/adapters/coagulation.py` — M4
- `particula/execution/adapters/condensation.py` — M4
- `particula/execution/tests/coagulation_adapter_test.py` — M4
- `particula/execution/tests/coagulation_integration_test.py` — M4
- `particula/execution/tests/condensation_adapter_test.py` — M4
- `particula/execution/tests/condensation_integration_test.py` — M4
- `particula/integration_tests/charged_coagulation_comparison_test.py` — M4
- `particula/integration_tests/coagulation_integration_test.py` — M4
- `particula/integration_tests/condensation_latent_heat_conservation_test.py` — M4
- `particula/integration_tests/condensation_particle_resolved_test.py` — M4
- `particula/integration_tests/native_aerosol_construction_test.py` — M6 (legacy-only assertion removal)
- `particula/integration_tests/quick_start_test.py` — M5
- `particula/runnable.py` — M4
- `particula/tests/execution_test.py` — M4
- `particula/tests/legacy_aerosol_builder_test.py` — M6 (legacy-only assertion removal)
- `particula/tests/legacy_aerosol_test.py` — M6 (legacy-only assertion removal)
- `particula/tests/runnable_test.py` — M4
- `plugins/particula-helper/opencode/payload/scripts/run_example.py` — M5
- `plugins/particula-helper/scripts/run_example.py` — M5
- `readme.md` — M5
