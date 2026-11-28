# Beta source index

Inventory of every tracked file under `particula_beta/` at the pinned donor revision: 81 runtime/package files and 90 tests, fixtures and support files (171 total). Top-level analysis campaigns, generated outputs, the manuscript and development tooling sit outside the importable package. The beta [license](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/license) and [dependency manifest](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/pyproject.toml) apply when selecting source for a port.


## particula_beta

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/__init__.py) | Package surface | Define focused exports in the new owning module when its first API ships. |

## particula_beta/analysis/multi_instrument

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/analysis/multi_instrument/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/analysis/multi_instrument/__init__.py) | Package surface | Define focused exports in the new owning module when its first API ships. |
| [particula_beta/analysis/multi_instrument/combined_optimizer.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/analysis/multi_instrument/combined_optimizer.py) | `create_combined_cost_function`, `MultiInstrumentOptimizer` | Study retrieval; revisit joint objectives after two qualified responses. |

## particula_beta/data

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/data/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/__init__.py) | Package surface | Define focused exports in the new owning module when its first API ships. |
| [particula_beta/data/lake.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/lake.py) | `Lake` | Use a mapping of named measurements with independent clocks. |
| [particula_beta/data/lake_stats.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/lake_stats.py) | `get_lake_average_std` | Apply data window utilities per measurement, preserving coverage. |
| [particula_beta/data/loader.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/loader.py) | `data_raw_loader`, `filter_list` | Port text, clock and sizer formatters to data.readers; split persistence and ARM decoding. |
| [particula_beta/data/loader_interface.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/loader_interface.py) | `get_new_files`, `load_files_interface` | Replace Stream construction with data.read; retain named profiles and file provenance. |
| [particula_beta/data/loader_setting_builders.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/loader_setting_builders.py) | `Loader1DSettingsBuilder`, `DataChecksBuilder` | Translate supported settings into versioned profiles; retain configuration test cases. |
| [particula_beta/data/measurement_data.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/measurement_data.py) | `BulkWithSize` | Extract measured quantities and axes; keep assumed species properties in explicit reconstruction. |
| [particula_beta/data/merger.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/merger.py) | `combine_data`, `stream_add_data` | Port explicit append, align and duplicate handling to data.operations; expose gap and tail policies. |
| [particula_beta/data/mixin.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/mixin.py) | `RelativeFolderMixin`, `FilenameRegexMixin` | Replace builder property setters with schema constraints and field diagnostics. |

## particula_beta/data/process

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/data/process/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/__init__.py) | Package surface | Define focused exports in the new owning module when its first API ships. |
| [particula_beta/data/process/aerodynamic_convert.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/aerodynamic_convert.py) | `convert_aerodynamic_to_physical_radius`, `convert_physical_to_aerodynamic_radius` | Assess iterative radius conversion against native aerodynamic size properties before combining APS/SMPS. |
| [particula_beta/data/process/aerosol_state_generator.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/aerosol_state_generator.py) | `AerosolStateGenerator` | Study-side reconstruction; qualify explicit native-state adapter separately. |
| [particula_beta/data/process/bimodal_generator.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/bimodal_generator.py) | `BimodalAerosolGenerator` | Keep synthetic populations in examples; reuse native distribution builders. |
| [particula_beta/data/process/chamber_rate_fitting.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/chamber_rate_fitting.py) | `chi_squared_error`, `hessian_standard_error` | Keep chamber-specific inverse fits and uncertainty evaluation in studies. |
| [particula_beta/data/process/composition_generator.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/composition_generator.py) | `CompositionGenerator` | Keep species allocation assumptions explicit in study reconstruction. |

## particula_beta/data/process/database_lookup

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/data/process/database_lookup/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/database_lookup/__init__.py) | Package surface | Keep legacy namespace in donor alongside the deferred implementation. |
| [particula_beta/data/process/database_lookup/mie_database_builder.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/database_lookup/mie_database_builder.py) | `get_m_real_grid`, `get_m_imag_grid` | Retain lookup table generation as historical benchmark; native optics owns production physics. |
| [particula_beta/data/process/database_lookup/mie_database_lookup.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/database_lookup/mie_database_lookup.py) | `get_database_status`, `lookup_mie_parameters` | Retain lookup accuracy cases for comparison; avoid an import-time database. |

## particula_beta/data/process

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/data/process/distribution_utilities.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/distribution_utilities.py) | `generate_multimodal_distribution`, `get_radius_via_masses` | Compare moments and quadratures to native spectra; use existing particle properties for mass and radius. |
| [particula_beta/data/process/kappa_via_extinction.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/kappa_via_extinction.py) | `extinction_ratio_wet_dry`, `fit_extinction_ratio_with_kappa` | Defer inversion to a validated humidified CAPS example with uncertainty. |
| [particula_beta/data/process/landscape_analysis.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/landscape_analysis.py) | `setup_landscape_logging`, `run_parameter_sweep` | Keep sweep orchestration, CSV reporting and plotting in studies. |
| [particula_beta/data/process/lognormal_2mode.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/lognormal_2mode.py) | `cost_function`, `get_bounds` | Keep fitting in analysis examples until a second validated caller needs a shared fit. |
| [particula_beta/data/process/mass_adjustment.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/mass_adjustment.py) | `get_adjusted_masses` | Keep mass-prior adjustment in reconstruction; test budgets if promoted. |
| [particula_beta/data/process/mie_angular.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/mie_angular.py) | `discretize_scattering_angles`, `calculate_scattering_angles` | Reuse native differential cross sections and angular optics. |
| [particula_beta/data/process/mie_bulk.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/mie_bulk.py) | `discretize_auto_mieq`, `discretize_mie_parameters` | Reuse native particle/population optics; document SI conversions at instrument boundary. |
| [particula_beta/data/process/mixing_state.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/mixing_state.py) | `get_set_mixing_state`, `get_volume_conserved_mixing_state_masses` | Keep mass redistribution in physical reconstruction; verify species and volume closure separately. |

## particula_beta/data/process/ml_analysis

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/data/process/ml_analysis/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/ml_analysis/__init__.py) | Package surface | Keep legacy namespace in donor alongside the deferred implementation. |
| [particula_beta/data/process/ml_analysis/generate_and_train_2mode_sizer.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/ml_analysis/generate_and_train_2mode_sizer.py) | `generate_simulated_data`, `normalize_max` | Keep synthetic-data training pipeline in research archive. |
| [particula_beta/data/process/ml_analysis/get_ml_folder.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/ml_analysis/get_ml_folder.py) | `get_ml_analysis_folder` | Retire model-path lookup with the legacy trained-model workflow. |
| [particula_beta/data/process/ml_analysis/lognormal_2mode_NN128.pkl](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/ml_analysis/lognormal_2mode_NN128.pkl) | Serialized fitted model | Leave serialized training artifact in donor; preserve provenance for research comparison. |
| [particula_beta/data/process/ml_analysis/run_ml_trainings.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/ml_analysis/run_ml_trainings.py) | Training entry point | Leave training entry point with model and its environment. |

## particula_beta/data/process

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/data/process/optical_instrument.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/optical_instrument.py) | `CapsInstrumentKeywordBuilder`, `caps_processing` | Port CAPS observables and RH handling to instruments.optical_extinction_scattering on native optics. |
| [particula_beta/data/process/sample_distribution.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/sample_distribution.py) | `get_samples_from_tabulated_pdf` | Use as a study-side sampling reference; require seeded distribution closure for promotion. |
| [particula_beta/data/process/scattering_truncation.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/scattering_truncation.py) | `get_truncated_scattering`, `trunc_mono` | Reimplement CAPS geometry/integration and calibration on native angular optics. |
| [particula_beta/data/process/size_distribution.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/size_distribution.py) | `mean_properties`, `sizer_mean_properties` | Port justified moments; use existing conservative rebinning; treat instrument merging as inference. |
| [particula_beta/data/process/stats.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/stats.py) | `merge_formatting`, `average_to_interval` | Port interval and QC semantics; use native bin integration and explicit gap policy. |

## particula_beta/data

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/data/reference_data.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/reference_data.py) | Material property tables | Keep conditional material constants with cited study/physics owners; exclude from measured defaults. |
| [particula_beta/data/settings_generator.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/settings_generator.py) | `for_general_1d_load`, `for_general_sizer_1d_2d_load` | Migrate settings templates and source references to data profile examples and legacy translation. |
| [particula_beta/data/stream.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/stream.py) | `Stream`, `StreamAveraged` | Replace float matrix and string headers with typed Polars columns, grids and provenance. |
| [particula_beta/data/stream_stats.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/stream_stats.py) | `drop_masked`, `average_std` | Port interval selection, validity and averaging semantics to data.operations; qualify derivatives separately. |

## particula_beta/data/taichi

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/data/taichi/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/taichi/__init__.py) | Package surface | Keep legacy namespace in donor alongside the deferred implementation. |
| [particula_beta/data/taichi/taichi_bessel.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/taichi/taichi_bessel.py) | `c_add`, `c_mul` | Retain as numerical comparison only; Particula GPU physics already uses Warp. |
| [particula_beta/data/taichi/taichi_mie.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/taichi/taichi_mie.py) | `compute_mie_efficiencies` | Reuse native Warp Mie support instead of shipping another GPU backend. |

## particula_beta/data/util

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/data/util/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/util/__init__.py) | Package surface | Define focused exports in the new owning module when its first API ships. |
| [particula_beta/data/util/convert_length.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/util/convert_length.py) | `get_length_from_volume`, `get_volume_from_length` | Review radius versus diameter conventions and reuse existing particle-property geometry. |

## particula_beta

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/exceptions.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/exceptions.py) | `ParticulaBetaError`, `LoaderError` | Define narrow profile/source/record diagnostics at the new data boundary. |

## particula_beta/instruments

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/instruments/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/__init__.py) | Package surface | Define focused exports in the new owning module when its first API ships. |
| [particula_beta/instruments/ccnc.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/ccnc.py) | `digital_twin_ccnc`, `ccnc_objective_alphafit` | Rebuild conditional growth and histogram in instruments.cloud_condensation_nuclei_counter after native-state and activation checks. |

## particula_beta/instruments/ccnc

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/instruments/ccnc/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/ccnc/__init__.py) | Package surface | Remove the dynamic sibling-file loading when reorganizing CCNC. |
| [particula_beta/instruments/ccnc/simulator.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/ccnc/simulator.py) | `CCNCSimulator` | Rebuild conditions on native state; verify that each advertised parameter affects the predicted observable. |

## particula_beta/instruments

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/instruments/ccnc_data_generator.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/ccnc_data_generator.py) | `generate_synthetic_aerosol` | Use seeded synthetic cases for CCNC tests after checking units and priors. |
| [particula_beta/instruments/ccnc_fitting.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/ccnc_fitting.py) | `ccnc_fitting_loop` | Keep campaign inversion in studies with independently tested likelihood and weights. |

## particula_beta/instruments/helpers

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/instruments/helpers/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/__init__.py) | Package surface | Define focused exports in the new owning module when its first API ships. |
| [particula_beta/instruments/helpers/base_optimizer.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/base_optimizer.py) | `BaseOptimizer` | Defer shared optimizer API until validated use in two instruments. |
| [particula_beta/instruments/helpers/base_simulator.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/base_simulator.py) | `BaseInstrumentSimulator` | Use domain-specific forward functions first; revisit shared signatures when needed. |
| [particula_beta/instruments/helpers/cost_functions.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/cost_functions.py) | `CostFunction`, `ChiSquaredCost` | Keep instrument error and weighting assumptions with each retrieval. |
| [particula_beta/instruments/helpers/noise.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/noise.py) | `add_measurement_noise` | Reuse seeded noise models in fixtures after checking measurement units and variance law. |
| [particula_beta/instruments/helpers/optimization.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/optimization.py) | Training entry point | Retain fit validation as study reference; keep losses away from parsing. |
| [particula_beta/instruments/helpers/optimization_types.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/optimization_types.py) | `OptimizationResult` | Revisit result records when retrievals have repeatable common outputs. |
| [particula_beta/instruments/helpers/statistics.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/statistics.py) | `get_mean_std` | Check zero-count and uncertainty conventions before porting to data statistics. |

## particula_beta/instruments

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/instruments/light_scattering_instrument.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/light_scattering_instrument.py) | `get_size_parameter`, `get_refractive_index` | Route refractive-index and cross-section calculations through native optics. |

## particula_beta/instruments/optical

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/instruments/optical/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/optical/__init__.py) | Package surface | Define focused exports in the new owning module when its first API ships. |
| [particula_beta/instruments/optical/optimizer.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/optical/optimizer.py) | `OpticalOptimizer`, `create_optical_cost_function` | Keep optical-specific inverse fitting in an example initially. |
| [particula_beta/instruments/optical/simulator.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/optical/simulator.py) | `OpticalSimulator` | Rebuild wavelength/RH response on native optics and test multi-wavelength outputs. |

## particula_beta/instruments

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/instruments/scattering_data_generator.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/scattering_data_generator.py) | `generate_monodisperse_distribution`, `generate_lognormal_distribution` | Use mono/lognormal controls as optical response test inputs. |
| [particula_beta/instruments/water_uptake.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/water_uptake.py) | `get_gas_phase_water_concentration`, `validate_rh_conditions` | Compare humidification conditions to native condensation and optical humidity models. |

## particula_beta/optimization

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/optimization/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/__init__.py) | Package surface | Keep legacy namespace in donor alongside the deferred implementation. |
| [particula_beta/optimization/benchmarking.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/benchmarking.py) | `PerformanceBenchmark` | Keep timing in study or performance harnesses. |
| [particula_beta/optimization/cost_functions.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/cost_functions.py) | `chi_squared`, `mean_squared_error` | Leave objectives with retrieval studies pending a shared likelihood contract. |
| [particula_beta/optimization/explorers.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/explorers.py) | `ParameterExplorer`, `AdaptiveSampler` | Keep parameter-search strategies in research tooling. |
| [particula_beta/optimization/identifiability_visualizers.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/identifiability_visualizers.py) | `plot_identifiability_vs_mass`, `plot_mass_comparison_grid` | Keep identifiability figures in study reporting. |
| [particula_beta/optimization/parallel_backends.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/parallel_backends.py) | `create_balanced_chunks`, `BaseParallelBackend` | Keep Dask/multiprocessing execution outside measurement dependencies. |
| [particula_beta/optimization/parallel_evaluator.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/parallel_evaluator.py) | `ParallelGridEvaluator` | Keep grid evaluation and caching with retrieval analysis. |
| [particula_beta/optimization/report_templates.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/report_templates.py) | `generate_mass_sweep_report`, `markdown_header` | Keep study-specific reporting outside data and instruments. |
| [particula_beta/optimization/slice_visualizers.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/slice_visualizers.py) | `plot_2d_slices_at_fixed_alpha`, `plot_alpha_effect_summary` | Keep parameter slice and isosurface figures in study plotting. |
| [particula_beta/optimization/visualizers.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/visualizers.py) | `StaticVisualizer`, `InteractiveVisualizer` | Keep plotting/analysis extras in study examples. |

## particula_beta

| Donor file | Representative contents | Proposed use |
| --- | --- | --- |
| [particula_beta/time_manage.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/time_manage.py) | `time_str_to_epoch`, `datetime64_from_epoch_array` | Port clock parsing and relative-time helpers to data.time, retaining precision and source zone. |
| [particula_beta/units.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/units.py) | `convert_units` | Reuse particula.util.convert_units; add quantity and basis checks in data profile validation. |

## Tests, fixture data and support files

These links identify the available evidence; a donor test becomes a Particula acceptance test only after its assumptions and expected values have been checked. Use CPC/SMPS files and loader tests for the first port, APS for the open channel, and optics/CCNC tests as comparison cases for later response work.


### particula_beta/analysis/multi_instrument/tests

- [particula_beta/analysis/multi_instrument/tests/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/analysis/multi_instrument/tests/__init__.py)
- [particula_beta/analysis/multi_instrument/tests/combined_optimizer_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/analysis/multi_instrument/tests/combined_optimizer_test.py)
- [particula_beta/analysis/multi_instrument/tests/fixtures.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/analysis/multi_instrument/tests/fixtures.py)

### particula_beta/data/process/database_lookup/tests

- [particula_beta/data/process/database_lookup/tests/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/database_lookup/tests/__init__.py)
- [particula_beta/data/process/database_lookup/tests/mie_database_builder_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/database_lookup/tests/mie_database_builder_test.py)
- [particula_beta/data/process/database_lookup/tests/mie_database_lookup_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/database_lookup/tests/mie_database_lookup_test.py)

### particula_beta/data/process/tests

- [particula_beta/data/process/tests/aerodynamic_convert_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/aerodynamic_convert_test.py)
- [particula_beta/data/process/tests/aerosol_state_generator_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/aerosol_state_generator_test.py)
- [particula_beta/data/process/tests/bimodal_generator_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/bimodal_generator_test.py)
- [particula_beta/data/process/tests/composition_generator_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/composition_generator_test.py)
- [particula_beta/data/process/tests/distribution_utilities_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/distribution_utilities_test.py)
- [particula_beta/data/process/tests/kappa_via_extinction_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/kappa_via_extinction_test.py)
- [particula_beta/data/process/tests/landscape_analysis_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/landscape_analysis_test.py)
- [particula_beta/data/process/tests/lognormal_2mode_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/lognormal_2mode_test.py)
- [particula_beta/data/process/tests/mass_adjustment_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/mass_adjustment_test.py)
- [particula_beta/data/process/tests/mie_angular_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/mie_angular_test.py)
- [particula_beta/data/process/tests/mie_bulk_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/mie_bulk_test.py)
- [particula_beta/data/process/tests/mixing_state_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/mixing_state_test.py)
- [particula_beta/data/process/tests/optical_instrument_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/optical_instrument_test.py)
- [particula_beta/data/process/tests/scattering_truncation_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/scattering_truncation_test.py)
- [particula_beta/data/process/tests/stats_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/process/tests/stats_test.py)

### particula_beta/data/taichi/tests

- [particula_beta/data/taichi/tests/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/taichi/tests/__init__.py)
- [particula_beta/data/taichi/tests/taichi_bessel_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/taichi/tests/taichi_bessel_test.py)
- [particula_beta/data/taichi/tests/taichi_mie_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/taichi/tests/taichi_mie_test.py)

### particula_beta/data/tests/example_data/APS_data

- [particula_beta/data/tests/example_data/APS_data/APS-20220707.txt](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/APS_data/APS-20220707.txt)
- [particula_beta/data/tests/example_data/APS_data/APS-20220710.txt](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/APS_data/APS-20220710.txt)

### particula_beta/data/tests/example_data/CPC_3010_data

- [particula_beta/data/tests/example_data/CPC_3010_data/CPC_3010_data_20220709_Jul.csv](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/CPC_3010_data/CPC_3010_data_20220709_Jul.csv)
- [particula_beta/data/tests/example_data/CPC_3010_data/CPC_3010_data_20220710_Jul.csv](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/CPC_3010_data/CPC_3010_data_20220710_Jul.csv)
- [particula_beta/data/tests/example_data/CPC_3010_data/stream_settings_cpc.json](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/CPC_3010_data/stream_settings_cpc.json)

### particula_beta/data/tests/example_data/SMPS_data

- [particula_beta/data/tests/example_data/SMPS_data/2022-07-07_095151_SMPS.csv](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/SMPS_data/2022-07-07_095151_SMPS.csv)
- [particula_beta/data/tests/example_data/SMPS_data/2022-07-10_094659_SMPS.csv](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/SMPS_data/2022-07-10_094659_SMPS.csv)
- [particula_beta/data/tests/example_data/SMPS_data/stream_settings_smps_1d.json](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/SMPS_data/stream_settings_smps_1d.json)
- [particula_beta/data/tests/example_data/SMPS_data/stream_settings_smps_2d.json](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/SMPS_data/stream_settings_smps_2d.json)

### particula_beta/data/tests/example_data/chamber_data

- [particula_beta/data/tests/example_data/chamber_data/2023-09-25_160155_SMPS.csv](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/chamber_data/2023-09-25_160155_SMPS.csv)
- [particula_beta/data/tests/example_data/chamber_data/stream_settings_smps_1d.json](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/chamber_data/stream_settings_smps_1d.json)
- [particula_beta/data/tests/example_data/chamber_data/stream_settings_smps_2d.json](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/chamber_data/stream_settings_smps_2d.json)

### particula_beta/data/tests/example_data

- [particula_beta/data/tests/example_data/get_example_data.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/get_example_data.py)
- [particula_beta/data/tests/example_data/lake_settings_cpc_smps.json](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/example_data/lake_settings_cpc_smps.json)

### particula_beta/data/tests

- [particula_beta/data/tests/exceptions_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/exceptions_test.py)
- [particula_beta/data/tests/init_exports_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/init_exports_test.py)
- [particula_beta/data/tests/lake_csv_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/lake_csv_test.py)
- [particula_beta/data/tests/lake_stats_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/lake_stats_test.py)
- [particula_beta/data/tests/lake_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/lake_test.py)
- [particula_beta/data/tests/loader_interface_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/loader_interface_test.py)
- [particula_beta/data/tests/loader_setting_builders_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/loader_setting_builders_test.py)
- [particula_beta/data/tests/loader_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/loader_test.py)
- [particula_beta/data/tests/measurement_data_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/measurement_data_test.py)
- [particula_beta/data/tests/merger_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/merger_test.py)
- [particula_beta/data/tests/mixin_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/mixin_test.py)
- [particula_beta/data/tests/reference_data_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/reference_data_test.py)
- [particula_beta/data/tests/settings_generator_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/settings_generator_test.py)
- [particula_beta/data/tests/stream_stats_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/stream_stats_test.py)
- [particula_beta/data/tests/stream_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/data/tests/stream_test.py)

### particula_beta/instruments/ccnc/tests

- [particula_beta/instruments/ccnc/tests/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/ccnc/tests/__init__.py)
- [particula_beta/instruments/ccnc/tests/equivalence_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/ccnc/tests/equivalence_test.py)
- [particula_beta/instruments/ccnc/tests/fixtures.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/ccnc/tests/fixtures.py)
- [particula_beta/instruments/ccnc/tests/simulator_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/ccnc/tests/simulator_test.py)

### particula_beta/instruments/helpers/tests

- [particula_beta/instruments/helpers/tests/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/tests/__init__.py)
- [particula_beta/instruments/helpers/tests/base_optimizer_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/tests/base_optimizer_test.py)
- [particula_beta/instruments/helpers/tests/base_simulator_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/tests/base_simulator_test.py)
- [particula_beta/instruments/helpers/tests/cost_functions_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/tests/cost_functions_test.py)
- [particula_beta/instruments/helpers/tests/noise_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/tests/noise_test.py)
- [particula_beta/instruments/helpers/tests/optimization_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/tests/optimization_test.py)
- [particula_beta/instruments/helpers/tests/optimization_types_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/tests/optimization_types_test.py)
- [particula_beta/instruments/helpers/tests/statistics_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/helpers/tests/statistics_test.py)

### particula_beta/instruments/optical/tests

- [particula_beta/instruments/optical/tests/__init__.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/optical/tests/__init__.py)
- [particula_beta/instruments/optical/tests/integration_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/optical/tests/integration_test.py)
- [particula_beta/instruments/optical/tests/optimizer_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/optical/tests/optimizer_test.py)
- [particula_beta/instruments/optical/tests/simulator_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/optical/tests/simulator_test.py)

### particula_beta/instruments/tests

- [particula_beta/instruments/tests/ccnc_data_generator_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/tests/ccnc_data_generator_test.py)
- [particula_beta/instruments/tests/ccnc_fitting_integration_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/tests/ccnc_fitting_integration_test.py)
- [particula_beta/instruments/tests/ccnc_fitting_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/tests/ccnc_fitting_test.py)
- [particula_beta/instruments/tests/ccnc_massfrac_refactor_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/tests/ccnc_massfrac_refactor_test.py)
- [particula_beta/instruments/tests/ccnc_objective_refactor_regression_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/tests/ccnc_objective_refactor_regression_test.py)
- [particula_beta/instruments/tests/ccnc_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/tests/ccnc_test.py)
- [particula_beta/instruments/tests/light_scattering_instrument_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/tests/light_scattering_instrument_test.py)
- [particula_beta/instruments/tests/scattering_data_generator_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/tests/scattering_data_generator_test.py)
- [particula_beta/instruments/tests/water_uptake_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/instruments/tests/water_uptake_test.py)

### particula_beta/optimization/tests

- [particula_beta/optimization/tests/cost_functions_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/tests/cost_functions_test.py)
- [particula_beta/optimization/tests/explorers_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/tests/explorers_test.py)
- [particula_beta/optimization/tests/fixtures.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/tests/fixtures.py)
- [particula_beta/optimization/tests/identifiability_visualizers_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/tests/identifiability_visualizers_test.py)
- [particula_beta/optimization/tests/numerical_stability_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/tests/numerical_stability_test.py)
- [particula_beta/optimization/tests/parallel_backends_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/tests/parallel_backends_test.py)
- [particula_beta/optimization/tests/parallel_performance_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/tests/parallel_performance_test.py)
- [particula_beta/optimization/tests/property_based_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/tests/property_based_test.py)
- [particula_beta/optimization/tests/report_templates_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/tests/report_templates_test.py)
- [particula_beta/optimization/tests/slice_visualizers_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/tests/slice_visualizers_test.py)
- [particula_beta/optimization/tests/visualizers_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/optimization/tests/visualizers_test.py)

### particula_beta/tests/integration

- [particula_beta/tests/integration/06_ccnc_optimizer_compare_test.py](https://github.com/Gorkowski/particula-dev-data/blob/4e56f7da0d29b6c5cb08d8dcc36cf40c990ef8b8/particula_beta/tests/integration/06_ccnc_optimizer_compare_test.py)
