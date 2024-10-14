# Generate And Train 2mode Sizer

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Data](../../index.md#data) / [Process](../index.md#process) / [Ml Analysis](./index.md#ml-analysis) / Generate And Train 2mode Sizer

> Auto-generated documentation for [particula.data.process.ml_analysis.generate_and_train_2mode_sizer](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py) module.

#### Attributes

- `logger` - Set up logging: logging.getLogger('particula')

- `TOTAL_NUMBER_SIMULATED` - Training parameters: 10000000


## create_pipeline

[Show source in generate_and_train_2mode_sizer.py:181](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L181)

Create a pipeline with normalization and MLPRegressor model.

#### Returns

A scikit-learn Pipeline object.

#### Signature

```python
def create_pipeline() -> Pipeline: ...
```



## evaluate_pipeline

[Show source in generate_and_train_2mode_sizer.py:316](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L316)

Evaluate the pipeline and print the mean squared error for each target.

#### Arguments

- `pipeline` - The trained pipeline.
- `X_test` - The test feature array.
- `y_test` - The test target array.

#### Signature

```python
def evaluate_pipeline(
    pipeline: Pipeline, x_test: NDArray[np.float64], y_test: NDArray[np.float64]
) -> None: ...
```



## generate_simulated_data

[Show source in generate_and_train_2mode_sizer.py:48](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L48)

Generate simulated lognormal aerosol particle size distributions.

#### Arguments

- `total_number_simulated` - Total number of simulated distributions.
- `number_of_modes_sim` - Number of modes to simulate (1, 2, or 3).
- `x_array_max_index` - Number of size bins in the particle size array.
- `lower_bound_gsd` - Lower bound for the geometric standard deviation.
- `upper_bound_gsd` - Upper bound for the geometric standard deviation.
- `seed` - Random seed for reproducibility.

#### Returns

- `x_values` - Array of particle sizes.
- `mode_index_sim` - Array of simulated mode indices.
- `geomertic_standard_deviation_sim` - Array of simulated geometric
    standard deviations (GSDs).
- `number_of_particles_sim` - Array of simulated relative number
    concentrations.
- `number_pdf_sim` - Array of simulated probability density
    functions (PDFs).

#### Signature

```python
def generate_simulated_data(
    total_number_simulated: int = 10000,
    number_of_modes_sim: int = 2,
    x_array_max_index: int = 128,
    lower_bound_gsd: float = 1.0,
    upper_bound_gsd: float = 2.0,
    seed: int = 0,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
```



## load_and_cache_pipeline

[Show source in generate_and_train_2mode_sizer.py:368](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L368)

Load and cache the ML pipeline if not already loaded.

#### Arguments

- `filename` - Path to the pipeline file.

#### Returns

The loaded pipeline.

#### Signature

```python
def load_and_cache_pipeline(filename: str) -> Pipeline: ...
```



## load_pipeline

[Show source in generate_and_train_2mode_sizer.py:355](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L355)

Load a pipeline from a file.

#### Arguments

- `filename` - The filename to load the pipeline from.

#### Returns

The loaded pipeline.

#### Signature

```python
def load_pipeline(filename: str) -> Pipeline: ...
```



## lognormal_2mode_ml_guess

[Show source in generate_and_train_2mode_sizer.py:455](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L455)

Load the machine learning pipeline, interpolate the concentration PDF,
and predict lognormal parameters.

#### Arguments

- `file_name` - Path to the saved ML pipeline file.
- `x_values` - Array of x-values (particle sizes).
- `concentration_pdf` - Array of concentration PDF values.

#### Returns

- `mode_values_guess` - Predicted mode values after rescaling.
- `geometric_standard_deviation_guess` - Predicted geometric standard
    deviations after rescaling.
- `number_of_particles_guess` - Predicted number of particles after
    rescaling.

#### Signature

```python
def lognormal_2mode_ml_guess(
    logspace_x: NDArray[np.float64], concentration_pdf: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
```



## looped_lognormal_2mode_ml_guess

[Show source in generate_and_train_2mode_sizer.py:540](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L540)

Loop through the concentration PDFs to get the best guess.

#### Arguments

- `logspace_x` - Array of x-values (particle sizes).
- `concentration_pdf` - Matrix of concentration PDF values.

#### Returns

Tuple:
- `-` *mode_values_guess* - Predicted mode values after rescaling.
- `-` *geometric_standard_deviation_guess* - Predicted geometric standard
    deviations after rescaling.
- `-` *number_of_particles_guess* - Predicted number of particles after
    rescaling.

#### Signature

```python
def looped_lognormal_2mode_ml_guess(
    logspace_x: NDArray[np.float64], concentration_pdf: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
```



## normalize_max

[Show source in generate_and_train_2mode_sizer.py:133](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L133)

Normalize each sample in X by dividing by its maximum value.

#### Arguments

- `X` - The input array to be normalized.

#### Returns

The normalized array.

#### Signature

```python
def normalize_max(x_input: NDArray[np.float64]) -> NDArray[np.float64]: ...
```



## normalize_targets

[Show source in generate_and_train_2mode_sizer.py:146](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L146)

Normalize the mode index, GSD, and relative number concentration.

#### Arguments

- `mode_index_sim` - Array of mode indices.
- `geomertic_standard_deviation_sim` - Array of geometric standard
    deviations (GSDs).
- `number_of_particles_sim` - Array of relative number concentrations.
- `x_array_max_index` - Maximum index for the mode.
- `lower_bound_gsd` - Lower bound for the geometric standard
    deviation (GSD).
- `upper_bound_gsd` - Upper bound for the geometric standard
    deviation (GSD).

#### Returns

- `y` - Normalized array combining mode indices, GSDs, and relative
    number concentrations.

#### Signature

```python
def normalize_targets(
    mode_index_sim: NDArray[np.int64],
    geomertic_standard_deviation_sim: NDArray[np.float64],
    number_of_particles_sim: NDArray[np.float64],
    x_array_max_index: int,
    lower_bound_gsd: float,
    upper_bound_gsd: float,
) -> NDArray[np.float64]: ...
```



## save_pipeline

[Show source in generate_and_train_2mode_sizer.py:344](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L344)

Save the trained pipeline to a file.

#### Arguments

- `pipeline` - The trained pipeline.
- `filename` - The filename to save the pipeline to.

#### Signature

```python
def save_pipeline(pipeline: Pipeline, filename: str) -> None: ...
```



## train_network_and_save

[Show source in generate_and_train_2mode_sizer.py:386](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L386)

Train the neural network and save the pipeline.

#### Signature

```python
def train_network_and_save(): ...
```



## train_pipeline

[Show source in generate_and_train_2mode_sizer.py:208](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L208)

Train the pipeline and return the trained model along with train/test data.

#### Arguments

- `X` - The feature array.
- `y` - The target array.
- `test_size` - The proportion of the dataset to include in the test split.
- `random_state` - Random seed for reproducibility.

#### Returns

- `pipeline` - The trained pipeline.
X_train, X_test, y_train, y_test: The training and testing data splits.

#### Signature

```python
def train_pipeline(
    x_input: NDArray[np.float64],
    y: NDArray[np.float64],
    test_split_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[
    Pipeline,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
```



## train_pipeline_with_progress

[Show source in generate_and_train_2mode_sizer.py:248](https://github.com/uncscode/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L248)

Train the pipeline in batches with progress tracking, and return the
trained model along with train/test data.

#### Arguments

- `X` - The feature array.
- `y` - The target array.
- `test_size` - The proportion of the dataset to include in the test split.
- `random_state` - Random seed for reproducibility.
- `n_batches` - Number of batches to split the training into.

#### Returns

- `pipeline` - The trained pipeline.
X_train, X_test, y_train, y_test: The training and testing data splits.

#### Signature

```python
def train_pipeline_with_progress(
    x_input: NDArray[np.float64],
    y: NDArray[np.float64],
    test_split_size: float = 0.3,
    random_state: int = 42,
    n_batches: int = 10,
) -> Tuple[
    Pipeline,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
```
