"""
Prototype for generating and training a neural network to predict the
parameters of a 2-mode lognormal distribution from a concentration PDF.
"""

import os
import logging
import warnings
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import joblib  # type: ignore
from tqdm import tqdm


from scipy.interpolate import interp1d  # type: ignore
from sklearn.utils import shuffle  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from particula.next.particles.properties import lognormal_pdf_distribution
from particula.data.process.ml_analysis.get_ml_folder import (
    get_ml_analysis_folder,
)

# Set up logging
logger = logging.getLogger("particula")

# Suppress all warnings
warnings.filterwarnings("ignore")

# Training parameters
TOTAL_NUMBER_SIMULATED = 1_000_000
NUMBER_OF_MODES_SIM = 2
X_ARRAY_MAX_INDEX = 128
LOWER_BOUND_GSD = 1.0
UPPER_BOUND_GSD = 3.0
NOISE_FACTOR = 0.3  # must be between 0 < NOISE_FACTOR < 1
SAVE_NAME = "lognormal_2mode_NN128.pkl"

_cached_pipeline: Optional[Pipeline] = None


# pylint: disable=too-many-arguments
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
]:
    """
    Generate simulated lognormal aerosol particle size distributions.

    Arguments:
        total_number_simulated: Total number of simulated distributions.
        number_of_modes_sim: Number of modes to simulate (1, 2, or 3).
        x_array_max_index: Number of size bins in the particle size array.
        lower_bound_gsd: Lower bound for the geometric standard deviation.
        upper_bound_gsd: Upper bound for the geometric standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        x_values: Array of particle sizes.
        mode_index_sim: Array of simulated mode indices.
        geomertic_standard_deviation_sim: Array of simulated geometric
            standard deviations (GSDs).
        number_of_particles_sim: Array of simulated relative number
            concentrations.
        number_pdf_sim: Array of simulated probability density
            functions (PDFs).
    """
    np.random.seed(seed)

    # Define particle size array
    x_values = np.logspace(0, 3, x_array_max_index)

    # Generate random mode indices
    mode_index_sim = np.random.randint(  # type: ignore
        0, x_array_max_index - 1, [total_number_simulated, number_of_modes_sim]
    )
    mode_index_sim = np.sort(mode_index_sim, axis=1)
    mode_values_sim = [x_values[i] for i in mode_index_sim]

    # Generate geometric standard deviation and number of particles
    geomertic_standard_deviation_sim = np.random.uniform(
        lower_bound_gsd,
        upper_bound_gsd,
        [total_number_simulated, number_of_modes_sim],
    )
    geomertic_standard_deviation_sim = np.sort(
        geomertic_standard_deviation_sim, axis=1
    )

    number_of_particles_sim = np.random.uniform(
        0, 10, [total_number_simulated, number_of_modes_sim]
    )
    number_of_particles_sim /= number_of_particles_sim.sum(axis=1)[
        :, np.newaxis
    ]

    # Initialize array for the PDFs
    number_pdf_sim = np.zeros([total_number_simulated, x_values.shape[0]])

    # Generate the simulated data
    for i in range(total_number_simulated):
        number_pdf_sim[i] = lognormal_pdf_distribution(
            x_values=x_values,
            mode=mode_values_sim[i],
            geometric_standard_deviation=geomertic_standard_deviation_sim[i],
            number_of_particles=number_of_particles_sim[i],
        )

    return (
        x_values,
        mode_index_sim,
        geomertic_standard_deviation_sim,
        number_of_particles_sim,
        number_pdf_sim,
    )


def normalize_max(x_input: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalize each sample in X by dividing by its maximum value.

    Arguments:
        X: The input array to be normalized.

    Returns:
        The normalized array.
    """
    return x_input / x_input.max(axis=1, keepdims=True)


def normalize_targets(
    mode_index_sim: NDArray[np.int64],
    geomertic_standard_deviation_sim: NDArray[np.float64],
    number_of_particles_sim: NDArray[np.float64],
    x_array_max_index: int,
    lower_bound_gsd: float,
    upper_bound_gsd: float,
) -> NDArray[np.float64]:
    """
    Normalize the mode index, GSD, and relative number concentration.

    Arguments:
        mode_index_sim: Array of mode indices.
        geomertic_standard_deviation_sim: Array of geometric standard
            deviations (GSDs).
        number_of_particles_sim: Array of relative number concentrations.
        x_array_max_index: Maximum index for the mode.
        lower_bound_gsd: Lower bound for the geometric standard
            deviation (GSD).
        upper_bound_gsd: Upper bound for the geometric standard
            deviation (GSD).

    Returns:
        y: Normalized array combining mode indices, GSDs, and relative
            number concentrations.
    """
    y_mode = mode_index_sim / x_array_max_index
    y_gsd = (geomertic_standard_deviation_sim - lower_bound_gsd) / (
        upper_bound_gsd - lower_bound_gsd
    )
    y_rel_num = number_of_particles_sim
    y = np.hstack([y_mode, y_gsd, y_rel_num])  # Concatenate all labels
    return y


def create_pipeline() -> Pipeline:
    """
    Create a pipeline with normalization and MLPRegressor model.

    Returns:
        A scikit-learn Pipeline object.
    """
    return Pipeline(
        [
            (
                "normalize",
                FunctionTransformer(normalize_max, validate=False),
            ),  # Custom normalization
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(128, 128),
                    max_iter=1000,
                    activation="relu",
                    solver="adam",
                    random_state=42,
                ),
            ),
        ]
    )


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
]:
    """
    Train the pipeline and return the trained model along with train/test data.

    Arguments:
        X: The feature array.
        y: The target array.
        test_size: The proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.

    Returns:
        pipeline: The trained pipeline.
        X_train, X_test, y_train, y_test: The training and testing data splits.
    """
    x_train, x_test, y_train, y_test = train_test_split(  # type: ignore
        x_input, y, test_size=test_split_size, random_state=random_state
    )
    x_train = np.array(x_train, dtype=np.float64)
    x_test = np.array(x_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)

    pipeline = create_pipeline()
    pipeline.fit(x_train, y_train)  # type: ignore

    return pipeline, x_train, x_test, y_train, y_test


# pylint: disable=too-many-locals
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
]:
    """
    Train the pipeline in batches with progress tracking, and return the
    trained model along with train/test data.

    Arguments:
        X: The feature array.
        y: The target array.
        test_size: The proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
        n_batches: Number of batches to split the training into.

    Returns:
        pipeline: The trained pipeline.
        X_train, X_test, y_train, y_test: The training and testing data splits.
    """
    x_train, x_test, y_train, y_test = train_test_split(  # type: ignore
        x_input, y, test_size=test_split_size, random_state=random_state
    )

    pipeline = create_pipeline()

    # Split the data into batches
    n_samples = x_train.shape[0]  # type: ignore
    batch_size = int(n_samples // n_batches)  # type: ignore

    # Shuffle data
    x_train, y_train = shuffle(  # type: ignore
        x_train, y_train, random_state=random_state  # type: ignore
    )

    # Initialize the model with warm_start=True to allow incremental learning
    model = pipeline.named_steps["model"]  # type: ignore
    model.warm_start = True

    for i in tqdm(range(n_batches), desc="Training Progress"):
        start = i * batch_size
        end = (  # type: ignore
            (i + 1) * batch_size
            if (i + 1) * batch_size < n_samples
            else n_samples
        )
        end = int(end)  # type: ignore

        # Train the model incrementally on each batch
        pipeline.fit(x_train[start:end], y_train[start:end])  # type: ignore

    # coerce to float64
    x_train = np.array(x_train, dtype=np.float64)
    x_test = np.array(x_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)

    return pipeline, x_train, x_test, y_train, y_test


def evaluate_pipeline(
    pipeline: Pipeline,
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
) -> None:
    """
    Evaluate the pipeline and print the mean squared error for each target.

    Arguments:
        pipeline: The trained pipeline.
        X_test: The test feature array.
        y_test: The test target array.
    """
    y_pred = pipeline.predict(x_test)  # type: ignore
    mse = mean_squared_error(y_test, y_pred)  # type: ignore
    print(f"Overall Mean Squared Error: {mse}")

    number_of_modes_sim = y_test.shape[1] // 3
    for i, target_name in enumerate(
        ["Mode Index", "GSD", "Relative Number Concentration"]
    ):
        target_mse = mean_squared_error(  # type: ignore
            y_test[:, i::number_of_modes_sim],  # type: ignore
            y_pred[:, i::number_of_modes_sim],  # type: ignore
        )
        print(f"Mean Squared Error for {target_name}: {target_mse}")


def save_pipeline(pipeline: Pipeline, filename: str) -> None:
    """
    Save the trained pipeline to a file.

    Arguments:
        pipeline: The trained pipeline.
        filename: The filename to save the pipeline to.
    """
    joblib.dump(pipeline, filename)  # type: ignore


def load_pipeline(filename: str) -> Pipeline:
    """
    Load a pipeline from a file.

    Arguments:
        filename: The filename to load the pipeline from.

    Returns:
        The loaded pipeline.
    """
    return joblib.load(filename)  # type: ignore


def load_and_cache_pipeline(filename: str) -> Pipeline:
    """
    Load and cache the ML pipeline if not already loaded.

    Arguments:
        filename: Path to the pipeline file.

    Returns:
        The loaded pipeline.
    """
    global _cached_pipeline  # pylint: disable=global-statement

    if _cached_pipeline is None:
        _cached_pipeline = load_pipeline(filename=filename)

    return _cached_pipeline


def train_network_and_save():
    """Train the neural network and save the pipeline."""
    (
        _,
        mode_index_sim,
        geomertic_standard_deviation_sim,
        number_of_particles_sim,
        number_pdf_sim,
    ) = generate_simulated_data(
        total_number_simulated=TOTAL_NUMBER_SIMULATED,
        number_of_modes_sim=NUMBER_OF_MODES_SIM,
        x_array_max_index=X_ARRAY_MAX_INDEX,
        lower_bound_gsd=LOWER_BOUND_GSD,
        upper_bound_gsd=UPPER_BOUND_GSD,
        seed=0,
    )

    # Normalize targets
    y = normalize_targets(
        mode_index_sim=mode_index_sim,
        geomertic_standard_deviation_sim=geomertic_standard_deviation_sim,
        number_of_particles_sim=number_of_particles_sim,
        x_array_max_index=X_ARRAY_MAX_INDEX,
        lower_bound_gsd=LOWER_BOUND_GSD,
        upper_bound_gsd=UPPER_BOUND_GSD,
    )

    # Train the pipeline
    pipeline, _, x_test, _, y_test = train_pipeline(
        x_input=number_pdf_sim, y=y
    )

    # Evaluate the pipeline
    print("Evaluating pipeline:")
    evaluate_pipeline(pipeline, x_test, y_test)

    # Save the pipeline
    folder_path = get_ml_analysis_folder()
    save_path = os.path.join(folder_path, SAVE_NAME)
    save_pipeline(pipeline, save_path)

    # train with noise
    # % Add noise to the simulated data
    number_pdf_sim_noisy = number_pdf_sim * np.random.uniform(
        low=1 - NOISE_FACTOR,
        high=1 + NOISE_FACTOR,
        size=[number_pdf_sim.shape[0], number_pdf_sim.shape[1]],
    )
    # Train the pipeline with noisy data
    (
        pipeline_noisy,
        _,
        x_test_noisy,
        _,
        y_test_noisy,
    ) = train_pipeline(x_input=number_pdf_sim_noisy, y=y)

    # Evaluate the pipeline with noisy data
    print("Evaluating pipeline with noisy data:")
    evaluate_pipeline(pipeline_noisy, x_test_noisy, y_test_noisy)

    # Save the pipeline with noisy data
    save_pipeline(pipeline, save_path)


# pylint: disable=too-many-locals
def lognormal_2mode_ml_guess(
    logspace_x: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Load the machine learning pipeline, interpolate the concentration PDF,
    and predict lognormal parameters.

    Arguments:
        file_name: Path to the saved ML pipeline file.
        x_values: Array of x-values (particle sizes).
        concentration_pdf: Array of concentration PDF values.

    Returns:
        mode_values_guess: Predicted mode values after rescaling.
        geometric_standard_deviation_guess: Predicted geometric standard
            deviations after rescaling.
        number_of_particles_guess: Predicted number of particles after
            rescaling.
    """
    # Load the ML pipeline
    folder_path = get_ml_analysis_folder()
    load_path = os.path.join(folder_path, SAVE_NAME)
    ml_pipeline = load_and_cache_pipeline(filename=load_path)

    # Generate ml_x_values based on the range of x_values
    ml_x_values = np.logspace(
        start=np.log10(logspace_x.min()),
        stop=np.log10(logspace_x.max()),
        num=X_ARRAY_MAX_INDEX,
        dtype=np.float64,
    )

    # Interpolate the concentration to match ml_x_values
    interpolator = interp1d(
        logspace_x,
        concentration_pdf,
        kind="linear",
        fill_value="extrapolate",  # type: ignore
    )
    interpolated_concentration_pdf = interpolator(ml_x_values)  # type: ignore

    # Predict the concentration using the ML pipeline
    predicted_params = np.array(
        ml_pipeline.predict(  # type: ignore
            interpolated_concentration_pdf.reshape(1, -1)  # type: ignore
        ),
        dtype=np.float64,
    )

    # Extract predicted parameters
    mode_guess = predicted_params[0, 0:2]
    geometric_standard_deviation_guess = predicted_params[0, 2:4]
    number_of_particles_guess = predicted_params[0, 4:6]

    # Rescale mode values
    mode_index_guess = mode_guess * X_ARRAY_MAX_INDEX
    interp_mode = interp1d(
        np.arange(X_ARRAY_MAX_INDEX),
        ml_x_values,
        kind="linear",
        fill_value="extrapolate",  # type: ignore
    )
    mode_values_guess = np.array(
        interp_mode(mode_index_guess), dtype=np.float64
    )

    # Rescale geometric standard deviations
    geometric_standard_deviation_guess = (
        geometric_standard_deviation_guess
        * (UPPER_BOUND_GSD - LOWER_BOUND_GSD)
        + LOWER_BOUND_GSD
    )

    # Integrate the concentration to get the total number of particles
    total_particles = np.trapz(concentration_pdf, logspace_x)
    number_of_particles_guess = number_of_particles_guess * total_particles

    return (
        mode_values_guess,
        geometric_standard_deviation_guess,
        number_of_particles_guess,
    )


def looped_lognormal_2mode_ml_guess(
    logspace_x: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Loop through the concentration PDFs to get the best guess.

    Arguments:
        logspace_x: Array of x-values (particle sizes).
        concentration_pdf: Matrix of concentration PDF values.

    Returns:
        Tuple:
        - mode_values_guess: Predicted mode values after rescaling.
        - geometric_standard_deviation_guess: Predicted geometric standard
            deviations after rescaling.
        - number_of_particles_guess: Predicted number of particles after
            rescaling.
    """
    n_rows = concentration_pdf.shape[0]
    mode_values_guess = np.zeros([n_rows, 2], dtype=np.float64)
    geometric_standard_deviation_guess = np.zeros(
        [n_rows, 2], dtype=np.float64
    )
    number_of_particles_guess = np.zeros([n_rows, 2], dtype=np.float64)

    for row in range(n_rows):
        (
            mode_values_guess[row],
            geometric_standard_deviation_guess[row],
            number_of_particles_guess[row],
        ) = lognormal_2mode_ml_guess(
            logspace_x=logspace_x,
            concentration_pdf=concentration_pdf[row],
        )

    return (
        mode_values_guess,
        geometric_standard_deviation_guess,
        number_of_particles_guess,
    )
