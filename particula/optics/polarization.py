"""Scattering-plane Mueller matrices and Stokes operations.

Use (parallel, perpendicular) electric-field components with
Q=|Eparallel|²-|Eperpendicular|², U=2 Re(Eparallel Eperpendicular*), and
V=-2 Im(Eparallel Eperpendicular*). Sphere scattering has Jones diagonal
(S2, S1). Equations follow Bohren and Huffman (1983), chapter 3,
https://doi.org/10.1002/9783527618156.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def get_mueller_matrix(
    perpendicular: ArrayLike, parallel: ArrayLike
) -> NDArray[np.float64]:
    """Construct the unnormalized sphere Mueller matrix from S1 and S2.

    Args:
        perpendicular: Complex perpendicular amplitude S1, dimensionless.
        parallel: Complex parallel amplitude S2, broadcastable with S1.

    Returns:
        Real matrix with broadcast shape plus (4, 4), in the scattering
        plane. Divide by medium wavenumber squared for m²/sr.

    Raises:
        ValueError: If either amplitude is nonfinite.
    """
    s1, s2 = np.broadcast_arrays(
        np.asarray(perpendicular, dtype=np.complex128),
        np.asarray(parallel, dtype=np.complex128),
    )
    if np.any(~np.isfinite(s1)) or np.any(~np.isfinite(s2)):
        raise ValueError("Scattering amplitudes must be finite")
    matrix = np.zeros(s1.shape + (4, 4), dtype=np.float64)
    s11 = (abs(s2) ** 2 + abs(s1) ** 2) / 2
    s12 = (abs(s2) ** 2 - abs(s1) ** 2) / 2
    product = s2 * s1.conj()
    matrix[..., 0, 0] = matrix[..., 1, 1] = s11
    matrix[..., 0, 1] = matrix[..., 1, 0] = s12
    matrix[..., 2, 2] = matrix[..., 3, 3] = product.real
    matrix[..., 2, 3] = product.imag
    matrix[..., 3, 2] = -product.imag
    return matrix


def rotate_stokes(stokes: ArrayLike, angle: ArrayLike) -> NDArray[np.float64]:
    """Express a Stokes vector in axes rotated toward the perpendicular axis.

    Args:
        stokes: Finite (..., 4) array ordered I,Q,U,V, in caller units.
        angle: Rotation [rad], broadcastable with the leading Stokes shape.
            Positive rotation uses Eparallel'=cos(angle)*Eparallel +
            sin(angle)*Eperpendicular.

    Returns:
        New vectors with Q'=cos(2a)Q+sin(2a)U, U'=-sin(2a)Q+cos(2a)U.
        I and V are unchanged. This operation does not normalize intensity.

    Raises:
        ValueError: If shapes or finite values are invalid.
    """
    vector = np.asarray(stokes, dtype=np.float64)
    theta = np.asarray(angle, dtype=np.float64)
    if vector.ndim < 1 or vector.shape[-1] != 4:
        raise ValueError("Stokes vectors require a final axis of length four")
    if np.any(~np.isfinite(vector)) or np.any(~np.isfinite(theta)):
        raise ValueError("Stokes vectors and rotation angles must be finite")
    shape = np.broadcast_shapes(vector.shape[:-1], theta.shape)
    result = np.broadcast_to(vector, shape + (4,)).copy()
    cosine, sine = np.cos(2 * theta), np.sin(2 * theta)
    result[..., 1] = cosine * vector[..., 1] + sine * vector[..., 2]
    result[..., 2] = -sine * vector[..., 1] + cosine * vector[..., 2]
    return result


def apply_mueller_matrix(
    matrix: ArrayLike,
    stokes: ArrayLike,
    input_rotation: ArrayLike = 0.0,
    output_rotation: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    """Apply R(output_rotation) M R(input_rotation) to a Stokes vector.

    Args:
        matrix: Finite (..., 4, 4) Mueller array; units set output units.
        stokes: Finite (..., 4) I,Q,U,V input, broadcastable with matrix.
        input_rotation: Rotation [rad] into the matrix's incident basis.
        output_rotation: Rotation [rad] from its outgoing basis.

    Returns:
        New output Stokes vectors, without intensity normalization or
        propagation-distance factors. Signed linear combinations are allowed.

    Raises:
        ValueError: If matrix/vector shapes or finite values are invalid.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim < 2 or array.shape[-2:] != (4, 4):
        raise ValueError("Mueller matrices require final axes (4, 4)")
    if np.any(~np.isfinite(array)):
        raise ValueError("Mueller matrices must be finite")
    incident = rotate_stokes(stokes, input_rotation)
    outgoing = np.matmul(array, incident[..., None])[..., 0]
    return rotate_stokes(outgoing, output_rotation)
