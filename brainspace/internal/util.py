from __future__ import annotations

import numpy as np
from typing import Callable, Tuple

class ParameterizedCurve:
    """
    Represents a parameterized 1D curve in input space.
    Maps a scalar parameter t to a point in d-dimensional space.
    """
    def __init__(self, function: Callable[[np.ndarray], np.ndarray]):
        """
        Args:
            function: a callable that takes a scalar or array of t values and returns corresponding points in d-dimensional space
        """
        self.function = function


    @staticmethod
    def axis_curve(input_dim, axis) -> ParameterizedCurve:
        """
        Create a line aligned with a single axis (e.g., varying along x_1).

        Args:
            input_dim: total dimensionality
            axis: which axis to vary (0-indexed)

        Returns:
            ParameterizedCurve varying along the specified axis
        """
        func = lambda t: np.eye(input_dim, 1)[axis] * t
        return ParameterizedCurve(func)


    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the line at parameter value(s) t.

        Args:
            t: scalar or array of parameter values

        Returns:
            Point(s) in d-dimensional space: start + t * direction
        """
        t = np.asarray(t)
        return self.function(t)


    def generate_curve(self, t_range: Tuple[float, float], n_points: int) -> np.ndarray:
        """
        Generate points along the line for visualization.

        Args:
            t_range: (t_min, t_max) range of parameter values
            n_points: number of points to generate

        Returns:
            Array of shape (n_points, input_dim) with points along the line
        """
        t_values = np.linspace(t_range[0], t_range[1], n_points)
        return self(t_values)

def kabsch_procrustes(A, B):
    """
    Implements the Kabsch Algorithm to align A to B via a pure rotation (no scaling or translation)

    Args:
        A (np.ndarray) (M, N): Matrix to be mapped.
        B (np.ndarray) (M, N): Target matrix.

    Returns:
        R (np.ndarray) (N, N): The optimal rotation matrix that minimizes the Frobenius norm of (A @ R) - B.
    """
    # 1. Compute the cross-covariance matrix between the two states
    H = A.T @ B

    # 2. Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # 3. Calculate the standard orthogonal matrix
    R = U @ Vt

    # 4. THE KABSCH CORRECTION: Check the determinant
    # A determinant of +1 means a pure rotation.
    # A determinant of -1 means a reflection (the cause of your jumping).
    if np.linalg.det(R) < 0:
        # If it's a reflection, flip the sign of the last singular vector
        Vt[-1, :] *= -1
        R = U @ Vt
    return R
