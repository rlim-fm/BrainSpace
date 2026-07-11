"""Invariants for geometry/alignment helpers in util.py."""
import numpy as np

from brainspace.internal.util import kabsch_procrustes, ParameterizedCurve


# ----------------------------------------------------------------------------
# kabsch_procrustes
# ----------------------------------------------------------------------------

def test_kabsch_recovers_known_rotation():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((20, 3))
    theta = 0.7
    Rtrue = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ])
    B = A @ Rtrue
    R = kabsch_procrustes(A, B)
    np.testing.assert_allclose(A @ R, B, atol=1e-8)


def test_kabsch_output_is_proper_rotation():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((15, 3))
    B = rng.standard_normal((15, 3))
    R = kabsch_procrustes(A, B)
    # Orthogonal (R R^T = I) and a proper rotation (det = +1, reflection corrected).
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
    assert np.linalg.det(R) > 0
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)


def test_kabsch_corrects_reflection():
    # A target related to A by a reflection must still yield a proper rotation.
    rng = np.random.default_rng(2)
    A = rng.standard_normal((30, 3))
    reflect = np.diag([1.0, 1.0, -1.0])
    B = A @ reflect
    R = kabsch_procrustes(A, B)
    assert np.linalg.det(R) > 0


# ----------------------------------------------------------------------------
# ParameterizedCurve
# ----------------------------------------------------------------------------

def test_parameterized_curve_evaluates_user_function():
    # General contract: __call__ / generate_curve return the function's output.
    curve = ParameterizedCurve(lambda t: np.stack([t, 2 * t], axis=-1))
    pts = curve.generate_curve((0.0, 1.0), n_points=5)
    assert pts.shape == (5, 2)
    np.testing.assert_allclose(pts[:, 1], 2 * pts[:, 0])


def test_axis_curve_scales_with_t():
    # axis_curve(axis=0) yields points proportional to the parameter t.
    curve = ParameterizedCurve.axis_curve(input_dim=4, axis=0)
    t = np.linspace(-1.0, 1.0, 10)
    pts = np.asarray(curve(t)).reshape(-1)
    np.testing.assert_allclose(pts, t)
