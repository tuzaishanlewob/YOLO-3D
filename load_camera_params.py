"""Backward-compatibility shim.

Camera parameter utilities now live in `camera_module.py`.
"""

from camera_module import apply_camera_params_to_estimator
from camera_module import build_camera_params_from_airsim
from camera_module import create_projection_matrix
from camera_module import load_camera_params

__all__ = [
    "load_camera_params",
    "build_camera_params_from_airsim",
    "create_projection_matrix",
    "apply_camera_params_to_estimator",
]
