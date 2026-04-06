"""Backward-compatibility shim.

Camera and back-projection utilities now live in `camera_module.py`.
"""

from camera_module import camera_to_world_point
from camera_module import depth_to_distance
from camera_module import estimate_from_depth_map
from camera_module import optical_to_airsim_camera_frame

__all__ = [
    "depth_to_distance",
    "optical_to_airsim_camera_frame",
    "camera_to_world_point",
    "estimate_from_depth_map",
]
