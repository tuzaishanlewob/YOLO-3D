import json
import math
import os

import numpy as np


def depth_to_distance(depth_value):
    """Map normalized relative depth (0-1) to display distance in meters."""
    return 1.0 + float(depth_value) * 9.0


def optical_to_airsim_camera_frame(point_optical):
    """Convert optical frame [x_right, y_down, z_forward] to AirSim camera frame [x_fwd, y_right, z_down]."""
    p = np.asarray(point_optical, dtype=float).reshape(3)
    return np.array([p[2], p[0], p[1]], dtype=float)


def camera_to_world_point(location_cam, world_transform):
    """Convert camera-coordinate point into world coordinates when transform is available."""
    if world_transform is None or location_cam is None:
        return None

    R = world_transform.get("R")
    if R is None:
        return None

    convention = world_transform.get("convention", "world_to_camera")

    # Prefer explicit camera center when available.
    cam_center = world_transform.get("camera_center")
    if cam_center is None and "t" in world_transform:
        t = np.asarray(world_transform["t"], dtype=float).reshape(3, 1)
        cam_center = -R.T @ t

    if cam_center is None:
        return None

    if convention == "camera_to_world":
        location_cam_airsim = optical_to_airsim_camera_frame(location_cam)
        return R @ location_cam_airsim + cam_center.squeeze()

    return R.T @ np.asarray(location_cam, dtype=float).reshape(3) + cam_center.squeeze()


def estimate_from_depth_map(
    depth_estimator,
    depth_map,
    bbox,
    class_name,
    camera_matrix,
    world_transform,
    is_metric,
    method_suffix,
    center_x=None,
    center_y=None,
):
    """Estimate 3D point from depth map using back-projection and optional world transform."""
    if depth_map is None:
        return None

    depth_value_raw, depth_method_raw = depth_estimator.estimate_object_depth(
        depth_map,
        bbox,
        class_name=class_name,
        object_id=None,
        center_x=center_x,
        center_y=center_y,
    )

    distance_m = float(depth_value_raw) if is_metric else float(depth_to_distance(depth_value_raw))

    if center_x is not None and center_y is not None:
        cx = float(center_x)
        cy = float(center_y)
    else:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

    pt2 = np.array([cx, cy, 1.0], dtype=float)
    location_cam_est = np.linalg.inv(camera_matrix) @ pt2 * distance_m
    location_world_est = camera_to_world_point(location_cam_est, world_transform)

    return {
        "depth_value": depth_value_raw,
        "depth_method": f"{depth_method_raw}+{method_suffix}",
        "depth_unit": "m" if is_metric else "norm",
        "distance_m": distance_m,
        "location_cam": location_cam_est,
        "location_world": location_world_est,
        "uv_est": np.array([cx, cy], dtype=float),
    }


def create_projection_matrix(camera_matrix, R=None, t=None):
    """Create a projection matrix from camera intrinsic and extrinsic parameters."""
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros((3, 1))

    RT = np.hstack((R, t))
    return camera_matrix @ RT


def load_camera_params(params_file):
    """Load camera parameters from a JSON file."""
    if not os.path.exists(params_file):
        print(f"Warning: Camera parameters file {params_file} not found. Using default parameters.")
        return None

    try:
        with open(params_file, "r") as f:
            params = json.load(f)

        params["camera_matrix"] = np.array(params["camera_matrix"])
        params["dist_coeffs"] = np.array(params.get("dist_coeffs", []))

        if "rotation_matrix" in params:
            R = np.array(params["rotation_matrix"])
            params["R"] = R

            if "translation_vector" in params:
                t_cam = np.array(params["translation_vector"]).reshape(3, 1)
                params["t"] = t_cam
                params["projection_matrix"] = create_projection_matrix(params["camera_matrix"], R, t_cam)
            elif "camera_center" in params:
                center = np.array(params["camera_center"]).reshape(3, 1)
                params["camera_center"] = center
                t_cam = -R @ center
                params["t"] = t_cam
                params["projection_matrix"] = create_projection_matrix(params["camera_matrix"], R, t_cam)
            else:
                t_cam = np.zeros((3, 1))
                params["t"] = t_cam
                params["projection_matrix"] = create_projection_matrix(params["camera_matrix"], R, t_cam)

        elif "projection_matrix" in params:
            params["projection_matrix"] = np.array(params["projection_matrix"])
            try:
                K_inv = np.linalg.inv(params["camera_matrix"])
                RT = K_inv @ params["projection_matrix"]
                params["R"] = RT[:, :3]
                params["t"] = RT[:, 3:].reshape(3, 1)
            except Exception:
                pass
        else:
            params["projection_matrix"] = create_projection_matrix(params["camera_matrix"])

        if "R" in params and "camera_center" in params:
            params["convention"] = "world_to_camera"

        print(f"Loaded camera parameters from {params_file}")
        print(f"Camera matrix:\n{params['camera_matrix']}")
        print(f"Projection matrix:\n{params['projection_matrix']}")
        return params
    except Exception as e:
        print(f"Error loading camera parameters: {e}")
        return None


def build_camera_params_from_airsim(camera_info, image_width, image_height, airsim_utils_module):
    """Build camera matrices from AirSim camera info."""
    hfov = math.radians(camera_info.fov)
    fx = image_width / (2 * math.tan(hfov / 2))
    fy = fx
    cx = image_width * 0.5
    cy = image_height * 0.5

    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    r_cam_to_world = airsim_utils_module.rotation_matrix_from_quat(camera_info.pose.orientation)
    r_world_to_cam = r_cam_to_world.T
    camera_center = np.array(
        [
            camera_info.pose.position.x_val,
            camera_info.pose.position.y_val,
            camera_info.pose.position.z_val,
        ],
        dtype=float,
    ).reshape(3, 1)
    t = -r_world_to_cam @ camera_center
    projection_matrix = camera_matrix @ np.hstack((r_world_to_cam, t))

    return {
        "camera_matrix": camera_matrix,
        "projection_matrix": projection_matrix,
        "R": r_world_to_cam,
        "t": t,
        "camera_center": camera_center,
        "R_cam_to_world": r_cam_to_world,
        "convention": "camera_to_world",
    }


def apply_camera_params_to_estimator(bbox3d_estimator, params):
    """Apply camera parameters to a 3D bounding box estimator."""
    if params is None:
        print("Warning: No camera parameters provided. Using default parameters.")
        return bbox3d_estimator

    if "camera_matrix" in params:
        bbox3d_estimator.K = params["camera_matrix"]

    if "projection_matrix" in params:
        bbox3d_estimator.P = params["projection_matrix"]

    print("Applied camera parameters to 3D bounding box estimator")
    return bbox3d_estimator
