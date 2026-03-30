import numpy as np


def optical_to_airsim_camera_frame(point_optical):
    """Convert optical frame [x_right, y_down, z_forward] to AirSim camera frame [x_fwd, y_right, z_down]."""
    p = np.asarray(point_optical, dtype=float).reshape(3)
    # x_fwd = z_opt, y_right = x_opt, z_down = y_opt
    return np.array([p[2], p[0], p[1]], dtype=float)


def camera_to_world_point(location_cam, world_transform):
    """Convert camera-coordinate point into world coordinates when transform is available."""
    if world_transform is None or location_cam is None:
        return None

    R = world_transform['R']
    cam_center = world_transform['camera_center']
    convention = world_transform.get('convention', 'world_to_camera')

    if convention == 'camera_to_world':
        # AirSim rotation uses camera/body frame (x_fwd, y_right, z_down),
        # while pinhole back-projection gives optical frame (x_right, y_down, z_fwd).
        location_cam_airsim = optical_to_airsim_camera_frame(location_cam)
        return R @ location_cam_airsim + cam_center.squeeze()
    return R.T @ location_cam + cam_center.squeeze()


def estimate_from_depth_map(
    depth_estimator,
    depth_map,
    bbox,
    class_name,
    camera_matrix,
    world_transform,
    depth_to_distance,
    is_metric,
    method_suffix,
):
    """Estimate 3D point from depth map using back-projection and optional world transform."""
    if depth_map is None:
        return None

    depth_value_raw, depth_method_raw = depth_estimator.estimate_object_depth(
        depth_map,
        bbox,
        class_name=class_name,
        # Keep V2 and V3 independent (no shared EMA state between versions).
        object_id=None,
    )

    distance_m = float(depth_value_raw) if is_metric else float(depth_to_distance(depth_value_raw))

    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    pt2 = np.array([cx, cy, 1.0], dtype=float)

    location_cam_est = np.linalg.inv(camera_matrix) @ pt2 * distance_m
    location_world_est = camera_to_world_point(location_cam_est, world_transform)

    return {
        'depth_value': depth_value_raw,
        'depth_method': f"{depth_method_raw}+{method_suffix}",
        'depth_unit': 'm' if is_metric else 'norm',
        'distance_m': distance_m,
        'location_cam': location_cam_est,
        'location_world': location_world_est,
        'uv_est': np.array([cx, cy], dtype=float),
    }
