import math

import cv2
import numpy as np


def airsim_camera_frame_to_optical(point_camera):
    """Convert AirSim camera frame [x_fwd, y_right, z_down] to optical [x_right, y_down, z_fwd]."""
    p = np.asarray(point_camera, dtype=float).reshape(3)
    # x_opt = y_cam, y_opt = z_cam, z_opt = x_cam
    return np.array([p[1], p[2], p[0]], dtype=float)


def depth_to_distance(depth_value):
    """Map normalized depth (0-1) to display distance in meters."""
    return 1.0 + float(depth_value) * 9.0


def get_airsim_scene_frame(client, camera_name, vehicle_name, airsim_module):
    """Fetch one RGB scene frame from AirSim and decode it to BGR."""
    responses = client.simGetImages([
        airsim_module.ImageRequest(camera_name, airsim_module.ImageType.Scene, False, True)
    ], vehicle_name)

    if not responses:
        return None

    response = responses[0]
    if response is None or not response.image_data_uint8:
        return None

    frame_buffer = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    if frame_buffer.size == 0:
        return None

    return cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)


def get_airsim_scene_and_depth(client, camera_name, vehicle_name, airsim_module):
    """Fetch synchronized scene and DepthPlanar frames from AirSim."""
    responses = client.simGetImages([
        airsim_module.ImageRequest(camera_name, airsim_module.ImageType.Scene, False, True),
        airsim_module.ImageRequest(camera_name, airsim_module.ImageType.DepthPlanar, True),
    ], vehicle_name)

    if not responses or len(responses) < 2:
        return None, None

    scene_resp, depth_resp = responses[0], responses[1]
    if scene_resp is None or not scene_resp.image_data_uint8:
        return None, None
    if depth_resp is None or not depth_resp.image_data_float:
        return None, None

    frame = cv2.imdecode(np.frombuffer(scene_resp.image_data_uint8, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None, None

    depth_map = airsim_module.list_to_2d_float_array(
        depth_resp.image_data_float,
        depth_resp.width,
        depth_resp.height,
    )
    return frame, depth_map


def colorize_metric_depth(depth_metric_map, colorize_depth_fn):
    """Convert metric depth map to a stable visualization range."""
    valid = depth_metric_map[np.isfinite(depth_metric_map) & (depth_metric_map > 0)]
    if valid.size == 0:
        return np.zeros((depth_metric_map.shape[0], depth_metric_map.shape[1], 3), dtype=np.uint8)

    lo = np.percentile(valid, 5)
    hi = np.percentile(valid, 95)
    hi = max(hi, lo + 1e-3)
    normalized = np.clip((depth_metric_map - lo) / (hi - lo), 0.0, 1.0)
    return colorize_depth_fn(normalized)


def build_camera_params_from_airsim(camera_info, image_width, image_height, airsim_utils_module):
    """Build camera params dict from AirSim camera info for estimator compatibility."""
    hfov = math.radians(camera_info.fov)
    aspect = image_width / image_height
    vfov = 2 * math.atan(math.tan(hfov / 2) / aspect)

    fx = image_width / (2 * math.tan(hfov / 2))
    fy = image_height / (2 * math.tan(vfov / 2))
    cx = image_width * 0.5
    cy = image_height * 0.5

    camera_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    R_cw = airsim_utils_module.rotation_matrix_from_quat(camera_info.pose.orientation)
    R_wc = R_cw.T

    camera_center = np.array([
        camera_info.pose.position.x_val,
        camera_info.pose.position.y_val,
        camera_info.pose.position.z_val,
    ], dtype=float).reshape(3, 1)

    t = -R_wc @ camera_center
    projection_matrix = camera_matrix @ np.hstack((R_wc, t))

    return {
        "camera_matrix": camera_matrix,
        "projection_matrix": projection_matrix,
        "R": R_wc,
        "t": t,
        "camera_center": camera_center,
        "R_cam_to_world": R_cw,
        "convention": "camera_to_world",
    }


def world_to_camera(world_point, transform):
    """Convert a world-space 3D point to camera coordinates."""
    if transform is None or world_point is None:
        return None

    point = np.asarray(world_point, dtype=float).reshape(3)
    cam_center = np.asarray(transform["camera_center"], dtype=float).reshape(3)
    R = np.asarray(transform["R"], dtype=float)
    convention = transform.get("convention", "world_to_camera")

    if convention == "camera_to_world":
        # AirSim world->camera result is in camera/body frame; convert to optical for K projection.
        point_cam_airsim = R.T @ (point - cam_center)
        return airsim_camera_frame_to_optical(point_cam_airsim)
    return R @ (point - cam_center)


def project_camera_point(cam_point, K):
    """Project camera-coordinate point into image coordinates."""
    point = np.asarray(cam_point, dtype=float).reshape(3)
    if not np.isfinite(point).all() or abs(point[2]) < 1e-6:
        return None

    pix = K @ point
    if abs(pix[2]) < 1e-6:
        return None
    return np.array([pix[0] / pix[2], pix[1] / pix[2]], dtype=float)
