import fnmatch
import math
import random
import time

import cv2
import numpy as np

import setup_path
import cosysairsim as airsim
from cosysairsim import utils as airsim_utils

from back_project import estimate_from_depth_map
from dataset_export import SegmentationDatasetExporter
from depth_model import DepthEstimator
from load_camera_params import build_camera_params_from_airsim
from segmentation_helpers import compute_masked_depth_stats

use_model = False
enable_tracking = True
enable_stream = True
enable_mask = True
show_preview = True

export_dataset = True
dataset_root = "dataset_airsim"
hdf5_include_segmentation = True
export_every_n_frames = 1
max_frames = None

device = 0
camera = "0"
vehicle_name = ""
radius_m = 200.0
pattern = "drone"
dataset_class_name = "drone"
scene_compressed = False
airsim_refresh_camera_params_every_frame = True

image_width_hint = 1920
image_height_hint = 1080
frames_per_camera_pose = 10
camera_start_points = [f"Cube{i}" for i in range(11)]
altitude_levels_per_point = 1
altitude_z_gap_m = 10.0
teleport_settle_secs = 0.2
pre_capture_wait_secs = 0.0
ignore_collision_on_teleport = True

spawn_random_drones_in_frustum = True
drone_labels = [f"drone{i}" for i in range(14)]
min_drones_per_frame = 1
max_drones_per_frame = 2
min_forward_m = 10.0
max_forward_m = 40.0
frustum_buffer_deg = 5.0
min_drone_separation_m = 2.0
max_spawn_attempts_per_drone = 10
spawn_scale_xyz = (1.0, 1.0, 1.0)
random_seed = None

yolo_model_size = "small"
yolo_weights = r"E:\Programs\AirSim\Cosys-AirSim\runs\detect\train9\weights\best.pt"
depth_model_size = "outdoor"
depth_raw_encoder = "vits"
depth_custom_weights = r"E:\Programs\AirSim\Cosys-AirSim\PythonClient\YOLO-3D\checkpoints\depth_anything_v2_metric_hypersim_vits.pth"

use_model = use_model and (not export_dataset)
show_preview = show_preview and (not export_dataset)
match_pattern = pattern if any(char in pattern for char in "*?[]") else f"{pattern}*"


def vec3(v):
    return [float(v.x_val), float(v.y_val), float(v.z_val)]


def pose_dict(pose):
    return {
        "position": {"x": float(pose.position.x_val), "y": float(pose.position.y_val), "z": float(pose.position.z_val)},
        "orientation": {
            "w": float(pose.orientation.w_val),
            "x": float(pose.orientation.x_val),
            "y": float(pose.orientation.y_val),
            "z": float(pose.orientation.z_val),
        },
    }


def pose_ok(pose):
    if pose is None:
        return False
    values = vec3(pose.position) + [
        float(pose.orientation.w_val),
        float(pose.orientation.x_val),
        float(pose.orientation.y_val),
        float(pose.orientation.z_val),
    ]
    return all(math.isfinite(value) for value in values)


def to_float_list(value):
    if value is None:
        return None
    value = np.asarray(value, dtype=float).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        return None
    return [float(v) for v in value.tolist()]


def camera_to_world(camera_pose, point_cam):
    rotation = np.asarray(airsim_utils.rotation_matrix_from_quat(camera_pose.orientation), dtype=float)
    return rotation @ np.asarray(point_cam, dtype=float).reshape(3) + np.asarray(vec3(camera_pose.position), dtype=float)


def world_pose(camera_pose, relative_pose):
    xyz = camera_to_world(camera_pose, vec3(relative_pose.position))
    return {
        "position": {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])},
        "orientation": {
            "w": float((camera_pose.orientation * relative_pose.orientation).w_val),
            "x": float((camera_pose.orientation * relative_pose.orientation).x_val),
            "y": float((camera_pose.orientation * relative_pose.orientation).y_val),
            "z": float((camera_pose.orientation * relative_pose.orientation).z_val),
        },
    }

_COLOR_MAP = None

def get_color_encoded_id(instance_id):
    if instance_id is None:
        return None
    global _COLOR_MAP
    if _COLOR_MAP is None:
        _COLOR_MAP = airsim_utils.load_colormap()
        
    try:
        color = _COLOR_MAP[int(instance_id)]
        return int(color[0]) + (int(color[1]) << 8) + (int(color[2]) << 16)
    except Exception:
        return None

def find_mask_centroid(seg_mask, instance_id):
    coords = None if seg_mask is None or instance_id is None else np.argwhere(seg_mask == int(instance_id))
    if coords is None or coords.size == 0:
        return None
    center_y, center_x = coords.mean(axis=0)
    return float(center_x), float(center_y)


def draw_bbox(image, bbox, label, color):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def fetch_airsim_bundle(client):
    responses = client.simGetImages(
        [
            airsim.ImageRequest(camera, airsim.ImageType.Scene, False, bool(scene_compressed)),
            airsim.ImageRequest(camera, airsim.ImageType.DepthPlanar, True, False),
            airsim.ImageRequest(camera, airsim.ImageType.Segmentation, False, False),
        ],
        vehicle_name,
    )
    if not responses or len(responses) < 3:
        return None, None, None, None

    scene_resp, depth_resp, seg_resp = responses[:3]
    if scene_resp is None or not scene_resp.image_data_uint8:
        return None, None, None, None

    if scene_compressed:
        encoded = np.frombuffer(scene_resp.image_data_uint8, dtype=np.uint8)
        frame = None if encoded.size == 0 else cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    else:
        width = int(scene_resp.width)
        height = int(scene_resp.height)
        pixel_count = width * height
        flat = np.frombuffer(scene_resp.image_data_uint8, dtype=np.uint8)
        channels = 0 if pixel_count <= 0 else flat.size // pixel_count
        if width <= 0 or height <= 0 or channels not in (3, 4) or flat.size < pixel_count * 3:
            return None, None, None, None
        frame = cv2.cvtColor(flat[: pixel_count * channels].reshape(height, width, channels)[:, :, :3], cv2.COLOR_RGB2BGR)
    if frame is None:
        return None, None, None, None

    depth_map = None
    if depth_resp is not None and depth_resp.image_data_float:
        depth_map = airsim.list_to_2d_float_array(depth_resp.image_data_float, depth_resp.width, depth_resp.height)
        depth_map = np.asarray(depth_map, dtype=np.float32)

    seg_mask = None
    seg_rgb = None
    if seg_resp is not None and seg_resp.image_data_uint8:
        width = int(seg_resp.width)
        height = int(seg_resp.height)
        pixel_count = width * height
        flat = np.frombuffer(seg_resp.image_data_uint8, dtype=np.uint8)
        channels = 0 if pixel_count <= 0 else flat.size // pixel_count
        if width > 0 and height > 0 and channels in (3, 4) and flat.size >= pixel_count * 3:
            seg_rgb = flat[: pixel_count * channels].reshape(height, width, channels)[:, :, :3]
            seg_mask = (
                seg_rgb[:, :, 0].astype(np.uint32)
                + (seg_rgb[:, :, 1].astype(np.uint32) << 8)
                + (seg_rgb[:, :, 2].astype(np.uint32) << 16)
            )

    return frame, depth_map, seg_mask, seg_rgb


def configure_detection(client):
    patterns = {match_pattern, match_pattern.lower(), match_pattern.upper()}
    for name in drone_labels:
        drone_pattern = name if any(char in name for char in "*?[]") else f"{name}*"
        patterns.update({drone_pattern, drone_pattern.lower(), drone_pattern.upper()})
    client.simClearDetectionMeshNames(camera, airsim.ImageType.Scene, vehicle_name)
    client.simSetDetectionFilterRadius(camera, airsim.ImageType.Scene, float(radius_m) * 100.0, vehicle_name)
    for name in sorted(patterns):
        client.simAddDetectionFilterMeshName(camera, airsim.ImageType.Scene, name, vehicle_name)


def spawn_drones(client, camera_info, width, height, session_tag, frame_tag):
    if not spawn_random_drones_in_frustum or not drone_labels:
        return []

    fov_h = math.radians(camera_info.fov)
    fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0) / (float(width) / float(height)))
    tan_h = math.tan(max(0.01, fov_h / 2.0 - math.radians(frustum_buffer_deg)))
    tan_v = math.tan(max(0.01, fov_v / 2.0 - math.radians(frustum_buffer_deg)))
    scale = airsim.Vector3r(*[float(v) for v in spawn_scale_xyz])
    count = random.randint(max(0, int(min_drones_per_frame)), max(int(min_drones_per_frame), int(max_drones_per_frame)))
    chosen = random.sample(drone_labels, k=min(len(drone_labels), count))
    occupied = []
    spawned = []

    for spawn_index, label in enumerate(chosen):
        for _ in range(max(1, int(max_spawn_attempts_per_drone))):
            forward = random.uniform(min_forward_m, max_forward_m)
            point_cam = [
                forward,
                random.uniform(-forward * tan_h, forward * tan_h),
                random.uniform(-forward * tan_v, forward * tan_v),
            ]
            world_xyz = camera_to_world(camera_info.pose, point_cam)
            if any(np.linalg.norm(world_xyz - prev_xyz) < min_drone_separation_m for prev_xyz in occupied):
                continue

            pitch = math.radians(random.uniform(-20.0, 20.0))
            roll = math.radians(random.uniform(-20.0, 20.0))
            yaw = math.radians(random.uniform(-180.0, 180.0))
            pose = airsim.Pose(
                airsim.Vector3r(float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])),
                airsim.euler_to_quaternion(roll, pitch, yaw),
            )
            desired_name = f"{label}_frustum_{session_tag}_{frame_tag}_{spawn_index}"
            spawned_name = client.simSpawnObject(desired_name, label.lower(), pose, scale)
            if not spawned_name:
                continue

            try:
                instance_id = int(client.simGetSegmentationObjectID(spawned_name))
                if instance_id <= 0:
                    instance_id = None
            except Exception:
                instance_id = None

            spawned.append(
                {
                    "name": str(spawned_name),
                    "instance_id": instance_id,
                    "class_name": dataset_class_name,
                    "class_id": 0,
                    "world_position": {"x": float(world_xyz[0]), "y": float(world_xyz[1]), "z": float(world_xyz[2])},
                    "camera_position": {"x": float(point_cam[0]), "y": float(point_cam[1]), "z": float(point_cam[2])},
                }
            )
            occupied.append(world_xyz)
            break

    return spawned


def capture_poses(client):
    frames = max(1, int(frames_per_camera_pose))
    if not export_dataset:
        while True:
            yield None, None, None, None, 0, None, False
        return

    found_pose = False
    for marker_name in camera_start_points:
        try:
            marker_pose = client.simGetObjectPose(marker_name)
        except Exception as exc:
            print(f"Warning: Could not read pose for {marker_name}: {exc}")
            continue
        if not pose_ok(marker_pose):
            print(f"Warning: Skipping invalid marker pose for {marker_name}")
            continue
        found_pose = True
        for altitude_level in range(max(1, int(altitude_levels_per_point))):
            camera_pose = airsim.Pose(
                airsim.Vector3r(
                    marker_pose.position.x_val,
                    marker_pose.position.y_val,
                    marker_pose.position.z_val - altitude_level * altitude_z_gap_m,
                ),
                marker_pose.orientation,
            )
            for sample_index in range(frames):
                yield (
                    f"{marker_name}_z{altitude_level}_{sample_index:03d}",
                    marker_name,
                    marker_pose,
                    camera_pose,
                    altitude_level,
                    sample_index,
                    sample_index == 0,
                )

    if found_pose:
        return

    fallback_pose = None
    try:
        fallback_pose = client.simGetVehiclePose(vehicle_name)
    except Exception:
        pass
    if not pose_ok(fallback_pose):
        fallback_pose = client.simGetCameraInfo(camera, vehicle_name).pose

    for sample_index in range(frames):
        yield f"static_z0_{sample_index:03d}", None, None, fallback_pose, 0, sample_index, sample_index == 0


def init_models():
    detector = None
    if use_model:
        try:
            from detection_model import ObjectDetector

            detector = ObjectDetector(model_size=yolo_model_size, device=device, weights_path=yolo_weights or None)
        except Exception as exc:
            print(f"Warning: YOLO initialization failed, continuing without model detections: {exc}")
            detector = None

        try:
            depth_name = depth_raw_encoder if depth_custom_weights else depth_model_size
            depth_estimator = DepthEstimator(model_size=depth_name, device=device, weights_path=depth_custom_weights or None)
        except Exception as exc:
            print(f"Warning: Depth model initialization failed, using AirSim DepthPlanar only: {exc}")
            depth_estimator = DepthEstimator(skip_model_init=True, device=device)
    else:
        depth_estimator = DepthEstimator(skip_model_init=True, device=device)
    return detector, depth_estimator


def main():
    if random_seed is not None:
        random.seed(random_seed)

    detector, depth_estimator = init_models()
    dataset_writer = None
    camera_info_saved = False
    session_tag = time.strftime("%Y%m%d_%H%M%S")

    if export_dataset:
        try:
            dataset_writer = SegmentationDatasetExporter(dataset_root=dataset_root)
            dataset_writer.create_dataset_yaml(class_names=[dataset_class_name], num_classes=1)
            print(f"Dataset export enabled: {dataset_root}")
        except Exception as exc:
            print(f"Error: Dataset export setup failed: {exc}")
            return

    client = airsim.VehicleClient()
    try:
        client.confirmConnection()
    except Exception as exc:
        print(f"Error: Could not connect to AirSim: {exc}")
        cv2.destroyAllWindows()
        return

    configure_detection(client)

    frame_count = 0
    width = int(image_width_hint)
    height = int(image_height_hint)
    camera_info = None
    params = None
    world_transform = None

    try:
        for frame_tag, marker_name, marker_pose, transport_pose, altitude_level, sample_index, moved_vehicle in capture_poses(client):
            spawned_objects = []

            if moved_vehicle and transport_pose is not None:
                client.simSetVehiclePose(transport_pose, bool(ignore_collision_on_teleport), vehicle_name)
                if teleport_settle_secs > 0.0:
                    time.sleep(float(teleport_settle_secs))
                if pre_capture_wait_secs > 0.0:
                    time.sleep(float(pre_capture_wait_secs))
                camera_info = None

            try:
                if camera_info is None or params is None or world_transform is None:
                    camera_info = client.simGetCameraInfo(camera, vehicle_name)
                    params = build_camera_params_from_airsim(camera_info, width, height, airsim_utils)
                    world_transform = {
                        "R": params["R_cam_to_world"],
                        "camera_center": params["camera_center"],
                        "convention": "camera_to_world",
                    }

                if export_dataset:
                    spawned_objects = spawn_drones(client, camera_info, width, height, session_tag, frame_tag)
                    if spawn_random_drones_in_frustum and not spawned_objects:
                        print(f"Warning: No drones spawned for {frame_tag}, skipping frame.")
                        continue

                frame, frame_depth_planar, seg_mask, seg_rgb = fetch_airsim_bundle(client)
                if frame is None:
                    print("Warning: Empty frame from AirSim, skipping")
                    continue

                frame_count += 1
                old_width, old_height = width, height
                height, width = frame.shape[:2]
                if (
                    camera_info is None
                    or params is None
                    or world_transform is None
                    or airsim_refresh_camera_params_every_frame
                    or moved_vehicle
                    or width != old_width
                    or height != old_height
                ):
                    camera_info = client.simGetCameraInfo(camera, vehicle_name)
                    params = build_camera_params_from_airsim(camera_info, width, height, airsim_utils)
                    world_transform = {
                        "R": params["R_cam_to_world"],
                        "camera_center": params["camera_center"],
                        "convention": "camera_to_world",
                    }

                camera_pose_world = pose_dict(camera_info.pose)
                camera_position_world = vec3(camera_info.pose.position)
                result_frame = frame.copy() if show_preview else None
                depth_colored = depth_estimator.colorize_depth(frame_depth_planar) if show_preview and frame_depth_planar is not None else None
                model_detections_metadata = []

                if detector is not None:
                    detection_frame = frame.copy()
                    try:
                        detection_frame, yolo_detections = detector.detect(
                            detection_frame,
                            track=enable_tracking,
                            stream=enable_stream,
                        )
                    except Exception as exc:
                        print(f"Warning: Model detection failed this frame: {exc}")
                        yolo_detections = []

                    try:
                        model_depth_map = depth_estimator.estimate_depth(frame)
                    except Exception as exc:
                        print(f"Warning: Model depth failed this frame: {exc}")
                        model_depth_map = None
                    else:
                        if show_preview:
                            depth_colored = depth_estimator.colorize_depth(model_depth_map)

                    if show_preview:
                        result_frame = detection_frame

                    class_names = detector.get_class_names()
                    for bbox, score, class_id, obj_id in yolo_detections:
                        class_name = str(class_names[int(class_id)])
                        estimate = None
                        if model_depth_map is not None:
                            estimate = estimate_from_depth_map(
                                depth_estimator=depth_estimator,
                                depth_map=model_depth_map,
                                bbox=bbox,
                                class_name=class_name,
                                camera_matrix=params["camera_matrix"],
                                world_transform=world_transform,
                                is_metric=bool(
                                    getattr(depth_estimator, "is_metric_depth", False)
                                    or getattr(depth_estimator, "use_raw_model", False)
                                ),
                                method_suffix="model-linalg",
                            )

                        model_detections_metadata.append(
                            {
                                "bbox": [float(v) for v in bbox],
                                "score": float(score),
                                "class_name": class_name,
                                "class_id": int(class_id),
                                "object_id": None if obj_id is None else int(obj_id),
                                "depth_m": None if estimate is None else float(estimate["distance_m"]),
                                "location_world": None if estimate is None else to_float_list(estimate["location_world"]),
                                "uv_est": None if estimate is None else to_float_list(estimate["uv_est"]),
                            }
                        )

                # Build lookup for spawned objects by their labels for easier matching
                spawned_by_id = {obj.get("instance_id"): obj for obj in spawned_objects if obj.get("instance_id")}
                
                gt_detections_metadata = []
                for det in client.simGetDetections(camera, airsim.ImageType.Scene, vehicle_name) or []:
                    det_name = str(det.name).strip() if det.name else ""
                    if match_pattern and not fnmatch.fnmatch(det_name.lower(), match_pattern.lower()):
                        continue

                    bbox = [
                        float(det.box2D.min.x_val),
                        float(det.box2D.min.y_val),
                        float(det.box2D.max.x_val),
                        float(det.box2D.max.y_val),
                    ]
                    
                    # Depth estimate from AirSim's depth map
                    planar_estimate = None
                    if frame_depth_planar is not None:
                        planar_estimate = estimate_from_depth_map(
                            depth_estimator=depth_estimator,
                            depth_map=frame_depth_planar,
                            bbox=bbox,
                            class_name=dataset_class_name,
                            camera_matrix=params["camera_matrix"],
                            world_transform=world_transform,
                            is_metric=True,
                            method_suffix="depthplanar-linalg",
                        )

                    # Get segmentation mask metadata if available
                    mask_metadata = {
                        "instance_id": None,
                        "centroid_uv": None,
                        "depth_m": None,
                        "location_world": None,
                        "valid_pixels": None,
                    }
                    
                    if enable_mask and seg_mask is not None:
                        # Try to get instance_id: first check spawned objects, then fall back to registry
                        instance_id = None
                        for obj in spawned_objects:
                            if obj.get("instance_id") and fnmatch.fnmatch(det_name.lower(), f"{obj.get('name', '')}*".lower()):
                                instance_id = obj.get("instance_id")
                                break
                        
                        if instance_id is None:
                            try:
                                # Fallback: query registry directly using detection name
                                instance_id = int(client.simGetSegmentationObjectID(det_name))
                                if instance_id <= 0:
                                    instance_id = None
                            except Exception:
                                instance_id = None

                        if instance_id is not None:
                            # seg_mask: 32-bit image where pixel values = instance IDs
                            # Find all pixels belonging to this instance and compute their centroid
                            encoded_color_id = get_color_encoded_id(instance_id)
                            centroid = find_mask_centroid(seg_mask, encoded_color_id)
                            depth_stats = compute_masked_depth_stats(frame_depth_planar, seg_mask, instance_id)
                            
                            mask_estimate = None
                            if centroid is not None and frame_depth_planar is not None:
                                mask_estimate = estimate_from_depth_map(
                                    depth_estimator=depth_estimator,
                                    depth_map=frame_depth_planar,
                                    bbox=bbox,
                                    class_name=dataset_class_name,
                                    camera_matrix=params["camera_matrix"],
                                    world_transform=world_transform,
                                    is_metric=True,
                                    method_suffix="mask-centroid-linalg",
                                    center_x=centroid[0],
                                    center_y=centroid[1],
                                )

                            mask_metadata = {
                                "instance_id": int(instance_id),
                                "centroid_uv": None if centroid is None else [float(centroid[0]), float(centroid[1])],
                                "depth_m": None if depth_stats is None else float(depth_stats["median"]),
                                "location_world": None if mask_estimate is None else to_float_list(mask_estimate["location_world"]),
                                "valid_pixels": None if depth_stats is None else int(depth_stats["valid_pixels"]),
                            }
                            if result_frame is not None and centroid is not None:
                                cv2.circle(result_frame, (int(centroid[0]), int(centroid[1])), 4, (255, 0, 255), -1)

                    gt_detections_metadata.append(
                        {
                            "bbox": bbox,
                            "score": 1.0,
                            "class_name": dataset_class_name,
                            "class_id": 0,
                            "object_name_full": det_name,
                            "relative_pose_camera": pose_dict(det.relative_pose),
                            "relative_position_camera": vec3(det.relative_pose.position),
                            "camera_position_world": camera_position_world,
                            "camera_pose_world": camera_pose_world,
                            "gt_pose_world": world_pose(camera_info.pose, det.relative_pose),
                            "depthplanar_depth_m": None if planar_estimate is None else float(planar_estimate["distance_m"]),
                            "depthplanar_world": None if planar_estimate is None else to_float_list(planar_estimate["location_world"]),
                            "depthplanar_uv": None if planar_estimate is None else to_float_list(planar_estimate["uv_est"]),
                            "mask": mask_metadata,
                        }
                    )

                    if result_frame is not None:
                        label = dataset_class_name
                        if planar_estimate is not None:
                            label = f"{dataset_class_name} GT {planar_estimate['distance_m']:.1f}m"
                        draw_bbox(result_frame, bbox, label, (255, 0, 0))

                if export_dataset and dataset_writer is not None and frame_count % max(1, int(export_every_n_frames)) == 0:
                    dataset_writer.save_frame(
                        frame_id=frame_count,
                        rgb_image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        seg_mask=seg_mask if hdf5_include_segmentation else None,
                        segmentation_rgb=seg_rgb if hdf5_include_segmentation else None,
                        depth_map=frame_depth_planar,
                        metadata={
                            "source": "airsim",
                            "class_name": dataset_class_name,
                            "class_id": 0,
                            "session_tag": session_tag,
                            "frame_tag": frame_tag,
                            "camera_name": camera,
                            "vehicle_name": vehicle_name,
                            "frame_index": int(frame_count),
                            "camera_pose_world": camera_pose_world,
                            "camera_position_world": camera_position_world,
                            "capture_state": {
                                "camera_marker": marker_name,
                                "marker_pose": None if marker_pose is None else pose_dict(marker_pose),
                                "transport_pose": None if transport_pose is None else pose_dict(transport_pose),
                                "altitude_level": int(altitude_level),
                                "sample_index": sample_index,
                                "moved_vehicle": bool(moved_vehicle),
                            },
                            "spawned_objects": spawned_objects,
                            "gt_detections": gt_detections_metadata,
                            "model_detections": model_detections_metadata,
                        },
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )

                    if not camera_info_saved:
                        dataset_writer.save_camera_info(
                            camera_matrix=params["camera_matrix"],
                            img_shape=(height, width),
                            camera_info_dict={
                                "camera_name": camera,
                                "vehicle_name": vehicle_name,
                                "fov_degrees": float(camera_info.fov),
                                "class_name": dataset_class_name,
                                "frames_per_camera_pose": int(frames_per_camera_pose),
                            },
                        )
                        camera_info_saved = True

                if result_frame is not None:
                    cv2.putText(result_frame, f"Frame {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    if depth_colored is None:
                        depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.imshow("AirSim Result", result_frame)
                    cv2.imshow("AirSim Depth", depth_colored)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

                if max_frames is not None and frame_count >= int(max_frames):
                    break
            finally:
                for spawned in reversed(spawned_objects):
                    try:
                        client.simDestroyObject(spawned["name"])
                    except Exception as exc:
                        print(f"Warning: failed to destroy {spawned['name']}: {exc}")
                if spawned_objects:
                    time.sleep(0.05)
    finally:
        if dataset_writer is not None:
            print(f"Dataset export finished: {dataset_writer.get_dataset_stats()}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        cv2.destroyAllWindows()
