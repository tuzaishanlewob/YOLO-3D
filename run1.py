import fnmatch
import math
import time

import cv2
import numpy as np

import setup_path
import cosysairsim as airsim  # type: ignore
from cosysairsim import utils as airsim_utils

from back_project import estimate_from_depth_map  # type: ignore
from dataset_export import SegmentationDatasetExporter  # type: ignore
from depth_model import DepthEstimator  # type: ignore
from segmentation_helpers import compute_masked_depth_stats  # type: ignore
from load_camera_params import build_camera_params_from_airsim

use_model = False
enable_tracking = True
enable_stream = True
enable_mask = True
show_preview = True

export_dataset = False
dataset_root = "dataset_airsim"
hdf5_include_segmentation = True
export_every_n_frames = 1
max_frames = None

device = 0
camera = "0"
vehicle_name = ""
radius_m = 200.0
pattern = "drone*"
scene_compressed = False
airsim_refresh_camera_params_every_frame = True

yolo_model_size = "small"
yolo_weights = r"E:\Programs\AirSim\Cosys-AirSim\runs\detect\train9\weights\best.pt"
depth_model_size = "outdoor"
depth_raw_encoder = "vits"
depth_custom_weights = r"E:\Programs\AirSim\Cosys-AirSim\PythonClient\YOLO-3D\checkpoints\depth_anything_v2_metric_hypersim_vits.pth"


def depth_to_distance(depth_value):
    """Map normalized relative depth (0-1) to display distance in meters."""
    return 1.0 + float(depth_value) * 9.0


def short_airsim_class_name(object_name):
    if object_name is None:
        return "object"
    name = str(object_name).strip()
    if not name:
        return "object"
    return name.split("_", 1)[0]


def to_float_list(value):
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return [float(v) for v in arr.tolist()]


def pack_color_to_instance_id(color):
    arr = np.asarray(color, dtype=np.uint32).reshape(-1)
    if arr.size < 3:
        return None
    return int(arr[0]) + (int(arr[1]) << 8) + (int(arr[2]) << 16)


def build_name_to_instance_id_map(object_names, color_map):
    name_to_instance_id = {}
    if not object_names or color_map is None:
        return name_to_instance_id

    for idx, object_name in enumerate(object_names):
        if idx >= len(color_map):
            break
        instance_id = pack_color_to_instance_id(color_map[idx])
        if instance_id is None:
            continue
        name_to_instance_id[str(object_name).strip()] = int(instance_id)
    return name_to_instance_id


def find_mask_centroid(seg_mask, instance_id):
    if seg_mask is None or instance_id is None:
        return None
    coords = np.argwhere(seg_mask == int(instance_id))
    if coords.size == 0:
        return None
    centroid_y, centroid_x = coords.mean(axis=0)
    return float(centroid_x), float(centroid_y)


def ensure_class_id(class_name_to_id, class_name):
    if class_name not in class_name_to_id:
        class_name_to_id[class_name] = len(class_name_to_id)
    return class_name_to_id[class_name]


def draw_bbox(image, bbox, label, color, label_offset=8):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(0, y1 - label_offset)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


def decode_scene_response(scene_resp, compressed):
    if scene_resp is None or not scene_resp.image_data_uint8:
        return None

    if compressed:
        frame_buffer = np.frombuffer(scene_resp.image_data_uint8, dtype=np.uint8)
        if frame_buffer.size == 0:
            return None
        return cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)

    width = int(scene_resp.width)
    height = int(scene_resp.height)
    if width <= 0 or height <= 0:
        return None

    rgb_flat = np.frombuffer(scene_resp.image_data_uint8, dtype=np.uint8)
    pixel_count = width * height
    if pixel_count <= 0 or rgb_flat.size < pixel_count * 3:
        return None

    channels = rgb_flat.size // pixel_count
    if channels not in (3, 4):
        return None

    scene_rgb = rgb_flat[: pixel_count * channels].reshape((height, width, channels))
    scene_rgb = scene_rgb[:, :, :3]
    return cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR)


def decode_segmentation_response(seg_resp):
    if seg_resp is None or not seg_resp.image_data_uint8:
        return None, None

    width = int(seg_resp.width)
    height = int(seg_resp.height)
    if width <= 0 or height <= 0:
        return None, None

    seg_flat = np.frombuffer(seg_resp.image_data_uint8, dtype=np.uint8)
    pixel_count = width * height
    if pixel_count <= 0 or seg_flat.size < pixel_count * 3:
        return None, None

    channels = seg_flat.size // pixel_count
    if channels not in (3, 4):
        return None, None

    seg_rgb = seg_flat[: pixel_count * channels].reshape((height, width, channels))[:, :, :3]
    seg_mask = (
        seg_rgb[:, :, 0].astype(np.uint32)
        + (seg_rgb[:, :, 1].astype(np.uint32) << 8)
        + (seg_rgb[:, :, 2].astype(np.uint32) << 16)
    )
    return seg_mask, seg_rgb


def get_airsim_scene_depth_seg(client, camera_name, vehicle_name, airsim_module, scene_compressed=False):
    responses = client.simGetImages(
        [
            airsim_module.ImageRequest(camera_name, airsim_module.ImageType.Scene, False, bool(scene_compressed)),
            airsim_module.ImageRequest(camera_name, airsim_module.ImageType.DepthPlanar, True, False),
            airsim_module.ImageRequest(camera_name, airsim_module.ImageType.Segmentation, False, False),
        ],
        vehicle_name,
    )

    if not responses or len(responses) < 3:
        return None, None, None, None

    scene_resp, depth_resp, seg_resp = responses[0], responses[1], responses[2]

    frame_bgr = decode_scene_response(scene_resp, compressed=bool(scene_compressed))
    if frame_bgr is None:
        return None, None, None, None

    depth_map = None
    if depth_resp is not None and depth_resp.image_data_float:
        depth_map = airsim_module.list_to_2d_float_array(
            depth_resp.image_data_float,
            depth_resp.width,
            depth_resp.height,
        )
        depth_map = np.asarray(depth_map, dtype=np.float32)

    seg_mask, seg_rgb = decode_segmentation_response(seg_resp)
    return frame_bgr, depth_map, seg_mask, seg_rgb

def fetch_airsim_bundle(client):
    return get_airsim_scene_depth_seg(
        client,
        camera,
        vehicle_name,
        airsim,
        scene_compressed=scene_compressed,
    )


def init_models():
    detector = None

    if use_model:
        try:
            from detection_model import ObjectDetector  # type: ignore

            detector = ObjectDetector(
                model_size=yolo_model_size,
                device=device,
                weights_path=yolo_weights if yolo_weights else None,
            )
        except Exception as exc:
            print(f"Warning: YOLO initialization failed, continuing without model detections: {exc}")
            detector = None

        try:
            depth_selector = depth_raw_encoder if depth_custom_weights else depth_model_size
            depth_estimator = DepthEstimator(
                model_size=depth_selector,
                device=device,
                weights_path=depth_custom_weights if depth_custom_weights else None,
            )
        except Exception as exc:
            print(f"Warning: Depth model initialization failed, using AirSim DepthPlanar only: {exc}")
            depth_estimator = DepthEstimator(skip_model_init=True, device=device)
    else:
        depth_estimator = DepthEstimator(skip_model_init=True, device=device)

    return detector, depth_estimator

def main():
    detector, depth_estimator = init_models()
    dataset_export = None
    class_name_to_id = {}
    last_dataset_yaml_classes = -1
    camera_info_saved = False

    if export_dataset:
        try:
            dataset_export = SegmentationDatasetExporter(dataset_root=dataset_root)
            print(f"Dataset export enabled: {dataset_root}")
        except Exception as exc:
            print(f"Warning: Dataset export disabled: {exc}")
            dataset_export = None

    client = airsim.VehicleClient()
    try:
        client.confirmConnection()
    except Exception as exc:
        print(f"Error: Could not connect to AirSim: {exc}")
        cv2.destroyAllWindows()
        return

    client.simClearDetectionMeshNames(camera, airsim.ImageType.Scene, vehicle_name)
    client.simSetDetectionFilterRadius(camera, airsim.ImageType.Scene, float(radius_m) * 100.0, vehicle_name)
    client.simAddDetectionFilterMeshName(camera, airsim.ImageType.Scene, pattern, vehicle_name)

    color_map = None
    current_object_list = []
    try:
        color_map = client.simGetSegmentationColorMap()
    except Exception as exc:
        print(f"Warning: Could not read segmentation color map: {exc}")

    try:
        current_object_list = client.simListInstanceSegmentationObjects()
    except Exception as exc:
        print(f"Warning: Could not read segmentation object list: {exc}")

    name_to_instance_id = build_name_to_instance_id_map(current_object_list, color_map)

    frame_count = 0
    params = None
    camera_info = None
    world_transform = None
    
    while True:
        frame, frame_depth_planar, seg_mask, seg_rgb = fetch_airsim_bundle(client)
        if frame is None:
            print("Warning: Empty frame from AirSim, skipping")
            continue

        frame_count += 1
        height, width = frame.shape[:2]

        if params is None or airsim_refresh_camera_params_every_frame:
            camera_info = client.simGetCameraInfo(camera, vehicle_name)
            params = build_camera_params_from_airsim(camera_info, width, height, airsim_utils)
            world_transform = {
                "R": params["R_cam_to_world"],
                "camera_center": params["camera_center"],
                "convention": "camera_to_world",
            }

        result_frame = frame.copy()
        depth_colored = (
            depth_estimator.colorize_depth(frame_depth_planar)
            if frame_depth_planar is not None
            else np.zeros((height, width, 3), dtype=np.uint8)
        )

        model_depth_map = None
        model_depth_is_metric = bool(
            getattr(depth_estimator, "is_metric_depth", False) or getattr(depth_estimator, "use_raw_model", False)
        )
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
                depth_colored = depth_estimator.colorize_depth(model_depth_map)
            except Exception as exc:
                print(f"Warning: Model depth failed this frame, keeping AirSim depth preview: {exc}")
                model_depth_map = None

            result_frame = detection_frame
            class_names = detector.get_class_names()

            for bbox, score, class_id, obj_id in yolo_detections:
                class_name = str(class_names[int(class_id)])
                dataset_class_id = ensure_class_id(class_name_to_id, class_name)
                model_estimate = None
                if model_depth_map is not None:
                    model_estimate = estimate_from_depth_map(
                        depth_estimator=depth_estimator,
                        depth_map=model_depth_map,
                        bbox=bbox,
                        class_name=class_name,
                        camera_matrix=params["camera_matrix"],
                        world_transform=world_transform,
                        depth_to_distance=depth_to_distance,
                        is_metric=model_depth_is_metric,
                        method_suffix="model-linalg",
                    )

                model_detections_metadata.append(
                    {
                        "bbox": [float(v) for v in bbox],
                        "score": float(score),
                        "class_name": class_name,
                        "class_id": int(dataset_class_id),
                        "model_class_id": int(class_id),
                        "object_id": None if obj_id is None else int(obj_id),
                        "depth_m": None if model_estimate is None else float(model_estimate["distance_m"]),
                        "location_world": None if model_estimate is None else to_float_list(model_estimate["location_world"]),
                        "uv_est": None if model_estimate is None else to_float_list(model_estimate["uv_est"]),
                    }
                )

        gt_detections = client.simGetDetections(camera, airsim.ImageType.Scene, vehicle_name) or []
        gt_detections_metadata = []

        for det in gt_detections:
            full_name = str(det.name).strip() if det.name else "object"
            if not fnmatch.fnmatch(full_name.lower(), pattern.lower()):
                continue

            bbox = [
                float(det.box2D.min.x_val),
                float(det.box2D.min.y_val),
                float(det.box2D.max.x_val),
                float(det.box2D.max.y_val),
            ]
            class_name = short_airsim_class_name(full_name)
            dataset_class_id = ensure_class_id(class_name_to_id, class_name)

            planar_estimate = None
            if frame_depth_planar is not None:
                planar_estimate = estimate_from_depth_map(
                    depth_estimator=depth_estimator,
                    depth_map=frame_depth_planar,
                    bbox=bbox,
                    class_name=class_name,
                    camera_matrix=params["camera_matrix"],
                    world_transform=world_transform,
                    depth_to_distance=depth_to_distance,
                    is_metric=True,
                    method_suffix="depthplanar-linalg",
                )

            mask_metadata = {
                "instance_id": None,
                "centroid_uv": None,
                "depth_m": None,
                "location_world": None,
                "valid_pixels": None,
            }

            if enable_mask and seg_mask is not None:
                instance_id = name_to_instance_id.get(full_name)
                if instance_id is None:
                    try:
                        current_object_list = client.simListInstanceSegmentationObjects()
                        name_to_instance_id = build_name_to_instance_id_map(current_object_list, color_map)
                        instance_id = name_to_instance_id.get(full_name)
                    except Exception:
                        instance_id = None

                if instance_id is not None:
                    centroid = find_mask_centroid(seg_mask, instance_id)
                    depth_stats = compute_masked_depth_stats(frame_depth_planar, seg_mask, instance_id)
                    mask_estimate = None
                    if centroid is not None and frame_depth_planar is not None:
                        centroid_x, centroid_y = centroid
                        mask_estimate = estimate_from_depth_map(
                            depth_estimator=depth_estimator,
                            depth_map=frame_depth_planar,
                            bbox=bbox,
                            class_name=class_name,
                            camera_matrix=params["camera_matrix"],
                            world_transform=world_transform,
                            depth_to_distance=depth_to_distance,
                            is_metric=True,
                            method_suffix="mask-centroid-linalg",
                            center_x=centroid_x,
                            center_y=centroid_y,
                        )

                    mask_metadata = {
                        "instance_id": int(instance_id),
                        "centroid_uv": None if centroid is None else [float(centroid[0]), float(centroid[1])],
                        "depth_m": None if depth_stats is None else float(depth_stats["median"]),
                        "location_world": None if mask_estimate is None else to_float_list(mask_estimate["location_world"]),
                        "valid_pixels": None if depth_stats is None else int(depth_stats["valid_pixels"]),
                    }

                    if show_preview and centroid is not None:
                        cv2.circle(result_frame, (int(centroid[0]), int(centroid[1])), 4, (255, 0, 255), -1)

            gt_rel_pose = [
                float(det.relative_pose.position.x_val),
                float(det.relative_pose.position.y_val),
                float(det.relative_pose.position.z_val),
            ]

            gt_detections_metadata.append(
                {
                    "bbox": bbox,
                    "score": 1.0,
                    "class_name": class_name,
                    "class_id": int(dataset_class_id),
                    "object_name_full": full_name,
                    "relative_pose_airsim_camera": gt_rel_pose,
                    "depthplanar_depth_m": None if planar_estimate is None else float(planar_estimate["distance_m"]),
                    "depthplanar_world": None if planar_estimate is None else to_float_list(planar_estimate["location_world"]),
                    "depthplanar_uv": None if planar_estimate is None else to_float_list(planar_estimate["uv_est"]),
                    "mask": mask_metadata,
                }
            )

            label = class_name
            if planar_estimate is not None:
                label = f"{class_name} GT {mask_estimate['distance_m']:.1f}m"
            draw_bbox(result_frame, bbox, label, (255, 0, 0))

        if export_dataset and dataset_export is not None and frame_count % max(1, int(export_every_n_frames)) == 0:
            frame_metadata = {
                "source": "airsim",
                "camera_name": camera,
                "vehicle_name": vehicle_name,
                "frame_index": int(frame_count),
                "gt_detections": gt_detections_metadata,
                "model_detections": model_detections_metadata,
            }
            dataset_export.save_frame(
                frame_id=frame_count,
                rgb_image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                seg_mask=seg_mask if hdf5_include_segmentation else None,
                segmentation_rgb=seg_rgb if hdf5_include_segmentation else None,
                depth_map=frame_depth_planar,
                metadata=frame_metadata,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

            if not camera_info_saved and camera_info is not None:
                dataset_export.save_camera_info(
                    camera_matrix=params["camera_matrix"],
                    img_shape=(height, width),
                    camera_info_dict={
                        "camera_name": camera,
                        "vehicle_name": vehicle_name,
                        "fov_degrees": float(camera_info.fov),
                    },
                )
                camera_info_saved = True

            if len(class_name_to_id) != last_dataset_yaml_classes:
                ordered_names = [name for name, _ in sorted(class_name_to_id.items(), key=lambda item: item[1])]
                dataset_export.create_dataset_yaml(
                    class_names=ordered_names,
                    num_classes=len(ordered_names),
                )
                last_dataset_yaml_classes = len(class_name_to_id)

        fps_label = f"Frame {frame_count}"
        cv2.putText(
            result_frame,
            fps_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        if show_preview:
            cv2.imshow("AirSim Result", result_frame)
            cv2.imshow("AirSim Depth", depth_colored)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        if max_frames is not None and frame_count >= int(max_frames):
            break

    if dataset_export is not None:
        stats = dataset_export.get_dataset_stats()
        print(f"Dataset export finished: {stats}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        cv2.destroyAllWindows()
