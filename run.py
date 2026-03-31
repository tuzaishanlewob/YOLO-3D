import os
import setup_path 
import sys
import time
import cv2
import numpy as np
import torch

try:
    import cosysairsim as airsim  # type: ignore
    from cosysairsim import utils as airsim_utils  # type: ignore
except ImportError:
    airsim = None
    airsim_utils = None

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import our modules
from detection_model import ObjectDetector # type: ignore
from depth_model import DepthEstimator # type: ignore
from bbox3d_utils import BBox3DEstimator, BirdEyeView # type: ignore
from load_camera_params import load_camera_params, apply_camera_params_to_estimator # type: ignore
from back_project import estimate_from_depth_map # type: ignore
from runtime_ui import ( # type: ignore
    VERSION_ORDER,
    RuntimeDashboard,
    apply_version_selection,
    get_version_selection,
    launch_config_ui,
    short_airsim_class_name,
)
from runtime_helpers import ( # type: ignore
    build_camera_params_from_airsim,
    colorize_metric_depth,
    depth_to_distance,
    get_airsim_scene_and_depth,
    get_airsim_scene_frame,
    project_camera_point,
    world_to_camera,
)
from runtime_config import get_default_config # type: ignore
from segmentation_helpers import (  # type: ignore
    get_airsim_segmentation_mask,
    get_present_instance_ids,
    find_best_instance_for_detection,
    compute_masked_depth_stats,
    refine_bbox_from_mask,
)
from dataset_export import SegmentationDatasetExporter # type: ignore


def _bbox_iou(box_a, box_b):
    """Compute IoU for two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def _match_yolo_detection(gt_detection, yolo_detections):
    """Pick the best YOLO detection for a GT detection using IoU + class affinity."""
    if not yolo_detections:
        return None

    gt_bbox = gt_detection.get('bbox')
    gt_class = str(gt_detection.get('class_name', '')).lower()
    best = None
    best_score = -1.0

    for det in yolo_detections:
        det_bbox = det.get('bbox')
        if det_bbox is None:
            continue

        iou = _bbox_iou(gt_bbox, det_bbox)
        det_class = str(det.get('class_name', '')).lower()
        class_bonus = 0.0
        if gt_class and det_class and (det_class.startswith(gt_class) or gt_class.startswith(det_class)):
            class_bonus = 0.1

        score = iou + class_bonus
        if score > best_score:
            best_score = score
            best = det

    if best_score < 0.05:
        return None
    return best


def _resolve_enabled_versions(cfg):
    """Map requested version toggles to the versions this run can actually execute."""
    requested = get_version_selection(cfg)
    use_airsim_ground_truth = bool(cfg.get("use_airsim_ground_truth"))
    use_airsim_source = bool(cfg.get("use_airsim_source"))
    enable_segmentation = bool(cfg.get("enable_segmentation"))

    enabled = {
        "v1": requested["v1"] and use_airsim_ground_truth,
        "v2": requested["v2"] and use_airsim_ground_truth,
        "v3": requested["v3"],
        "v4": requested["v4"] and use_airsim_source and enable_segmentation,
    }

    warnings = []
    if requested["v1"] and not enabled["v1"]:
        warnings.append("V1 requires AirSim ground-truth mode.")
    if requested["v2"] and not enabled["v2"]:
        warnings.append("V2 requires AirSim ground-truth depth.")
    if requested["v4"] and not use_airsim_source:
        warnings.append("V4 requires AirSim as the frame source.")
    if requested["v4"] and use_airsim_source and not enable_segmentation:
        warnings.append("V4 requires segmentation to be enabled.")

    if not any(enabled[key] for key in ("v2", "v3", "v4")):
        fallback = "v2" if use_airsim_ground_truth else "v3"
        enabled[fallback] = True
        warnings.append(f"No computable version was selected, so {fallback.upper()} was enabled automatically.")

    return requested, enabled, warnings


def _format_versions(version_flags):
    selected = [version_key.upper() for version_key in VERSION_ORDER if version_flags.get(version_key)]
    return ", ".join(selected) if selected else "none"

def main():
    """Main function."""
    defaults = get_default_config()

    cfg = defaults.copy()
    if defaults['show_config_ui']:
        user_cfg = launch_config_ui(defaults)
        if user_cfg:
            cfg.update(user_cfg)
    cfg = apply_version_selection(cfg)

    source = cfg['source']
    output_path = cfg['output_path']
    use_airsim_source = cfg['use_airsim_source']
    integrated_preview_ui = cfg['integrated_preview_ui']
    airsim_camera_name = cfg['airsim_camera_name']
    airsim_vehicle_name = cfg['airsim_vehicle_name']
    airsim_refresh_camera_params_every_frame = cfg['airsim_refresh_camera_params_every_frame']

    use_airsim_ground_truth = cfg['use_airsim_ground_truth']
    gt_detection_mesh_pattern = cfg['gt_detection_mesh_pattern']
    gt_detection_radius_m = cfg['gt_detection_radius_m']
    gt_use_depthplanar = cfg['gt_use_depthplanar']

    yolo_model_size = cfg['yolo_model_size']
    yolo_weights = cfg['yolo_weights']
    depth_model_size = cfg['depth_model_size']

    # Phase 2: Segmentation
    enable_segmentation = cfg.get('enable_segmentation', True)

    # Phase 2: Dataset export
    export_dataset = cfg.get('export_dataset', False)
    dataset_root = cfg.get('dataset_root', 'dataset')
    hdf5_include_segmentation = cfg.get('hdf5_include_segmentation', True)
    dataset_export = None

    # Custom weights override model size selection.
    if yolo_weights is not None:
        yolo_weights = str(yolo_weights).strip()
    if not yolo_weights:
        yolo_weights = None
    else:
        print(f"Using custom YOLO weights: {yolo_weights}. yolo_model_size will be ignored.")

    device = cfg['device']
    conf_threshold = cfg['conf_threshold']
    iou_threshold = cfg['iou_threshold']
    classes = None

    enable_tracking = cfg['enable_tracking']
    enable_bev = cfg['enable_bev']
    enable_pseudo_3d = cfg['enable_pseudo_3d']
    enable_stream = cfg['enable_stream']

    camera_params_file = cfg['camera_params_file']
    use_airsim_camera_info = cfg['use_airsim_camera_info']

    if use_airsim_ground_truth and not use_airsim_source:
        print("Ground-truth mode requires AirSim source. Enabling AirSim source automatically.")
        use_airsim_source = True
        cfg['use_airsim_source'] = True

    requested_versions, enabled_versions, version_warnings = _resolve_enabled_versions(cfg)
    enable_v2 = enabled_versions['v2']
    enable_v3 = enabled_versions['v3']
    enable_v4 = enabled_versions['v4']
    selected_versions = tuple(version_key for version_key in VERSION_ORDER if enabled_versions[version_key])
    print(f"Requested versions: {_format_versions(requested_versions)}")
    print(f"Active versions: {_format_versions(enabled_versions)}")
    for warning in version_warnings:
        print(f"Version selection: {warning}")
    # ===============================================
    
    print(f"Using device: {device}")
    
    # Initialize dataset exporter if enabled
    if export_dataset:
        try:
            dataset_export = SegmentationDatasetExporter(
                dataset_root=dataset_root,
                split_ratios={
                    'train': cfg.get('dataset_split_train', 0.7),
                    'val': cfg.get('dataset_split_val', 0.15),
                    'test': cfg.get('dataset_split_test', 0.15),
                }
            )
            print(f"Initialized dataset exporter: saving to {dataset_root}")
        except Exception as e:
            print(f"Warning: Could not initialize dataset exporter: {e}")
            export_dataset = False
    
    # Initialize models
    print("Initializing models...")
    detector = None
    need_yolo_detection = (not use_airsim_ground_truth) or enable_v3
    if need_yolo_detection:
        try:
            detector = ObjectDetector(
                model_size=yolo_model_size,
                conf_thres=conf_threshold,
                iou_thres=iou_threshold,
                classes=classes,
                device=device,
                weights_path=yolo_weights
            )
        except Exception as e:
            print(f"Error initializing object detector: {e}")
            print("Falling back to CPU for object detection")
            detector = ObjectDetector(
                model_size=yolo_model_size,
                conf_thres=conf_threshold,
                iou_thres=iou_threshold,
                classes=classes,
                device='cpu'
            )
    else:
            print("Ground-truth mode without V3 enabled: skipping YOLO detector initialization")

    need_model_depth = enable_v3 or (enable_v4 and (not use_airsim_ground_truth or not gt_use_depthplanar))
    skip_depth_model = not need_model_depth
    try:
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device,
            skip_model_init=skip_depth_model
        )
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        print("Falling back to CPU for depth estimation")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device='cpu',
            skip_model_init=skip_depth_model
        )
    
    # Initialize 3D bounding box estimator with default parameters
    # Simplified approach - focus on 2D detection with depth information
    bbox3d_estimator = BBox3DEstimator()

    # Load and apply camera extrinsics/intrinsics if a file was provided
    params = None
    world_transform = None  # dict with R + camera_center + convention when available
    if not use_airsim_source and camera_params_file is not None:
        params = load_camera_params(camera_params_file)
        bbox3d_estimator = apply_camera_params_to_estimator(bbox3d_estimator, params)
        if params is not None and 'R' in params:
            R = params['R']
            if 'camera_center' in params:
                world_transform = {
                    'R': R,
                    'camera_center': params['camera_center'],
                    'convention': 'world_to_camera'
                }
            elif 't' in params:
                # if only t (camera coords) provided, no world transform available
                world_transform = None
    
    # Initialize Bird's Eye View if enabled
    if enable_bev:
        # Use a scale that works well for the 1-5 meter range
        bev = BirdEyeView(scale=60, size=(300, 300))  # Increased scale to spread objects out
    
    cap = None
    airsim_client = None
    first_depth_frame = None

    # Open input source (AirSim or OpenCV source)
    if use_airsim_source:
        if airsim is None:
            print("Error: cosysairsim module is not available. Set use_airsim_source=False or install cosysairsim.")
            return

        print("Connecting to AirSim...")
        try:
            airsim_client = airsim.VehicleClient()
            airsim_client.confirmConnection()
        except Exception as e:
            print(f"Error: Could not connect to AirSim: {e}")
            return

        print(f"Connected to AirSim. Camera: {airsim_camera_name}, Vehicle: '{airsim_vehicle_name}'")
        if use_airsim_ground_truth:
            try:
                airsim_client.simSetDetectionFilterRadius(
                    airsim_camera_name,
                    airsim.ImageType.Scene,
                    float(gt_detection_radius_m) * 100.0,
                    airsim_vehicle_name
                )
                airsim_client.simAddDetectionFilterMeshName(
                    airsim_camera_name,
                    airsim.ImageType.Scene,
                    gt_detection_mesh_pattern,
                    airsim_vehicle_name
                )
                print(f"Configured AirSim GT detection filter: pattern='{gt_detection_mesh_pattern}', radius={gt_detection_radius_m}m")
            except Exception as e:
                print(f"Warning: Failed to configure AirSim detection filter: {e}")

        if use_airsim_ground_truth:
            first_frame, first_depth_frame = get_airsim_scene_and_depth(airsim_client, airsim_camera_name, airsim_vehicle_name, airsim)
        else:
            first_frame = get_airsim_scene_frame(airsim_client, airsim_camera_name, airsim_vehicle_name, airsim)

        if first_frame is None:
            print("Error: Could not retrieve initial scene frame from AirSim")
            return

        height, width = first_frame.shape[:2]
        fps = 30

        if use_airsim_camera_info:
            try:
                camera_info = airsim_client.simGetCameraInfo(airsim_camera_name, airsim_vehicle_name)
                params = build_camera_params_from_airsim(camera_info, width, height, airsim_utils)
                bbox3d_estimator = apply_camera_params_to_estimator(bbox3d_estimator, params)
                world_transform = {
                    'R': params['R_cam_to_world'],
                    'camera_center': params['camera_center'],
                    'convention': 'camera_to_world'
                }
                print("Applied dynamic camera parameters from AirSim camera info")
            except Exception as e:
                print(f"Warning: Failed to read AirSim camera info. Using estimator defaults: {e}")
    else:
        try:
            if isinstance(source, str) and source.isdigit():
                source = int(source)  # Convert string number to integer for webcam
        except ValueError:
            pass  # Keep as string (for video file)

        print(f"Opening video source: {source}")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:  # Sometimes happens with webcams
            fps = 30

        first_frame = None
    
    # Initialize video writer
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"

    dashboard = RuntimeDashboard(enabled=integrated_preview_ui, selected_versions=selected_versions)
    if integrated_preview_ui and not dashboard.enabled:
        print("Integrated preview UI unavailable. Run will continue without live image windows.")

    def dashboard_wants_stop():
        if not dashboard.enabled:
            return False
        dashboard.process_events()
        return dashboard.stop_requested
    
    print("Starting processing...")
    
    # Main loop
    while True:
        if dashboard_wants_stop():
            print("Dashboard requested stop.")
            break
            
        try:
            # Read frame from selected source
            if first_frame is not None:
                frame = first_frame
                first_frame = None
                frame_depth_planar = first_depth_frame
                first_depth_frame = None
            elif use_airsim_source:
                if use_airsim_ground_truth:
                    frame, frame_depth_planar = get_airsim_scene_and_depth(airsim_client, airsim_camera_name, airsim_vehicle_name, airsim)
                else:
                    frame = get_airsim_scene_frame(airsim_client, airsim_camera_name, airsim_vehicle_name, airsim)
                    frame_depth_planar = None

                if frame is None:
                    print("Warning: Empty frame from AirSim, skipping")
                    continue

                if use_airsim_camera_info and airsim_refresh_camera_params_every_frame:
                    try:
                        h_frame, w_frame = frame.shape[:2]
                        camera_info = airsim_client.simGetCameraInfo(airsim_camera_name, airsim_vehicle_name)
                        params = build_camera_params_from_airsim(camera_info, w_frame, h_frame, airsim_utils)
                        bbox3d_estimator = apply_camera_params_to_estimator(bbox3d_estimator, params)
                        world_transform = {
                            'R': params['R_cam_to_world'],
                            'camera_center': params['camera_center'],
                            'convention': 'camera_to_world'
                        }
                    except Exception as e:
                        print(f"Warning: Could not refresh AirSim camera params this frame: {e}")
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_depth_planar = None
            
            # Fetch segmentation mask if enabled
            frame_seg_mask = None
            frame_seg_rgb = None
            if use_airsim_source and enable_segmentation:
                try:
                    frame_seg_mask, frame_seg_rgb, _ = get_airsim_segmentation_mask(
                        airsim_client, airsim_camera_name, airsim_vehicle_name, airsim
                    )
                except Exception as e:
                    if frame_count == 0:  # Only warn on first frame
                        print(f"Warning: Could not fetch segmentation mask: {e}")
                    frame_seg_mask = None
                    frame_seg_rgb = None
            
            # Make copies for different visualizations
            original_frame = frame.copy()
            detection_frame = frame.copy()
            depth_frame = frame.copy()
            result_frame = frame.copy()
            v4_preview_frame = None
            if dashboard.enabled and enable_v4:
                if frame_seg_rgb is not None:
                    v4_preview_frame = cv2.cvtColor(frame_seg_rgb, cv2.COLOR_RGB2BGR)
                else:
                    v4_preview_frame = frame.copy()
                    cv2.putText(
                        v4_preview_frame,
                        "Segmentation preview unavailable",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

            if dashboard_wants_stop():
                print("Dashboard requested stop.")
                break
            
            # Step 1: Object Detection
            gt_detections = []
            yolo_detections_for_v3 = []
            detections = []
            if use_airsim_ground_truth:
                try:
                    gt_objects = airsim_client.simGetDetections(
                        airsim_camera_name,
                        airsim.ImageType.Scene,
                        airsim_vehicle_name
                    )

                    if gt_objects:
                        for idx, obj in enumerate(gt_objects):
                            x1 = float(obj.box2D.min.x_val)
                            y1 = float(obj.box2D.min.y_val)
                            x2 = float(obj.box2D.max.x_val)
                            y2 = float(obj.box2D.max.y_val)

                            if x2 <= x1 or y2 <= y1:
                                continue

                            full_object_name = str(obj.name).strip() if obj.name else "object"
                            class_name = short_airsim_class_name(full_object_name)
                            obj_id = (abs(hash(full_object_name)) % 1000000) if enable_tracking else None
                            gt_world = None
                            try:
                                
                                obj_pose = airsim_client.simGetObjectPose(class_name)
                                if obj_pose is not None and hasattr(obj_pose, 'position'):
                                    gt_world = np.array([
                                        obj_pose.position.x_val,
                                        obj_pose.position.y_val,
                                        obj_pose.position.z_val
                                    ], dtype=float)
                                    if not np.isfinite(gt_world).all():
                                        gt_world = None
                            except Exception:
                                gt_world = None

                            gt_detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'score': 1.0,
                                'class_name': class_name,
                                'object_name_full': full_object_name,
                                'object_id': obj_id,
                                'gt_world': gt_world
                            })

                            cv2.rectangle(
                                detection_frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (255, 0, 0),
                                2
                            )
                            cv2.putText(
                                detection_frame,
                                class_name,
                                (int(x1), max(0, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1
                            )
                except Exception as e:
                    print(f"Error during AirSim ground-truth detection: {e}")
                    cv2.putText(detection_frame, "GT Detection Error", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # V3 uses YOLO detection boxes even in GT mode.
                if enable_v3 and detector is not None:
                    try:
                        _, yolo_detections = detector.detect(
                            frame.copy(),
                            track=False,
                            stream=enable_stream
                        )
                        class_names = detector.get_class_names()
                        for det in yolo_detections:
                            bbox, score, class_id, obj_id = det
                            yolo_detections_for_v3.append({
                                'bbox': bbox,
                                'score': score,
                                'class_name': class_names[class_id],
                                'object_id': obj_id,
                            })
                    except Exception as e:
                        print(f"Warning: YOLO detection for V3 failed this frame: {e}")

                detections = gt_detections
            else:
                try:
                    detection_frame, yolo_detections = detector.detect(
                        detection_frame,
                        track=enable_tracking,
                        stream=enable_stream
                    )
                    class_names = detector.get_class_names()
                    for det in yolo_detections:
                        bbox, score, class_id, obj_id = det
                        detections.append({
                            'bbox': bbox,
                            'score': score,
                            'class_name': class_names[class_id],
                            'object_id': obj_id,
                            'gt_world': None
                        })
                    yolo_detections_for_v3 = detections
                except Exception as e:
                    print(f"Error during object detection: {e}")
                    cv2.putText(detection_frame, "Detection Error", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if dashboard_wants_stop():
                print("Dashboard requested stop.")
                break
            
            # Step 2: Depth sources
            depth_map_planar = None
            depth_map_model = None
            depth_map_model_is_metric = False
            depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
            try:
                if use_airsim_ground_truth and frame_depth_planar is not None:
                    depth_map_planar = frame_depth_planar.astype(np.float32)

                if need_model_depth:
                    depth_map_model = depth_estimator.estimate_depth(original_frame)
                    depth_map_model_is_metric = bool(getattr(depth_estimator, 'is_metric_depth', False))

                if use_airsim_ground_truth and gt_use_depthplanar and depth_map_planar is not None:
                    depth_colored = colorize_metric_depth(depth_map_planar, depth_estimator.colorize_depth)
                elif depth_map_model is not None:
                    depth_colored = depth_estimator.colorize_depth(depth_map_model)
                elif depth_map_planar is not None:
                    depth_colored = colorize_metric_depth(depth_map_planar, depth_estimator.colorize_depth)
            except Exception as e:
                print(f"Error during depth estimation: {e}")
                cv2.putText(depth_colored, "Depth Error", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if dashboard_wants_stop():
                print("Dashboard requested stop.")
                break

            seg_instance_ids = []
            if frame_seg_mask is not None:
                seg_instance_ids = get_present_instance_ids(frame_seg_mask, min_pixels=32)

            # Step 3: 3D Bounding Box Estimation (V1/V2/V3/V4)
            boxes_3d = []
            active_ids = []
            v2_depth_err_list = []
            v2_world_err_list = []
            v2_pixel_err_list = []
            v3_depth_err_list = []
            v3_world_err_list = []
            v3_pixel_err_list = []
            v4_depth_err_list = []
            v4_world_err_list = []
            v4_pixel_err_list = []

            for detection in detections:
                try:
                    bbox = detection['bbox']
                    score = detection['score']
                    class_name = detection['class_name']
                    obj_id = detection['object_id']
                    gt_world = detection.get('gt_world')

                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2

                    gt_cam = None
                    gt_uv = None
                    depth_gt_m = None
                    if gt_world is not None and world_transform is not None:
                        gt_cam = world_to_camera(gt_world, world_transform)
                        if gt_cam is not None:
                            gt_uv = project_camera_point(gt_cam, bbox3d_estimator.K)
                            if np.isfinite(gt_cam).all():
                                depth_gt_m = float(np.linalg.norm(gt_cam))

                    # V2: depth map (DepthPlanar) + linalg
                    v2 = None
                    if enable_v2 and depth_map_planar is not None:
                        v2 = estimate_from_depth_map(
                            depth_estimator=depth_estimator,
                            depth_map=depth_map_planar,
                            bbox=bbox,
                            class_name=class_name,
                            camera_matrix=bbox3d_estimator.K,
                            world_transform=world_transform,
                            depth_to_distance=depth_to_distance,
                            is_metric=True,
                            method_suffix="depthmap-linalg",
                        )

                    v3_detection = None
                    if use_airsim_ground_truth and enable_v3:
                        v3_detection = _match_yolo_detection(detection, yolo_detections_for_v3)

                    v3_bbox = bbox
                    v3_class_name = class_name
                    if v3_detection is not None:
                        v3_bbox = v3_detection['bbox']
                        v3_class_name = v3_detection['class_name']

                    # V3: depth model + linalg
                    v3 = None
                    if enable_v3 and depth_map_model is not None:
                        v3 = estimate_from_depth_map(
                            depth_estimator=depth_estimator,
                            depth_map=depth_map_model,
                            bbox=v3_bbox,
                            class_name=v3_class_name,
                            camera_matrix=bbox3d_estimator.K,
                            world_transform=world_transform,
                            depth_to_distance=depth_to_distance,
                            is_metric=depth_map_model_is_metric,
                            method_suffix="model-linalg",
                        )

                    # V4: segmentation mask + depth map (if enabled)
                    v4 = None
                    v4_bbox = None
                    best_instance_id = None
                    if enable_v4 and frame_seg_mask is not None:
                        try:
                            best_instance_id, best_iou, _ = find_best_instance_for_detection(
                                bbox,
                                frame_seg_mask,
                                instance_ids=seg_instance_ids,
                                iou_threshold=0.1,
                            )

                            if best_instance_id is not None:
                                # Refine bbox from segmentation mask
                                v4_bbox = refine_bbox_from_mask(frame_seg_mask, best_instance_id, detection_bbox=bbox, expand_percent=5)
                                if v4_bbox is None:
                                    v4_bbox = bbox
                                
                                # Get depth from segmentation mask
                                v4_depth_map = depth_map_planar if (use_airsim_ground_truth and gt_use_depthplanar and depth_map_planar is not None) else depth_map_model
                                
                                # Compute mean depth over mask region
                                depth_stats = compute_masked_depth_stats(v4_depth_map, frame_seg_mask, best_instance_id)
                                if depth_stats is not None:
                                    v4_bbox_for_est = v4_bbox if v4_bbox is not None else bbox
                                    v4 = estimate_from_depth_map(
                                        depth_estimator=depth_estimator,
                                        depth_map=v4_depth_map,
                                        bbox=v4_bbox_for_est,
                                        class_name=class_name,
                                        camera_matrix=bbox3d_estimator.K,
                                        world_transform=world_transform,
                                        depth_to_distance=depth_to_distance,
                                        is_metric=True if (use_airsim_ground_truth and gt_use_depthplanar) else depth_map_model_is_metric,
                                        method_suffix="seg-mask-linalg",
                                    )
                        except Exception as e:
                            if frame_count == 0:
                                print(f"Warning: V4 segmentation estimation failed: {e}")
                            v4 = None

                    if v4_preview_frame is not None:
                        cv2.rectangle(
                            v4_preview_frame,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (0, 255, 255),
                            1
                        )
                        if v4_bbox is not None:
                            cv2.rectangle(
                                v4_preview_frame,
                                (int(v4_bbox[0]), int(v4_bbox[1])),
                                (int(v4_bbox[2]), int(v4_bbox[3])),
                                (255, 0, 255),
                                2
                            )
                        v4_label = f"{class_name} V4" if best_instance_id is not None else f"{class_name} no-mask"
                        cv2.putText(
                            v4_preview_frame,
                            v4_label,
                            (int(bbox[0]), max(0, int(bbox[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1
                        )

                    def calc_errors(version_result, version_key):
                        if version_result is None:
                            return None, None, None

                        depth_err_m = None
                        world_err_m = None
                        pixel_err_px = None

                        if depth_gt_m is not None:
                            depth_err_m = abs(version_result['distance_m'] - depth_gt_m)
                            if version_key == 'v2':
                                v2_depth_err_list.append(depth_err_m)
                            elif version_key == 'v3':
                                v3_depth_err_list.append(depth_err_m)
                            elif version_key == 'v4':
                                v4_depth_err_list.append(depth_err_m)

                        if gt_world is not None and version_result['location_world'] is not None:
                            world_err_m = float(np.linalg.norm(np.asarray(version_result['location_world']) - np.asarray(gt_world)))
                            if version_key == 'v2':
                                v2_world_err_list.append(world_err_m)
                            elif version_key == 'v3':
                                v3_world_err_list.append(world_err_m)
                            elif version_key == 'v4':
                                v4_world_err_list.append(world_err_m)

                        if gt_uv is not None:
                            pixel_err_px = float(np.linalg.norm(version_result['uv_est'] - gt_uv))
                            if version_key == 'v2':
                                v2_pixel_err_list.append(pixel_err_px)
                            elif version_key == 'v3':
                                v3_pixel_err_list.append(pixel_err_px)
                            elif version_key == 'v4':
                                v4_pixel_err_list.append(pixel_err_px)

                        return depth_err_m, world_err_m, pixel_err_px

                    v2_depth_err_m, v2_world_err_m, v2_pixel_err_px = calc_errors(v2, 'v2')
                    v3_depth_err_m, v3_world_err_m, v3_pixel_err_px = calc_errors(v3, 'v3')
                    v4_depth_err_m, v4_world_err_m, v4_pixel_err_px = calc_errors(v4, 'v4')

                    version_results = {
                        'v2': v2,
                        'v3': v3,
                        'v4': v4,
                    }
                    version_errors = {
                        'v2': (v2_depth_err_m, v2_world_err_m, v2_pixel_err_px),
                        'v3': (v3_depth_err_m, v3_world_err_m, v3_pixel_err_px),
                        'v4': (v4_depth_err_m, v4_world_err_m, v4_pixel_err_px),
                    }

                    primary_key = None
                    primary = None
                    for version_key in ('v4', 'v3', 'v2'):
                        if enabled_versions.get(version_key) and version_results.get(version_key) is not None:
                            primary_key = version_key
                            primary = version_results[version_key]
                            break
                    if primary is None:
                        continue

                    primary_depth_err_m, primary_world_err_m, primary_pixel_err_px = version_errors[primary_key]
                    location_world = gt_world if gt_world is not None else primary['location_world']
                    primary_class_name = v3_class_name if primary_key == 'v3' else class_name

                    box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': primary['depth_value'],
                        'depth_unit': primary['depth_unit'],
                        'depth_method': primary['depth_method'],
                        'class_name': primary_class_name,
                        'object_id': obj_id,
                        'score': score,
                        'location_cam': primary['location_cam'],
                        'location_world': location_world,
                        'location_world_est': primary['location_world'],
                        'location_world_gt': gt_world,
                        'depth_est_m': float(primary['distance_m']),
                        'depth_gt_m': depth_gt_m,
                        'depth_error_m': primary_depth_err_m,
                        'world_error_m': primary_world_err_m,
                        'pixel_error_px': primary_pixel_err_px,
                        'uv_est': primary['uv_est'],
                        'uv_gt': gt_uv,

                        # Explicit 3-version fields for table output.
                        'v1_world_gt': gt_world,
                        'v2_depth_m': v2['distance_m'] if v2 is not None else None,
                        'v2_world': v2['location_world'] if v2 is not None else None,
                        'v2_uv': v2['uv_est'] if v2 is not None else None,
                        'v2_depth_err_m': v2_depth_err_m,
                        'v2_world_err_m': v2_world_err_m,
                        'v2_pixel_err_px': v2_pixel_err_px,
                        'v3_depth_m': v3['distance_m'] if v3 is not None else None,
                        'v3_world': v3['location_world'] if v3 is not None else None,
                        'v3_uv': v3['uv_est'] if v3 is not None else None,
                        'v3_depth_err_m': v3_depth_err_m,
                        'v3_world_err_m': v3_world_err_m,
                        'v3_pixel_err_px': v3_pixel_err_px,
                        'v4_depth_m': v4['distance_m'] if v4 is not None else None,
                        'v4_world': v4['location_world'] if v4 is not None else None,
                        'v4_uv': v4['uv_est'] if v4 is not None else None,
                        'v4_depth_err_m': v4_depth_err_m,
                        'v4_world_err_m': v4_world_err_m,
                        'v4_pixel_err_px': v4_pixel_err_px,
                    }

                    boxes_3d.append(box_3d)

                    if box_3d.get('location_world') is not None:
                        print(f"Object {class_name} id={obj_id} world coord (m): {box_3d['location_world']}")

                    if obj_id is not None:
                        active_ids.append(obj_id)
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            # Clean up trackers for objects that are no longer detected
            bbox3d_estimator.cleanup_trackers(active_ids)
            depth_estimator.cleanup_depth_history(active_ids if enable_tracking else None)
            
            # Step 4: Visualization
            # Draw boxes on the result frame
            for box_3d in boxes_3d:
                try:
                    # Determine color based on class
                    class_name = box_3d['class_name'].lower()
                    if 'car' in class_name or 'vehicle' in class_name:
                        color = (0, 0, 255)  # Red
                    elif 'person' in class_name:
                        color = (0, 255, 0)  # Green
                    elif 'bicycle' in class_name or 'motorcycle' in class_name:
                        color = (255, 0, 0)  # Blue
                    elif 'potted plant' in class_name or 'plant' in class_name:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (255, 255, 255)  # White
                    
                    # Draw box with depth information
                    result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                except Exception as e:
                    print(f"Error drawing box: {e}")
                    continue
            
            # Draw Bird's Eye View if enabled
            if enable_bev:
                try:
                    # Reset BEV and draw objects
                    bev.reset()
                    for box_3d in boxes_3d:
                        bev.draw_box(box_3d)
                    bev_image = bev.get_image()
                    
                    # Resize BEV image to fit in the corner of the result frame
                    bev_height = height // 4  # Reduced from height/3 to height/4 for better fit
                    bev_width = bev_height
                    
                    # Ensure dimensions are valid
                    if bev_height > 0 and bev_width > 0:
                        # Resize BEV image
                        bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
                        
                        # Create a region of interest in the result frame
                        roi = result_frame[height - bev_height:height, 0:bev_width]
                        
                        # Simple overlay - just copy the BEV image to the ROI
                        result_frame[height - bev_height:height, 0:bev_width] = bev_resized
                        
                        # Add a border around the BEV visualization
                        cv2.rectangle(result_frame, 
                                     (0, height - bev_height), 
                                     (bev_width, height), 
                                     (255, 255, 255), 1)
                        
                        # Add a title to the BEV visualization
                        cv2.putText(result_frame, "Bird's Eye View", 
                                   (10, height - bev_height + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Error drawing BEV: {e}")
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:  # Update FPS every 10 frames
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps_value = frame_count / elapsed_time
                fps_display = f"FPS: {fps_value:.1f}"
            
            # Add FPS and device info to the result frame
            cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            metrics_lines = []
            if v2_depth_err_list:
                metrics_lines.append(f"V2 DepthErr(m): {np.mean(v2_depth_err_list):.2f}")
            if v2_world_err_list:
                metrics_lines.append(f"V2 WorldErr(m): {np.mean(v2_world_err_list):.2f}")
            if v2_pixel_err_list:
                metrics_lines.append(f"V2 2DErr(px): {np.mean(v2_pixel_err_list):.1f}")
            if v3_depth_err_list:
                metrics_lines.append(f"V3 DepthErr(m): {np.mean(v3_depth_err_list):.2f}")
            if v3_world_err_list:
                metrics_lines.append(f"V3 WorldErr(m): {np.mean(v3_world_err_list):.2f}")
            if v3_pixel_err_list:
                metrics_lines.append(f"V3 2DErr(px): {np.mean(v3_pixel_err_list):.1f}")
            if v4_depth_err_list:
                metrics_lines.append(f"V4 DepthErr(m): {np.mean(v4_depth_err_list):.2f}")
            if v4_world_err_list:
                metrics_lines.append(f"V4 WorldErr(m): {np.mean(v4_world_err_list):.2f}")
            if v4_pixel_err_list:
                metrics_lines.append(f"V4 2DErr(px): {np.mean(v4_pixel_err_list):.1f}")

            def fmt_float(value, digits=2):
                if value is None:
                    return "--"
                try:
                    v = float(value)
                    if not np.isfinite(v):
                        return "--"
                    return f"{v:.{digits}f}"
                except Exception:
                    return "--"

            def fmt_vec2(value):
                if value is None:
                    return "--"
                try:
                    arr = np.asarray(value, dtype=float).reshape(-1)
                    if arr.size < 2 or not np.isfinite(arr[:2]).all():
                        return "--"
                    return f"({arr[0]:.1f},{arr[1]:.1f})"
                except Exception:
                    return "--"

            def fmt_vec3(value):
                if value is None:
                    return "--"
                try:
                    arr = np.asarray(value, dtype=float).reshape(-1)
                    if arr.size < 3 or not np.isfinite(arr[:3]).all():
                        return "--"
                    return f"({arr[0]:.2f},{arr[1]:.2f},{arr[2]:.2f})"
                except Exception:
                    return "--"

            object_rows = []
            for box_3d in boxes_3d:
                obj_id = box_3d.get('object_id')
                obj_name = box_3d.get('class_name', 'object')
                obj_label = f"{obj_name}#{obj_id}" if obj_id is not None else str(obj_name)
                object_rows.append((
                    obj_label,
                    fmt_vec3(box_3d.get('v1_world_gt')),
                    fmt_float(box_3d.get('v2_depth_m')),
                    fmt_vec3(box_3d.get('v2_world')),
                    fmt_vec2(box_3d.get('v2_uv')),
                    fmt_float(box_3d.get('v3_depth_m')),
                    fmt_vec3(box_3d.get('v3_world')),
                    fmt_vec2(box_3d.get('v3_uv')),
                    fmt_float(box_3d.get('v4_depth_m')),
                    fmt_vec3(box_3d.get('v4_world')),
                    fmt_vec2(box_3d.get('v4_uv')),
                ))

            # Keep dashboard metrics visibly live even when GT error lists are empty.
            dashboard_metrics_lines = [
                f"Frame: {frame_count}",
                f"Detections: {len(detections)} | 3D: {len(boxes_3d)}"
            ]
            dashboard_metrics_lines.extend(metrics_lines)

            for i, line in enumerate(metrics_lines):
                cv2.putText(
                    result_frame,
                    line,
                    (10, 55 + (i * 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )
            
            
            # Export dataset if enabled (Phase 2)
            if export_dataset and dataset_export is not None:
                try:
                    # Prepare metadata for this frame
                    frame_metadata = {}
                    for det in detections:
                        class_name = det.get('class_name', 'unknown')
                        if class_name not in frame_metadata:
                            frame_metadata[class_name] = []
                        frame_metadata[class_name].append({
                            'bbox': det.get('bbox'),
                            'confidence': det.get('score', 1.0),
                            'object_name': det.get('object_name_full', class_name)
                        })
                    
                    # Save frame with all data
                    dataset_export.save_frame(
                        frame_id=frame_count,
                        rgb_image=cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB),
                        seg_mask=frame_seg_mask if hdf5_include_segmentation else None,
                        segmentation_rgb=frame_seg_rgb if hdf5_include_segmentation else None,
                        depth_map=depth_map_planar if (use_airsim_ground_truth and gt_use_depthplanar) else depth_map_model,
                        metadata=frame_metadata,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    # Save camera info on first frame
                    if frame_count == 1:
                        try:
                            dataset_export.save_camera_info(
                                camera_matrix=bbox3d_estimator.K,
                                img_shape=(height, width)
                            )
                            # Create YOLO dataset.yaml
                            class_names = list(set([d.get('class_name', 'unknown') for d in detections]))
                            dataset_export.create_dataset_yaml(
                                class_names=sorted(class_names),
                                num_classes=len(class_names)
                            )
                        except Exception as e:
                            print(f"Warning: Could not save camera info or dataset.yaml: {e}")
                except Exception as e:
                    if frame_count <= 2:  # Only warn on first couple frames
                        print(f"Warning: Dataset export failed: {e}")
            
            # Write frame to output video
            out.write(result_frame)

            status_text = (
                f"Source={'AirSim' if use_airsim_source else 'Video'} | "
                f"Mode={'GT' if use_airsim_ground_truth else 'YOLO'} | "
                f"Versions={_format_versions(enabled_versions)}"
            )
            metrics_text = "Metrics: " + (" | ".join(dashboard_metrics_lines) if dashboard_metrics_lines else "--")
            dashboard.update(
                result_frame=result_frame,
                detection_frame=detection_frame,
                depth_frame=depth_colored,
                v4_frame=v4_preview_frame,
                status_text=status_text,
                metrics_text=metrics_text,
                object_rows=object_rows
            )
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
    
    # Clean up
    print("Cleaning up resources...")
    if cap is not None:
        cap.release()
    dashboard.close()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        # Clean up OpenCV windows
        cv2.destroyAllWindows() 
