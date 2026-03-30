def get_default_config():
    """Centralized default configuration for YOLO-3D runtime."""
    return {
        # UI/runtime controls
        "show_config_ui": True,
        "integrated_preview_ui": True,
        "show_cv2_windows": False,

        # Input/Output
        "source": "0",
        "output_path": "output.mp4",
        "use_airsim_source": True,
        "airsim_camera_name": "0",
        "airsim_vehicle_name": "",
        "airsim_refresh_camera_params_every_frame": False,

        # Ground-truth AirSim mode
        "use_airsim_ground_truth": True,
        "gt_detection_mesh_pattern": "Cylinder*",
        "gt_detection_radius_m": 200.0,
        "gt_use_depthplanar": True,
        "compare_three_versions": True,

        # Model settings
        "yolo_model_size": "nano",
        "yolo_weights": r"E:\Programs\AirSim\Cosys-AirSim\runs\detect\train9\weights\best.pt",
        "depth_model_size": "outdoor",

        # Device settings
        "device": 0,

        # Detection settings
        "conf_threshold": 0.5,
        "iou_threshold": 0.45,

        # Feature toggles
        "enable_tracking": True,
        "enable_bev": True,
        "enable_pseudo_3d": True,
        "enable_stream": True,

        # Camera parameters
        "camera_params_file": "cam.json",
        "use_airsim_camera_info": True,
    }
