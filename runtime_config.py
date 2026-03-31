def get_default_config():
    """Centralized default configuration for YOLO-3D runtime."""
    return {
        # UI/runtime controls
        "show_config_ui": True,
        "integrated_preview_ui": True,

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
        "enable_v1": True,
        "enable_v2": True,
        "enable_v3": True,
        "enable_v4": False,
        "compare_three_versions": True,  # Legacy alias for V3

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

        # Segmentation (Phase 2)
        "enable_segmentation": True,
        "compare_four_versions": False,  # Legacy alias for V4

        # Dataset export (Phase 2)
        "export_dataset": False,
        "dataset_root": "dataset",
        "hdf5_include_segmentation": True,
        "dataset_split_train": 0.7,
        "dataset_split_val": 0.15,
        "dataset_split_test": 0.15,

        # Camera parameters
        "camera_params_file": "cam.json",
        "use_airsim_camera_info": True,
    }
