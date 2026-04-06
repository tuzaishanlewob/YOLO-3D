from detection_model import ObjectDetector
from depth_model import DepthEstimator


def init_detector(model_size, conf_threshold, iou_threshold, classes, device, weights_path=None):
    """Initialize detector with CPU fallback."""
    try:
        return ObjectDetector(
            model_size=model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device=device,
            weights_path=weights_path,
        )
    except Exception as exc:
        print(f"Error initializing object detector: {exc}")
        print("Falling back to CPU for object detection")
        return ObjectDetector(
            model_size=model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device="cpu",
            weights_path=weights_path,
        )


def init_depth_estimator(model_size, device, weights_path=None):
    """Initialize depth estimator with CPU fallback."""
    try:
        return DepthEstimator(model_size=model_size, device=device, weights_path=weights_path)
    except Exception as exc:
        print(f"Error initializing depth estimator: {exc}")
        print("Falling back to CPU for depth estimation")
        return DepthEstimator(model_size=model_size, device="cpu", weights_path=weights_path)
