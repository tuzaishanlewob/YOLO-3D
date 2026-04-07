import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image

# Optional local raw-weights model support.
# Expects a local implementation such as:
#   from depth_anything_v2.dpt import DepthAnythingV2
try:
    from depth_anything_v2.dpt import DepthAnythingV2#type:ignore
except Exception:
    DepthAnythingV2 = None


class DepthEstimator:
    """
    Depth estimation or extraction

    Supports:
    - Hugging Face pipeline models (default)
    - Raw .pth/.pt checkpoints via local DepthAnythingV2 class
    """

    def __init__(self, model_size='outdoor', device=None, skip_model_init=False, weights_path=None):
        """
        Initialize the depth estimator

        Args:
            model_size (str): Model size ('small', 'base', 'large', 'indoor', 'outdoor')
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
            skip_model_init (bool): If True, skip model/pipeline init (external depth source mode)
            weights_path (str|None): Optional custom model path.
                - If endswith .pth/.pt => load raw checkpoint into local DepthAnythingV2.
                - Else => used as HF model id/path for transformers pipeline.
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        self.weights_path = weights_path
        self.pipe = None
        self.raw_model = None
        self.use_raw_model = False

        # Set MPS fallback for operations not supported on Apple Silicon
        if self.device == 'mps':
            print("Using MPS device with CPU fallback for unsupported operations")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            # For Depth Anything v2, we'll use CPU directly due to MPS compatibility issues
            self.pipe_device = 'cpu'
            print("Forcing CPU for depth estimation pipeline due to MPS compatibility issues")
        else:
            self.pipe_device = self.device

        print(f"Using device: {self.device} for depth estimation (pipeline on {self.pipe_device})")

        # Map model size to model name
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf',
            'indoor': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
            'outdoor': 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf'
        }

        self.model_size = str(model_size).lower()
        self.is_metric_depth = self.model_size in ('indoor', 'outdoor')


        encoder = 'vits' # or 'vits', 'vitb'
        dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
        max_depth = 200 # 20 for indoor model, 80 for outdoor model

        if not skip_model_init:
            # Branch 1: raw .pth/.pt weights
            if weights_path and str(weights_path).lower().endswith(('.pth', '.pt')):
                self._init_raw_model(weights_path, model_size)
            else:
                # Branch 2: transformers pipeline (default / custom HF path)
                model_name = weights_path if weights_path else model_map.get(self.model_size, model_map['small'])
                self._init_pipeline_model(model_name)
        else:
            print("Skipping depth model pipeline initialization (external depth source mode)")

        # Temporal smoothing state keyed by tracked object id
        self.depth_history = {}
        self.depth_smoothing_alpha = 0.35
        self.max_missing_frames = 30
        self.frame_counter = 0

    def _init_raw_model(self, weights_path, encoder):
        """Initialize local model from raw .pth/.pt checkpoint."""
        if DepthAnythingV2 is None:
            raise ImportError(
                "Raw .pth/.pt loading requested, but DepthAnythingV2 is not importable. "
                "Make sure local code exists and is importable as "
                "`from depth_anything_v2.dpt import DepthAnythingV2`."
            )
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        encoder_aliases = {
            'small': 'vits',
            'base': 'vitb',
            'large': 'vitl',
            'indoor': 'vits',
            'outdoor': 'vits',
        }

        encoder_key = str(encoder).lower()
        if encoder_key in encoder_aliases:
            encoder_key = encoder_aliases[encoder_key]
        if encoder_key not in model_configs:
            print(f"Warning: Unknown raw depth encoder '{encoder}', defaulting to 'vits'")
            encoder_key = 'vits'

        max_depth = 80
        print(f"Loading raw checkpoint: {weights_path} (encoder={encoder_key})")
        model = DepthAnythingV2(**{**model_configs[encoder_key], 'max_depth': max_depth})
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model.eval()
        model = model.to(self.device)
        model.eval()

        self.raw_model = model
        self.use_raw_model = True
        print(f"Loaded raw depth model on {self.device}")

    def _init_pipeline_model(self, model_name):
        """Initialize Hugging Face depth-estimation pipeline."""
        try:
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            print(f"Loaded Depth Anything v2 model ({model_name}) on {self.pipe_device}")
        except Exception as e:
            # Fallback to CPU if there are issues
            print(f"Error loading model on {self.pipe_device}: {e}")
            print("Falling back to CPU for depth estimation")
            self.pipe_device = 'cpu'
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            print(f"Loaded Depth Anything v2 model ({model_name}) on CPU (fallback)")

    def _clip_bbox(self, bbox, shape):
        """Clip bbox to image bounds and return integer coordinates."""
        h, w = shape[:2]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _valid_depth_values(self, values):
        """Filter invalid depth values from a region."""
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        return arr[np.isfinite(arr) & (arr > 0)]

    def _robust_percentile(self, values, percentile):
        """Compute percentile after IQR-based outlier rejection."""
        if values.size == 0:
            return None

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        if iqr > 1e-6:
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            filtered = values[(values >= low) & (values <= high)]
            if filtered.size > 0:
                values = filtered

        return float(np.percentile(values, percentile))

    def _smooth_depth(self, object_id, depth_value):
        """Apply EMA smoothing for tracked objects."""
        if object_id is None:
            return float(depth_value), False

        obj_id = int(object_id)
        if obj_id in self.depth_history:
            prev = self.depth_history[obj_id]['depth']
            smoothed = self.depth_smoothing_alpha * float(depth_value) + (1.0 - self.depth_smoothing_alpha) * prev
        else:
            smoothed = float(depth_value)

        self.depth_history[obj_id] = {
            'depth': smoothed,
            'last_seen': self.frame_counter
        }
        return smoothed, True

    def cleanup_depth_history(self, active_ids=None):
        """Advance frame index and remove stale track history."""
        self.frame_counter += 1

        active_set = None
        if active_ids is not None:
            active_set = {int(obj_id) for obj_id in active_ids}

        stale_ids = []
        for obj_id, state in self.depth_history.items():
            too_old = (self.frame_counter - state['last_seen']) > self.max_missing_frames
            not_active = active_set is not None and obj_id not in active_set
            if too_old or not_active:
                stale_ids.append(obj_id)

        for obj_id in stale_ids:
            del self.depth_history[obj_id]

    def estimate_object_depth(self, depth_map, bbox, class_name='', object_id=None, center_x=None, center_y=None):
        # To-do: what is the use of class name? no use. delete it
        """
        Estimate object depth robustly from a detection bbox or specific point.

        Args:
            depth_map (np.ndarray): Depth map array.
            bbox (list/tuple): [x1, y1, x2, y2] bounding box.
            class_name (str): Object class name for percentile tuning.
            object_id (int|None): Tracking ID for temporal smoothing.
            center_x (float|None): Optional x-coordinate to sample depth directly.
            center_y (float|None): Optional y-coordinate to sample depth directly.
                If both center_x and center_y are provided, depth is taken at that point
                and bbox-based ROI estimation is skipped.

        Returns:
            tuple: (depth_value, method_tag)
        """
        # If explicit center point is provided, use it directly
        if center_x is not None and center_y is not None:
            cx = int(round(center_x))
            cy = int(round(center_y))
            depth_value = self.get_depth_at_point(depth_map, cx, cy)
            if np.isfinite(depth_value) and depth_value > 0:
                depth_value, smoothed = self._smooth_depth(object_id, depth_value)
                method = 'point-sample'
                if smoothed and object_id is not None:
                    method += '+ema'
                return float(depth_value), method
            # Fallback to bbox if point is invalid
            return 0.0, 'point-invalid'

        # Original bbox-based estimation
        clipped = self._clip_bbox(bbox, depth_map.shape)
        if clipped is None:
            return 0.0, 'invalid-bbox'

        x1, y1, x2, y2 = clipped
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        center_depth = self.get_depth_at_point(depth_map, cx, cy)

        roi = depth_map[y1:y2, x1:x2]
        valid_roi = self._valid_depth_values(roi)
        if valid_roi.size == 0:
            return float(center_depth), 'center-empty-roi'

        w = x2 - x1
        h = y2 - y1
        inner_scale = 0.6
        half_w = max(1, int((w * inner_scale) / 2))
        half_h = max(1, int((h * inner_scale) / 2))
        inner_x1 = max(x1, cx - half_w)
        inner_x2 = min(x2, cx + half_w)
        inner_y1 = max(y1, cy - half_h)
        inner_y2 = min(y2, cy + half_h)
        inner_roi = depth_map[inner_y1:inner_y2, inner_x1:inner_x2]
        valid_inner = self._valid_depth_values(inner_roi)

        class_lower = str(class_name).lower()
        if class_lower in ['person', 'cat', 'dog']:
            inner_percentile = 30
            roi_percentile = 35
        else:
            inner_percentile = 35
            roi_percentile = 40

        depth_candidates = []
        method_parts = []

        inner_depth = self._robust_percentile(valid_inner, inner_percentile) if valid_inner.size >= 10 else None
        if inner_depth is not None:
            depth_candidates.append((0.7, inner_depth))
            method_parts.append(f'inner-p{inner_percentile}')

        roi_depth = self._robust_percentile(valid_roi, roi_percentile)
        if roi_depth is not None:
            depth_candidates.append((0.3 if inner_depth is not None else 1.0, roi_depth))
            method_parts.append(f'roi-p{roi_percentile}')

        if depth_candidates:
            weight_sum = sum(weight for weight, _ in depth_candidates)
            depth_value = sum(weight * value for weight, value in depth_candidates) / weight_sum
            depth_method = '+'.join(method_parts)
        else:
            depth_value = float(center_depth)
            depth_method = 'center-fallback'

        if not np.isfinite(depth_value) or depth_value <= 0:
            depth_value = float(center_depth)
            depth_method = 'center-invalid-fallback'

        depth_value, smoothed = self._smooth_depth(object_id, depth_value)
        if smoothed and object_id is not None:
            depth_method += '+ema'

        return float(depth_value), depth_method

    def _estimate_depth_pipeline(self, image_rgb):
        """Inference through transformers pipeline."""
        pil_image = Image.fromarray(image_rgb)

        try:
            depth_result = self.pipe(pil_image)
            depth_map = depth_result["depth"]

            # Convert PIL Image to numpy array if needed
            if isinstance(depth_map, Image.Image):
                depth_map = np.array(depth_map)
            elif isinstance(depth_map, torch.Tensor):
                depth_map = depth_map.cpu().numpy()

            return depth_map
        except RuntimeError as e:
            # Handle potential MPS errors during inference
            if self.device == 'mps':
                print(f"MPS error during depth estimation: {e}")
                print("Temporarily falling back to CPU for this frame")
                # Create a CPU pipeline for this frame
                cpu_pipe = pipeline(task="depth-estimation", model=self.pipe.model.config._name_or_path, device='cpu')
                depth_result = cpu_pipe(pil_image)
                depth_map = depth_result["depth"]

                # Convert PIL Image to numpy array if needed
                if isinstance(depth_map, Image.Image):
                    depth_map = np.array(depth_map)
                elif isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
                return depth_map
            else:
                # Re-raise the error if not MPS
                raise

    def _estimate_depth_raw(self, image_rgb):
        """Inference through local raw model loaded from .pth/.pt."""
        # Preferred path for DepthAnythingV2 implementations
        if hasattr(self.raw_model, "infer_image"):
            depth = self.raw_model.infer_image(image_rgb)
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
            return depth

        # Generic fallback if infer_image is not available
        img = image_rgb.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.raw_model(img)

        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, torch.Tensor):
            out = out.squeeze().detach().cpu().numpy()

        return out

    def estimate_depth(self, image):
        """
        Estimate depth from an image

        Args:
            image (numpy.ndarray): Input image (BGR format)

        Returns:
            numpy.ndarray: Depth map
              - normalized to 0-1 for relative models
              - metric (meters) for metric models ('indoor', 'outdoor') and most raw checkpoints
        """
        if self.pipe is None and not self.use_raw_model:
            raise RuntimeError(
                "Depth estimator is not initialized. "
                "Set skip_model_init=False and/or provide valid weights_path."
            )

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get depth map from selected backend
        if self.use_raw_model:
            depth_map = self._estimate_depth_raw(image_rgb)
        else:
            depth_map = self._estimate_depth_pipeline(image_rgb)

        depth_map = np.asarray(depth_map, dtype=np.float32)

        # Normalize only for non-metric HF relative depth models.
        # Raw checkpoints are treated as metric/absolute-like by default (no forced normalization).
        if (not self.is_metric_depth) and (not self.use_raw_model):
            depth_min = np.nanmin(depth_map)
            depth_max = np.nanmax(depth_map)
            if np.isfinite(depth_min) and np.isfinite(depth_max) and depth_max > depth_min:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        return depth_map

    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """
        Colorize depth map for visualization

        Args:
            depth_map (numpy.ndarray): Depth map (normalized or metric)
            cmap (int): OpenCV colormap

        Returns:
            numpy.ndarray: Colorized depth map (BGR format)
        """
        # Metric-depth models (indoor/outdoor/raw) output meters; normalize for display only.
        if self.is_metric_depth or self.use_raw_model or np.nanmax(depth_map) > 1.5:
            valid = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
            if valid.size == 0:
                depth_map_uint8 = np.zeros(depth_map.shape, dtype=np.uint8)
            else:
                lo = np.percentile(valid, 5)
                hi = max(np.percentile(valid, 95), lo + 1e-3)
                disp = np.clip((depth_map - lo) / (hi - lo), 0.0, 1.0)
                depth_map_uint8 = (disp * 255).astype(np.uint8)
        else:
            depth_map_uint8 = (depth_map * 255).astype(np.uint8)

        colored_depth = cv2.applyColorMap(depth_map_uint8, cmap)
        return colored_depth

    def get_depth_at_point(self, depth_map, x, y):
        """
        Get depth value at a specific point

        Args:
            depth_map (numpy.ndarray): Depth map
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            float: Depth value at (x, y)
        """
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[y, x]
        return 0.0

