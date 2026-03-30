import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image

class DepthEstimator:
    """
    Depth estimation using Depth Anything v2
    """
    def __init__(self, model_size='small', device=None, skip_model_init=False):
        """
        Initialize the depth estimator
        
        Args:
            model_size (str): Model size ('small', 'base', 'large')
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
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
        
        model_name = model_map.get(model_size.lower(), model_map['small'])
        self.model_size = str(model_size).lower()
        self.is_metric_depth = self.model_size in ('indoor', 'outdoor')
        
        self.pipe = None
        if not skip_model_init:
            # Create pipeline
            try:
                self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
                print(f"Loaded Depth Anything v2 {model_size} model on {self.pipe_device}")
            except Exception as e:
                # Fallback to CPU if there are issues
                print(f"Error loading model on {self.pipe_device}: {e}")
                print("Falling back to CPU for depth estimation")
                self.pipe_device = 'cpu'
                self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
                print(f"Loaded Depth Anything v2 {model_size} model on CPU (fallback)")
        else:
            print("Skipping depth model pipeline initialization (external depth source mode)")

        # Temporal smoothing state keyed by tracked object id
        self.depth_history = {}
        self.depth_smoothing_alpha = 0.35
        self.max_missing_frames = 30
        self.frame_counter = 0

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

    def estimate_object_depth(self, depth_map, bbox, class_name='', object_id=None):
        """
        Estimate object depth robustly from a detection bbox.

        Returns:
            tuple: (depth_value, method_tag)
        """
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
    
    def estimate_depth(self, image):
        """
        Estimate depth from an image
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            
        Returns:
            numpy.ndarray: Depth map (normalized to 0-1)
        """
        if self.pipe is None:
            raise RuntimeError("Depth estimation pipeline is not initialized. Set skip_model_init=False to use model inference.")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Get depth map
        try:
            depth_result = self.pipe(pil_image)
            depth_map = depth_result["depth"]
            
            # Convert PIL Image to numpy array if needed
            if isinstance(depth_map, Image.Image):
                depth_map = np.array(depth_map)
            elif isinstance(depth_map, torch.Tensor):
                depth_map = depth_map.cpu().numpy()
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
            else:
                # Re-raise the error if not MPS
                raise
        
        # Normalize depth map to 0-1
        if not self.is_metric_depth:
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map
    
    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """
        Colorize depth map for visualization
        
        Args:
            depth_map (numpy.ndarray): Depth map (normalized to 0-1)
            cmap (int): OpenCV colormap
            
        Returns:
            numpy.ndarray: Colorized depth map (BGR format)
        """
        # Metric-depth models (indoor/outdoor) output meters; normalize for display only.
        if self.is_metric_depth or np.nanmax(depth_map) > 1.5:
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
    
    def get_depth_in_region(self, depth_map, bbox, method='median'):
        """
        Get depth value in a region defined by a bounding box
        
        Args:
            depth_map (numpy.ndarray): Depth map
            bbox (list): Bounding box [x1, y1, x2, y2]
            method (str): Method to compute depth ('median', 'mean', 'min')
            
        Returns:
            float: Depth value in the region
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)
        
        # Extract region
        region = depth_map[y1:y2, x1:x2]
        valid_region = self._valid_depth_values(region)
        
        if valid_region.size == 0:
            return 0.0
        
        # Compute depth based on method
        if method == 'median':
            return float(np.median(valid_region))
        elif method == 'mean':
            return float(np.mean(valid_region))
        elif method == 'min':
            return float(np.min(valid_region))
        else:
            return float(np.median(valid_region))