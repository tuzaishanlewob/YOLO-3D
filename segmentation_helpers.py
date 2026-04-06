"""
Segmentation utilities for instance-aware object detection.
Handles:
- Fetching segmentation masks from AirSim
- Extracting instances from masks
- Computing depth statistics over masked regions
"""

import numpy as np
import cv2


def get_airsim_segmentation_mask(airsim_client, camera_name, vehicle_name, airsim_module):
    """
    Fetch instance segmentation mask from AirSim.
    
    Returns:
        seg_mask (numpy array, uint32): Instance ID at each pixel.
        seg_rgb (numpy array, uint8): Raw segmentation RGB image.
        object_list (list): List of object names in scene.
    """
    try:
        responses = airsim_client.simGetImages([
            airsim_module.ImageRequest(camera_name, airsim_module.ImageType.Segmentation, False, False)
        ], vehicle_name)
        if not responses:
            return None, None, []

        resp = responses[0]
        if resp is None or not resp.image_data_uint8 or resp.width <= 0 or resp.height <= 0:
            return None, None, []

        expected = int(resp.width) * int(resp.height) * 3
        seg_flat = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        if seg_flat.size < expected:
            return None, None, []

        seg_rgb = seg_flat[:expected].reshape((int(resp.height), int(resp.width), 3))

        # Convert 24-bit color to packed instance id.
        seg_mask = (
            seg_rgb[:, :, 0].astype(np.uint32)
            + (seg_rgb[:, :, 1].astype(np.uint32) << 8)
            + (seg_rgb[:, :, 2].astype(np.uint32) << 16)
        )

        try:
            object_list = airsim_client.simListInstanceSegmentationObjects()
        except Exception:
            object_list = []

        return seg_mask, seg_rgb, object_list
    except Exception as e:
        print(f"Error fetching segmentation mask: {e}")
        return None, None, []


def get_present_instance_ids(seg_mask, min_pixels=32):
    """Return instance IDs present in this frame above a small area threshold."""
    if seg_mask is None:
        return []

    ids, counts = np.unique(seg_mask, return_counts=True)
    present = []
    for instance_id, count in zip(ids.tolist(), counts.tolist()):
        if int(instance_id) == 0:
            continue
        if int(count) < int(min_pixels):
            continue
        present.append(int(instance_id))
    return present


def extract_instance_bbox(seg_mask, instance_id):
    """
    Extract bounding box from instance mask.
    
    Args:
        seg_mask (numpy array): Instance ID mask.
        instance_id (int): ID to extract.
    
    Returns:
        bbox [x1, y1, x2, y2] or None if not found.
    """
    if seg_mask is None:
        return None
    
    mask = (seg_mask == instance_id).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    
    if coords is None or len(coords) == 0:
        return None
    
    x1, y1, w, h = cv2.boundingRect(coords)
    x2, y2 = x1 + w, y1 + h
    
    return [float(x1), float(y1), float(x2), float(y2)]


def compute_masked_depth_stats(depth_map, seg_mask, instance_id):
    """
    Compute depth statistics for a masked region.
    
    Args:
        depth_map (numpy array): Depth value at each pixel.
        seg_mask (numpy array): Instance ID mask.
        instance_id (int): ID to analyze.
    
    Returns:
        dict with keys: mean, median, std, min, max, valid_pixels
    """
    if depth_map is None or seg_mask is None:
        return None
    
    mask = (seg_mask == instance_id).astype(np.uint8)
    masked_depth = depth_map[mask > 0]
    
    if len(masked_depth) == 0:
        return None
    
    # Filter invalid depths (0, nan, inf)
    valid_depth = masked_depth[np.isfinite(masked_depth) & (masked_depth > 0)]
    
    if len(valid_depth) == 0:
        return None
    
    return {
        'mean': float(np.mean(valid_depth)),
        'median': float(np.median(valid_depth)),
        'std': float(np.std(valid_depth)),
        'min': float(np.min(valid_depth)),
        'max': float(np.max(valid_depth)),
        'valid_pixels': int(len(valid_depth))
    }

