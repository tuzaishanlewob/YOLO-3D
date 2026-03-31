"""
Segmentation utilities for instance-aware object detection.
Handles:
- Fetching segmentation masks from AirSim
- Extracting instances from masks
- Computing instance bboxes and properties
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


def match_segmentation_to_detection(detection_bbox, seg_mask, instance_id, iou_threshold=0.3):
    """
    Check if a segmentation instance overlaps sufficiently with a detection bbox.
    
    Args:
        detection_bbox [x1, y1, x2, y2]: Detection bounding box.
        seg_mask (numpy array): Instance ID mask.
        instance_id (int): Instance ID to check.
        iou_threshold (float): Minimum IoU required.
    
    Returns:
        iou (float) or None if below threshold.
    """
    seg_bbox = extract_instance_bbox(seg_mask, instance_id)
    if seg_bbox is None:
        return None
    
    # Compute IoU between detection_bbox and seg_bbox
    ax1, ay1, ax2, ay2 = [float(v) for v in detection_bbox]
    bx1, by1, bx2, by2 = [float(v) for v in seg_bbox]
    
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
        return None
    
    iou = inter / union
    return iou if iou >= iou_threshold else None


def refine_bbox_from_mask(seg_mask, instance_id, detection_bbox=None, expand_percent=10):
    """
    Refine a detection bbox using the segmentation mask.
    
    Args:
        seg_mask (numpy array): Instance ID mask.
        instance_id (int): Instance to refine.
        detection_bbox [x1,y1,x2,y2]: Original detection (optional, for padding).
        expand_percent (float): Percentage to expand mask bbox.
    
    Returns:
        Refined bbox [x1, y1, x2, y2] or None.
    """
    mask_bbox = extract_instance_bbox(seg_mask, instance_id)
    if mask_bbox is None:
        return None
    
    x1, y1, x2, y2 = mask_bbox
    w = x2 - x1
    h = y2 - y1
    
    # Optionally expand by percentage
    if expand_percent > 0:
        exp_x = w * expand_percent / 100.0
        exp_y = h * expand_percent / 100.0
        x1 -= exp_x
        y1 -= exp_y
        x2 += exp_x
        y2 += exp_y
        
    # Clamp to image bounds.
    h_img, w_img = seg_mask.shape[:2]
    x1 = max(0.0, min(float(x1), float(max(0, w_img - 1))))
    y1 = max(0.0, min(float(y1), float(max(0, h_img - 1))))
    x2 = max(x1 + 1.0, min(float(x2), float(w_img)))
    y2 = max(y1 + 1.0, min(float(y2), float(h_img)))

    return [float(x1), float(y1), float(x2), float(y2)]


def find_best_instance_for_detection(detection_bbox, seg_mask, instance_ids=None, iou_threshold=0.1):
    """
    Find the best segmentation instance for a detection bbox.

    Returns:
        tuple: (instance_id, iou, instance_bbox)
    """
    if seg_mask is None:
        return None, 0.0, None

    if instance_ids is None:
        instance_ids = get_present_instance_ids(seg_mask)

    best_instance_id = None
    best_iou = 0.0
    best_bbox = None

    for instance_id in instance_ids:
        iou = match_segmentation_to_detection(
            detection_bbox,
            seg_mask,
            instance_id,
            iou_threshold=iou_threshold,
        )
        if iou is None or iou <= best_iou:
            continue
        best_instance_id = int(instance_id)
        best_iou = float(iou)
        best_bbox = extract_instance_bbox(seg_mask, instance_id)

    return best_instance_id, best_iou, best_bbox


def get_instance_visibility_score(seg_mask, instance_id, bbox=None):
    """
    Compute visibility as fraction of bbox pixels that are this instance.
    
    Args:
        seg_mask (numpy array): Instance ID mask.
        instance_id (int): Instance to analyze.
        bbox [x1,y1,x2,y2]: Region to analyze (default: whole image).
    
    Returns:
        visibility (float in [0, 1]).
    """
    if seg_mask is None:
        return 0.0
    
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(seg_mask.shape[1], x2)
        y2 = min(seg_mask.shape[0], y2)
        region = seg_mask[y1:y2, x1:x2]
        total_pixels = (x2 - x1) * (y2 - y1)
    else:
        region = seg_mask
        total_pixels = seg_mask.size
    
    if total_pixels == 0:
        return 0.0
    
    instance_pixels = np.sum(region == instance_id)
    return float(instance_pixels) / total_pixels
