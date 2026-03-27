import os
import setup_path 
import sys
import time
import math
import cv2
import numpy as np
import torch
from pathlib import Path

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

def main():
    """Main function."""
    # Configuration variables (modify these as needed)
    # ===============================================
    
    # Input/Output
    source = 0  # Path to input video file or webcam index (0 for default camera)
    output_path = "output.mp4"  # Path to output video file
    use_airsim_source = True  # If True, pull frames from AirSim instead of OpenCV VideoCapture
    airsim_camera_name = "0"  # AirSim camera name/id
    airsim_vehicle_name = ""  # AirSim vehicle name (empty for default)
    airsim_refresh_camera_params_every_frame = False  # Update intrinsics/extrinsics from AirSim each frame
    
    # Model settings
    yolo_model_size = "nano"  # YOLOv11 model size: "nano", "small", "medium", "large", "extra"
    yolo_weights = r"E:\Programs\AirSim\Cosys-AirSim\runs\detect\train9\weights\best.pt"         # Path to your custom .pt file or model ID (None to use pretrained size above)
    depth_model_size = "small"  # Depth Anything v2 model size: "small", "base", "large"
    
    # Device settings
    device = 0  # Force CPU for stability
    
    # Detection settings
    conf_threshold = 0.5  # Confidence threshold for object detection
    iou_threshold = 0.45  # IoU threshold for NMS
    classes = None  # Filter by class, e.g., [0, 1, 2] for specific classes, None for all classes
    
    # Feature toggles
    enable_tracking = True  # Enable object tracking
    enable_bev = True  # Enable Bird's Eye View visualization
    enable_pseudo_3d = True  # Enable pseudo-3D visualization
    enable_stream = True  # Use streaming mode for detector to lower memory usage
    
    # Camera parameters - simplified approach
    camera_params_file = "cam.json"  # Path to camera parameters file (None to use default parameters)
    use_airsim_camera_info = True  # In AirSim mode, derive K/P/R/t from simGetCameraInfo instead of cam.json
    # ===============================================
    
    print(f"Using device: {device}")

    def get_airsim_scene_frame(client, camera_name, vehicle_name=""):
        """Fetch one RGB scene frame from AirSim and decode it to BGR."""
        responses = client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, True)
        ], vehicle_name)

        if not responses:
            return None

        response = responses[0]
        if response is None or not response.image_data_uint8:
            return None

        frame_buffer = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        if frame_buffer.size == 0:
            return None

        return cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)

    def build_camera_params_from_airsim(camera_info, image_width, image_height):
        """Build camera params dict from AirSim camera info for estimator compatibility."""
        hfov = math.radians(camera_info.fov)
        aspect = image_width / image_height
        vfov = 2 * math.atan(math.tan(hfov / 2) / aspect)

        fx = image_width / (2 * math.tan(hfov / 2))
        fy = image_height / (2 * math.tan(vfov / 2))
        cx = image_width * 0.5
        cy = image_height * 0.5

        camera_matrix = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=float)

        # AirSim pose provides camera pose in world frame.
        # R_cw maps camera -> world. Convert to world->camera for projection matrix.
        R_cw = airsim_utils.rotation_matrix_from_quat(camera_info.pose.orientation)
        R_wc = R_cw.T

        camera_center = np.array([
            camera_info.pose.position.x_val,
            camera_info.pose.position.y_val,
            camera_info.pose.position.z_val
        ], dtype=float).reshape(3, 1)

        t = -R_wc @ camera_center
        projection_matrix = camera_matrix @ np.hstack((R_wc, t))

        return {
            'camera_matrix': camera_matrix,
            'projection_matrix': projection_matrix,
            'R': R_wc,
            't': t,
            'camera_center': camera_center,
            'R_cam_to_world': R_cw,
            'convention': 'camera_to_world'
        }
    
    # Initialize models
    print("Initializing models...")
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
    
    try:
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device
        )
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        print("Falling back to CPU for depth estimation")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device='cpu'
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
        first_frame = get_airsim_scene_frame(airsim_client, airsim_camera_name, airsim_vehicle_name)
        if first_frame is None:
            print("Error: Could not retrieve initial scene frame from AirSim")
            return

        height, width = first_frame.shape[:2]
        fps = 30

        if use_airsim_camera_info:
            try:
                camera_info = airsim_client.simGetCameraInfo(airsim_camera_name, airsim_vehicle_name)
                params = build_camera_params_from_airsim(camera_info, width, height)
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
    
    print("Starting processing...")
    
    # Main loop
    while True:
        # Check for key press at the beginning of each loop
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
            print("Exiting program...")
            break
            
        try:
            # Read frame from selected source
            if first_frame is not None:
                frame = first_frame
                first_frame = None
            elif use_airsim_source:
                frame = get_airsim_scene_frame(airsim_client, airsim_camera_name, airsim_vehicle_name)
                if frame is None:
                    print("Warning: Empty frame from AirSim, skipping")
                    continue

                if use_airsim_camera_info and airsim_refresh_camera_params_every_frame:
                    try:
                        h_frame, w_frame = frame.shape[:2]
                        camera_info = airsim_client.simGetCameraInfo(airsim_camera_name, airsim_vehicle_name)
                        params = build_camera_params_from_airsim(camera_info, w_frame, h_frame)
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
            
            # Make copies for different visualizations
            original_frame = frame.copy()
            detection_frame = frame.copy()
            depth_frame = frame.copy()
            result_frame = frame.copy()
            
            # Step 1: Object Detection
            try:
                detection_frame, detections = detector.detect(
                    detection_frame,
                    track=enable_tracking,
                    stream=enable_stream
                )
            except Exception as e:
                print(f"Error during object detection: {e}")
                detections = []
                cv2.putText(detection_frame, "Detection Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 2: Depth Estimation
            try:
                depth_map = depth_estimator.estimate_depth(original_frame)
                depth_colored = depth_estimator.colorize_depth(depth_map)
            except Exception as e:
                print(f"Error during depth estimation: {e}")
                # Create a dummy depth map
                depth_map = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 3: 3D Bounding Box Estimation
            boxes_3d = []
            active_ids = []
            
            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    
                    # Get class name
                    class_name = detector.get_class_names()[class_id]
                    
                    # Get depth in the region of the bounding box
                    # Try different methods for depth estimation
                    if class_name.lower() in ['person', 'cat', 'dog']:
                        # For people and animals, use the center point depth
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                        depth_method = 'center'
                    else:
                        # For other objects, use the median depth in the region
                        depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                        depth_method = 'median'
                    
                    # Calculate camera-coordinate location of object centre
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    distance = 1.0 + depth_value * 9.0  # replicate estimator mapping
                    pt2 = np.array([cx, cy, 1.0])
                    location_cam = np.linalg.inv(bbox3d_estimator.K) @ pt2 * distance
                    
                    # Optionally convert to world frame if a camera center transform is available
                    location_world = None
                    if world_transform is not None:
                        R = world_transform['R']
                        cam_center = world_transform['camera_center']
                        convention = world_transform.get('convention', 'world_to_camera')

                        if convention == 'camera_to_world':
                            location_world = R @ location_cam + cam_center.squeeze()
                        else:
                            # For world->camera extrinsics, X_w = R^T * X_c + C
                            location_world = R.T @ location_cam + cam_center.squeeze()
                    
                    # Create a simplified 3D box representation
                    box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': depth_value,
                        'depth_method': depth_method,
                        'class_name': class_name,
                        'object_id': obj_id,
                        'score': score,
                        'location_cam': location_cam,
                        'location_world': location_world
                    }
                    
                    boxes_3d.append(box_3d)
                    
                    # log world coordinates if computed
                    if box_3d.get('location_world') is not None:
                        print(f"Object {class_name} id={obj_id} world coord: {box_3d['location_world']}")
                    
                    # Keep track of active IDs for tracker cleanup
                    if obj_id is not None:
                        active_ids.append(obj_id)
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            # Clean up trackers for objects that are no longer detected
            bbox3d_estimator.cleanup_trackers(active_ids)
            
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
            
            
            # Add depth map to the corner of the result frame
            try:
                depth_height = height // 4
                depth_width = depth_height * width // height
                depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
                result_frame[0:depth_height, 0:depth_width] = depth_resized
            except Exception as e:
                print(f"Error adding depth map to result: {e}")
            
            # Write frame to output video
            out.write(result_frame)
            
            # Display frames
            cv2.imshow("3D Object Detection", result_frame)
            cv2.imshow("Depth Map", depth_colored)
            cv2.imshow("Object Detection", detection_frame)
            
            # Check for key press again at the end of the loop
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                print("Exiting program...")
                break
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Also check for key press during exception handling
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                print("Exiting program...")
                break
            continue
    
    # Clean up
    print("Cleaning up resources...")
    if cap is not None:
        cap.release()
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