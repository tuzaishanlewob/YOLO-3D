import os
import time
import cv2
import numpy as np
import torch

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import our modules
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from camera_module import apply_camera_params_to_estimator
from camera_module import estimate_from_depth_map
from camera_module import load_camera_params
from runtime_common import init_depth_estimator, init_detector

def main():
    """Main function."""
    # Configuration variables (modify these as needed)
    # ===============================================
    
    # Input/Output
    source = r"E:\Videos\80.mp4"  # Path to input video file or webcam index (0 for default camera)
    output_path = "output.mp4"  # Path to output video file
    
    # Model settings
    yolo_model_size = "nano"  # YOLOv11 model size: "nano", "small", "medium", "large", "extra"
    yolo_weights = r"E:\Programs\AirSim\Cosys-AirSim\runs\detect\train9\weights\best.pt"       # Path to your custom .pt file or model ID (None to use pretrained size above)
    depth_model_size = "small"  # Depth Anything v2 model size: "small", "base", "large"
    depth_weights = r"E:\Programs\AirSim\Cosys-AirSim\PythonClient\YOLO-3D\checkpoints\depth_anything_v2_metric_hypersim_vits.pth" # Optional path/model id for depth model (e.g. .pth/.pt checkpoint)
    
    # Device settings
    device = 0  # Force CPU for stability
    
    # Detection settings
    conf_threshold = 0.25  # Confidence threshold for object detection
    iou_threshold = 0.45  # IoU threshold for NMS
    classes = None  # Filter by class, e.g., [0, 1, 2] for specific classes, None for all classes
    
    # Feature toggles
    enable_tracking = True  # Enable object tracking
    enable_bev = True  # Enable Bird's Eye View visualization
    enable_pseudo_3d = True  # Enable pseudo-3D visualization
    enable_stream = True  # Use streaming mode for detector to lower memory usage
    
    # Camera parameters - simplified approach
    camera_params_file = "cam.json"  # Path to camera parameters file (None to use default parameters)
    # ===============================================
    
    print(f"Using device: {device}")
    
    # Initialize models
    print("Initializing models...")
    detector = init_detector(
        model_size=yolo_model_size,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        classes=classes,
        device=device,
        weights_path=yolo_weights,
    )
    depth_estimator = init_depth_estimator(
        model_size=depth_model_size,
        device=device,
        weights_path=depth_weights,
    )
    
    # Initialize 3D bounding box estimator with default parameters
    # Simplified approach - focus on 2D detection with depth information
    bbox3d_estimator = BBox3DEstimator()

    # Load and apply camera extrinsics/intrinsics if a file was provided
    params = None
    world_transform = None
    if camera_params_file is not None:
        params = load_camera_params(camera_params_file)
        bbox3d_estimator = apply_camera_params_to_estimator(bbox3d_estimator, params)
        if params is not None and 'R' in params:
            world_transform = {
                'R': params['R'],
                'camera_center': params.get('camera_center'),
                't': params.get('t'),
                'convention': params.get('convention', 'world_to_camera'),
            }
    
    # Initialize Bird's Eye View if enabled
    if enable_bev:
        # Use a scale that works well for the 1-5 meter range
        bev = BirdEyeView(scale=60, size=(300, 300))  # Increased scale to spread objects out
    
    # Open video source
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
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make copies for different visualizations
            original_frame = frame.copy()
            detection_frame = frame.copy()
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
                    
                    estimate = estimate_from_depth_map(
                        depth_estimator=depth_estimator,
                        depth_map=depth_map,
                        bbox=bbox,
                        class_name=class_name,
                        camera_matrix=bbox3d_estimator.K,
                        world_transform=world_transform,
                        is_metric=bool(getattr(depth_estimator, "is_metric_depth", False) or getattr(depth_estimator, "use_raw_model", False)),
                        method_suffix="video-linalg",
                    )
                    if estimate is None:
                        continue
                    
                    # Create a simplified 3D box representation
                    box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': estimate['depth_value'],
                        'depth_method': estimate['depth_method'],
                        'class_name': class_name,
                        'object_id': obj_id,
                        'score': score,
                        'location_cam': estimate['location_cam'],
                        'location_world': estimate['location_world']
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