#!/usr/bin/env python3
import os
import json
import numpy as np
import math
from pathlib import Path

def load_camera_params(params_file):
    """
    Load camera parameters from a JSON file.
    
    Args:
        params_file (str): Path to the JSON file containing camera parameters
        
    Returns:
        dict: Dictionary containing camera parameters
    """
    if not os.path.exists(params_file):
        print(f"Warning: Camera parameters file {params_file} not found. Using default parameters.")
        return None
    
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        # Convert lists to numpy arrays
        params['camera_matrix'] = np.array(params['camera_matrix'])
        params['dist_coeffs'] = np.array(params.get('dist_coeffs', []))
        
        # if rotation/translation given, build projection_matrix from them and store R,t
        if 'rotation_matrix' in params:
            # R is rotation matrix that transform world coordinates to camera coordinates, note as R_{wc}
            R = np.array(params['rotation_matrix'])
            params['R'] = R
            # handle two possible translation conventions
            if 'translation_vector' in params:
                # translation expressed in camera coordinates (world origin in camera frame)
                t_cam = np.array(params['translation_vector']).reshape(3, 1)
                params['t'] = t_cam
                # P_c = R_{wc}P_w + T = K@np.vstack((P_w,1)), K is the projection matrix
                params['projection_matrix'] = create_projection_matrix(params['camera_matrix'], R, t_cam)
            elif 'camera_center' in params:
                # camera center expressed in world coordinates
                center = np.array(params['camera_center']).reshape(3, 1)
                params['camera_center'] = center
                # convert to camera translation: t = -R * C
                t_cam = -R @ center
                params['t'] = t_cam
                # P_c = R_{wc}(P_w-T)=R_{wc}P_w - R_{wc}T=K@np.vstack((P_w,1))
                params['projection_matrix'] = create_projection_matrix(params['camera_matrix'], R, t_cam)
            else:
                # no translation provided, assume zero
                t_cam = np.zeros((3, 1))
                params['t'] = t_cam
                params['projection_matrix'] = create_projection_matrix(params['camera_matrix'], R, t_cam)
        elif 'projection_matrix' in params:
            params['projection_matrix'] = np.array(params['projection_matrix'])
            # extract R,t from P if possible
            try:
                # since P = K [R|t], recover [R|t] = K^{-1} P
                K_inv = np.linalg.inv(params['camera_matrix'])
                RT = K_inv @ params['projection_matrix']
                params['R'] = RT[:, :3]
                params['t'] = RT[:, 3:].reshape(3, 1)
            except Exception:
                pass
        else:
            # fall back to intrinsics only
            params['projection_matrix'] = create_projection_matrix(params['camera_matrix'])
        
        print(f"Loaded camera parameters from {params_file}")
        print(f"Camera matrix:\n{params['camera_matrix']}")
        print(f"Projection matrix:\n{params['projection_matrix']}")
        
        return params
    
    except Exception as e:
        print(f"Error loading camera parameters: {e}")
        return None

def build_camera_params_from_airsim(camera_info, image_width, image_height, airsim_utils_module):
    '''
    In a simulation like AirSim, if the sensor's physical aspect ratio (calculated from Filmback. SensorWidth / Filmback.SensorHeight) does not match
    the rendering resolution's aspect ratio (image_width / image_height), and the engine is forced to stretch the image to fit instead of cropping/letterboxing, 
    it results in unequal pixel scaling on the axes.
    '''
    hfov = math.radians(camera_info.fov)
    fx = image_width / (2 * math.tan(hfov / 2))
    fy = fx
    cx = image_width * 0.5
    cy = image_height * 0.5
    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    r_cam_to_world = airsim_utils_module.rotation_matrix_from_quat(camera_info.pose.orientation)
    r_world_to_cam = r_cam_to_world.T
    camera_center = np.array(
        [
            camera_info.pose.position.x_val,
            camera_info.pose.position.y_val,
            camera_info.pose.position.z_val,
        ],
        dtype=float,
    ).reshape(3, 1)
    t = -r_world_to_cam @ camera_center
    projection_matrix = camera_matrix @ np.hstack((r_world_to_cam, t))

    return {
        "camera_matrix": camera_matrix,
        "projection_matrix": projection_matrix,
        "R": r_world_to_cam,
        "t": t,
        "camera_center": camera_center,
        "R_cam_to_world": r_cam_to_world,
        "convention": "camera_to_world",
    }

def create_projection_matrix(camera_matrix, R=None, t=None):
    """
    Create a projection matrix from camera intrinsic and extrinsic parameters.
    
    Args:
        camera_matrix (numpy.ndarray): Camera intrinsic matrix (3x3)
        R (numpy.ndarray): Rotation matrix (3x3)
        t (numpy.ndarray): Translation vector (3x1)
        
    Returns:
        numpy.ndarray: Projection matrix (3x4)
    """
    if R is None:
        R = np.eye(3)
    
    if t is None:
        t = np.zeros((3, 1))
    
    # Combine rotation and translation, hstack receives sequence(a tuple)
    RT = np.hstack((R, t))
    
    # Create projection matrix
    projection_matrix = camera_matrix @ RT
    
    return projection_matrix

def apply_camera_params_to_estimator(bbox3d_estimator, params):
    """
    Apply camera parameters to a 3D bounding box estimator.
    
    Args:
        bbox3d_estimator: BBox3DEstimator instance
        params (dict): Dictionary containing camera parameters
        
    Returns:
        bbox3d_estimator: Updated BBox3DEstimator instance
    """
    if params is None:
        print("Warning: No camera parameters provided. Using default parameters.")
        return bbox3d_estimator
    
    # Update camera matrix
    if 'camera_matrix' in params:
        bbox3d_estimator.K = params['camera_matrix']
    
    # Update projection matrix
    if 'projection_matrix' in params:
        bbox3d_estimator.P = params['projection_matrix']
    
    print("Applied camera parameters to 3D bounding box estimator")
    
    return bbox3d_estimator

def main():
    """Example usage of the camera parameter functions."""
    # Configuration variables (modify these as needed)
    # ===============================================
    
    # Input file
    params_file = "camera_params.json"  # Path to camera parameters JSON file
    
    # Camera position (for example purposes)
    camera_height = 1.65  # Camera height above ground in meters
    # ===============================================
    
    # Load camera parameters
    params = load_camera_params(params_file)
    
    if params:
        print("\nCamera Parameters:")
        print(f"Image dimensions: {params['image_width']}x{params['image_height']}")
        print(f"Reprojection error: {params['reprojection_error']}")
        
        # Example of creating a projection matrix with different extrinsic parameters
        print(f"\nExample: Creating a projection matrix with camera raised {camera_height}m above ground")
        R = np.eye(3)
        t = np.array([[0], [camera_height], [0]])  # Camera above ground
        
        projection_matrix = create_projection_matrix(params['camera_matrix'], R, t)
        print(f"New projection matrix:\n{projection_matrix}")

if __name__ == "__main__":
    main() 