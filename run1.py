import os
import setup_path 
import sys
import time
import cv2
import numpy as np
import torch
import cosysairsim as airsim  # type: ignore
from cosysairsim import utils as airsim_utils
from detection_model import ObjectDetector #type: ignore
from depth_model import DepthEstimator #type: ignore

def main():
    use_airsim = True;use_model = True; 
    enable_tracking = True; enable_bev = True; enable_stream = True; 
    device = 0; camera = 0; radius = 200; pattern = "drone*"; 
    # yolo settings 
    yolo_weights = r"E:\Programs\AirSim\Cosys-AirSim\runs\detect\train9\weights\best.pt"; 
    yolo_model_size="nano"; yolo_custom_weights = True; 
    # depth model settings
    depth_model_size="outdooor"; depth_custom_weights = True; 
    if use_model and yolo_custom_weights and depth_custom_weights : 
        try:
            detector = ObjectDetector(
                device=device,
                weights_path=yolo_weights
            )
        except Exception as e:
            print(f"Error initializing object detector: {e}")
            print("Falling back to CPU for object detection")
            detector = ObjectDetector(
                weights_path=yolo_weights,
                device='cpu'
            )
    elif use_model:
        try:
            detector = ObjectDetector(
                model_size=yolo_model_size,
                device=device
            )
        except Exception as e:
            print(f"Error initializing object detector: {e}")
            print("Falling back to CPU for object detection")
            detector = ObjectDetector(
                model_size=yolo_model_size,
                device='cpu'
            )
    # airsim detection 
    client = airsim.VehicleClient()
    client.confirmConnection()
    if use_airsim:
        client.simClearDetectionMeshNames(camera, airsim.ImageType.Scene)
        client.simSetDetectionFilterRadius(camera, airsim.ImageType.Scene, radius)
        client.simAddDetectionFilterMeshName(camera, airsim.ImageType.Scene, pattern)
    










