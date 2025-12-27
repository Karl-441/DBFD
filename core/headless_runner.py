try:
    import cv2
except ImportError:
    # This will be caught by main.py
    raise ImportError("opencv-python (cv2) is not installed. Please run 'sudo apt install python3-opencv' or 'pip install opencv-python-headless'.")

import time
import pickle
import sys
import os
import numpy as np
import config
from algorithm.preprocess import preprocess_image
from algorithm.features import extract_features
from core.output_manager import OutputManager

import gc

def _setup_opencv_camera(camera_index):
    # Setup Camera
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    return cap

def run_headless(camera_index=0):
    print(f"Starting Headless Mode on Camera {camera_index}...")
    
    # Load Model
    pnn_model = None
    if os.path.exists(config.MODEL_PATH):
        with open(config.MODEL_PATH, 'rb') as f:
            pnn_model = pickle.load(f)
        print("PNN Model loaded.")
    else:
        print(f"Error: Model not found at {config.MODEL_PATH}")
        return

    # Setup Camera
    if config.USE_LIBCAMERA:
        print("Using Libcamera (Picamera2)...")
        try:
            from core.camera_wrapper import LibCameraWrapper
            cap = LibCameraWrapper(config.FRAME_WIDTH, config.FRAME_HEIGHT, config.FPS)
            print("Libcamera initialized.")
        except ImportError as e:
            print(f"Failed to load Libcamera: {e}")
            print("Falling back to OpenCV VideoCapture...")
            cap = _setup_opencv_camera(camera_index)
            if not cap: return
    else:
        cap = _setup_opencv_camera(camera_index)
        if not cap: return

    output_manager = OutputManager()
    last_save_time = 0
    save_interval = 2.0 # Save at most one image every 2 seconds to save disk/IO
    
    # Optimization: Pre-allocate variables if possible (Python dynamic typing makes this harder, but good to keep in mind)
    frame_count = 0

    print("Monitoring started. Press Ctrl+C to stop.")
    
    try:
        while True:
            start_time = time.time()
            
            # Polymorphic read
            if config.USE_LIBCAMERA:
                frame = cap.read() # LibCameraWrapper returns frame directly
                ret = frame is not None
            else:
                ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame.")
                time.sleep(1)
                continue

            # Detect
            detections, _ = detect_fire(frame, pnn_model)
            
            if detections:
                print(f"FIRE DETECTED! {len(detections)} regions.")
                
                # Save evidence
                if time.time() - last_save_time > save_interval:
                    # Draw boxes
                    vis = frame.copy()
                    for (x, y, w, h) in detections:
                        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(vis, "FIRE", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    path = output_manager.save_prediction(vis, detections, filename=f"fire_alert_{int(time.time())}.jpg")
                    print(f"Alert saved: {path}")
                    last_save_time = time.time()
                    
                    # Explicit delete
                    del vis
            
            # Explicit delete of heavy objects
            del frame
            
            # Periodic GC (every GC_INTERVAL frames)
            frame_count += 1
            if frame_count % config.GC_INTERVAL == 0:
                gc.collect()

            # FPS Control
            process_time = time.time() - start_time
            sleep_time = max(0, (1.0/config.FPS) - process_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping headless runner...")
    finally:
        cap.release()

def detect_fire(img, pnn_model):
    # Reuse logic from original main.py or encapsulate in algorithm module
    # For now, duplicate logic to avoid circular deps or complex refactor
    # Ideally this should be in algorithm/detector.py
    
    try:
        mask = preprocess_image(img)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        detections = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 20: continue
            
            component_mask = np.zeros_like(mask)
            component_mask[labels == i] = 255
            
            roi = img[y:y+h, x:x+w]
            roi_mask = component_mask[y:y+h, x:x+w]
            
            try:
                feats = extract_features(roi, roi_mask)
                pred = pnn_model.predict(feats)[0]
                if pred == 1:
                    detections.append((x, y, w, h))
            except:
                continue
                
        return detections, mask
    except Exception as e:
        # print(f"Detection error: {e}")
        return [], None
