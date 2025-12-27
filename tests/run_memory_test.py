
import sys
import os
import time
import psutil
import threading
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Mock modules BEFORE importing core.headless_runner
# We need to mock cv2.VideoCapture and picamera2

class MockVideoCapture:
    def __init__(self, *args, **kwargs):
        print("MockVideoCapture initialized")
        self.width = 640
        self.height = 480
    
    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self.width = int(value)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self.height = int(value)
        return True
        
    def isOpened(self):
        return True
        
    def read(self):
        # Create a dummy frame (random noise or black)
        # Random noise to trigger some processing
        frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        # Make a bright spot to potentially trigger "fire" detection candidates
        cv2.circle(frame, (100, 100), 20, (0, 0, 255), -1) 
        return True, frame
        
    def release(self):
        pass

# Patch cv2.VideoCapture
cv2.VideoCapture = MockVideoCapture

# Patch LibCameraWrapper if needed (we'll force config to use OpenCV for simplicity or patch both)
# Let's force config to NOT use libcamera for this test to rely on our cv2 mock
config.USE_LIBCAMERA = False
config.MAX_MEMORY_MB = 500 # Set user requirement

from core.headless_runner import run_headless

def monitor_memory(pid, duration, interval=0.5):
    process = psutil.Process(pid)
    max_mem = 0
    start_time = time.time()
    
    print(f"Starting memory monitor for PID {pid}...")
    
    measurements = []
    
    while time.time() - start_time < duration:
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / 1024 / 1024
        max_mem = max(max_mem, rss_mb)
        measurements.append(rss_mb)
        # print(f"Current Memory: {rss_mb:.2f} MB")
        
        if rss_mb > config.MAX_MEMORY_MB:
             print(f"WARNING: Memory exceeded limit! {rss_mb:.2f} MB > {config.MAX_MEMORY_MB} MB")
             
        time.sleep(interval)
        
    avg_mem = sum(measurements) / len(measurements)
    print("\n" + "="*40)
    print(f"Memory Test Results (Duration: {duration}s)")
    print(f"Max Memory: {max_mem:.2f} MB")
    print(f"Avg Memory: {avg_mem:.2f} MB")
    print(f"Limit: {config.MAX_MEMORY_MB} MB")
    print("="*40 + "\n")

def main():
    print("Preparing to run headless memory test...")
    
    pid = os.getpid()
    duration = 15 # Run for 15 seconds

    # Run monitor in main thread or join it
    # Run run_headless in a daemon thread so it dies when we exit
    
    runner_thread = threading.Thread(target=run_headless, args=(0,))
    runner_thread.daemon = True
    runner_thread.start()
    
    # Run monitor in this thread (blocking)
    monitor_memory(pid, duration)
    
    print("Test duration reached. Exiting.")

if __name__ == "__main__":
    main()
