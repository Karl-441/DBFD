import numpy as np
import time

class LibCameraWrapper:
    def __init__(self, width, height, fps):
        try:
            from picamzero import Camera
        except ImportError:
            raise ImportError("picamzero library not found. Please install it.")

        self.cam = Camera()
        # Configure camera
        # Note: picamzero API might differ slightly depending on version, 
        # but generally supports basic capture.
        # For high performance stream, we might need lower level access or just rapid capture.
        
        # Picamzero is high level. For raw performance, usually picamera2 is better,
        # but user asked for "rpicam" which often refers to the modern libcamera stack.
        # Picamera2 is the official python lib for libcamera.
        pass

# Actually, the standard library for libcamera in Python is 'Picamera2'.
# Let's implement using Picamera2 as it's the standard for RPi 4/5 (Bullseye/Bookworm).


class LibCameraWrapper:
    def __init__(self, width, height, fps):
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise ImportError("Picamera2 not installed. Run 'sudo apt install python3-libcamera'")

        self.picam2 = Picamera2()
        
        # Configure configuration
        # Use create_video_configuration for video streaming
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"},
            controls={"FrameDurationLimits": (int(1000000 / fps), int(1000000 / fps))} if fps > 0 else None
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        print(f"Picamera2 started at {width}x{height}")

    def read(self):
        # capture_array returns a numpy array (image)
        # This is a blocking call that waits for the next frame
        try:
            return self.picam2.capture_array()
        except Exception as e:
            print(f"Picamera2 capture error: {e}")
            return None

    def release(self):
        self.picam2.stop()
        self.picam2.close()
