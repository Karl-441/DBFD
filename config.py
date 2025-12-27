import os

# Hardware Constraints
# Raspberry Pi 4B (Optimized for low memory usage)
MAX_MEMORY_MB = 350  # Lowered threshold for GUI mode safety
GC_INTERVAL = 100 # Frames between forced garbage collection

# Camera Settings
USE_LIBCAMERA = False # Set to True to use Picamera2 (rpicam) instead of OpenCV VideoCapture
CAMERA_INDEX = 0  # Default camera
FRAME_WIDTH = 320  # Reduced resolution for performance and memory (320x240 is enough for fire detection)
FRAME_HEIGHT = 240
FPS = 10  # Lower FPS to save CPU and reduce frame buffering pressure

# Algorithm Settings
USE_PNN = True  # Default to PNN (lighter)
USE_YOLO = False  # Disable YOLO by default on 1GB Pi

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_pnn.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
