# User Manual for DBFD System

## 1. Introduction
The **Drone Based Fire Detector (DBFD)** system provides a comprehensive visual interface for detecting fire in images and video streams using advanced algorithms (PNN and YOLOv8).

## 2. Installation
1. Ensure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install PyQt6 opencv-python numpy mss ultralytics scikit-image
   ```

## 3. Launching the Application
Run the GUI application:
```bash
python d:\Github\DBFD\ui\gui.py
```

## 4. Features & Usage

### 4.1 Media Input
The left panel allows you to select the input source:
- **Upload Image**: Process a single static image (JPG, PNG, BMP).
- **Open Video File**: Process a pre-recorded video file (MP4, AVI).
- **Open Camera**: Connect to the default system camera (Webcam) for real-time detection.
- **Screen Capture**: Capture the primary monitor screen in real-time.

### 4.2 Algorithm Control
- **Select Model**:
  - **PNN (Color+Texture)**: Uses the custom probabilistic neural network trained on color (YCbCr) and texture (GLCM) features. Best for distinct fire colors and textures.
  - **YOLO (Deep Learning)**: Uses the YOLOv8n object detection model. Best for generalization.
- **Start/Stop**: Controls the processing pipeline.
- **Status**: Shows current system state.

### 4.3 Visualization & Export
- **Main Display**: Shows the processed feed with bounding boxes around detected fire regions.
- **FPS**: Shows the processing speed in Frames Per Second.
- **Save Current Frame**: Saves the currently displayed frame as an image.
- **Start/Stop Recording**: Records the processed video stream to an AVI file.

## 5. Troubleshooting
- **PNN Model Error**: Ensure `train_pnn.py` has been run to generate `model_pnn.pkl`.
- **Camera Error**: Ensure no other application is using the camera.
- **Performance**: High-resolution screen capture or video might reduce FPS. Resize the input or use a more powerful GPU for YOLO.

## 6. Security & Limitations
- **File Types**: Only standard image and video formats are allowed via the file picker.
- **Performance**: Large video files are processed frame-by-frame. Real-time performance depends on hardware.
