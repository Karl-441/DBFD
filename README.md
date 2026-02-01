# DBFD - Drone Based Fire Detector

DBFD is a comprehensive fire detection system designed for drone-based surveillance. It integrates traditional computer vision techniques with modern deep learning models to provide robust real-time fire detection capabilities.

## 🚀 Key Features

*   **Dual Algorithm Engine**:
    *   **PNN (Probabilistic Neural Network)**: Utilizes color (YCbCr) and texture (GLCM) features for efficient CPU-based detection.
    *   **YOLOv8 (You Only Look Once)**: State-of-the-art deep learning object detection for high accuracy.
    *   **Fusion Mode**: Combines both algorithms to minimize false positives.
*   **Real-time Visualization**:
    *   **GUI Interface**: Built with PyQt6, supporting drag-and-drop, real-time video feed, and result visualization.
    *   **Multi-Source Input**: Supports Images, Video Files, Webcams, and **Screen Capture** (for direct drone feed integration).
    *   **Instant Alert**: Visual "FIRE DETECTED" warning overlay on detection.
*   **Integrated Workflow**:
    *   **Auto-Training**: Built-in tool to train YOLO models on custom datasets with a single click.
    *   **Dataset Management**: Tools for labeling, merging, and managing fire datasets.
    *   **Auto-Push**: Automated Git synchronization for version control.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone git@github.com:Karl-441/DBFD.git
    cd DBFD
    ```

2.  **Install dependencies**:
    Double-click or run:
    ```bash
    install_dependencies.bat
    ```
    This will automatically create a virtual environment (`venv`) and install all required packages.

    *Note: For GPU acceleration, ensure you have the correct PyTorch version installed manually if needed, though default installation should work for most.*

## 🖥️ Usage

### 1. Start the Main Interface
Run the launcher script:
```bash
run_ui.bat
```
*   **Upload Image/Video**: Click buttons on the left panel.
*   **Select Algorithm**: Choose between PNN, YOLO, or Fusion.
*   **Real-time Detection**: The system will automatically process the input and display results.

### 2. Train YOLO Model
Use the automated training tool:
```bash
python tools/auto_train_yolo.py
```
*   Select your dataset folder (standard YOLO format).
*   The script will train the model and automatically deploy the best weights to the `models/` directory.

### 3. Data Labeling
Launch the standalone labeling tool:
```bash
run_labeling.bat
```

## 📂 Project Structure

*   `algorithm/`: Core detection logic (PNN, Feature Extraction, Fusion).
*   `core/`: System utilities (Dataset Manager, Output Manager).
*   `ui/`: PyQt6 graphical user interface.
*   `tools/`: Helper scripts for training and labeling.
*   `models/`: Stored model weights (.pkl, .pt).
*   `dataset/`: Training and validation data.
*   `output/`: Detection results and logs.

