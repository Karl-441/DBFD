# DBFD - Raspberry Pi Edition

Optimized Fire Detection System for Raspberry Pi 4B.

## Features
- **Low Memory Footprint**: Uses PNN algorithm by default (<500MB RAM).
- **Headless Mode**: Runs without GUI for maximum stability.
- **Auto-Recovery**: Systemd service support.
- **Memory Monitoring**: Auto-cleans or warns on high memory usage.

## Hardware Requirements
- Raspberry Pi 4B (1GB RAM minimum)
- Raspberry Pi Camera or USB Webcam
- Raspberry Pi OS (64-bit recommended)

## Installation

1. Clone or copy this repository to `/home/pi/DBFD-Raspberry`.
2. Run the installer:
   ```bash
   cd scripts
   chmod +x install_pi.sh
   ./install_pi.sh
   ```

## Usage

### Manual Start (Headless)
```bash
python3 main.py --headless
```

### GUI Mode (Requires Desktop)
```bash
python3 main.py
```

### Auto-Start (Service)
To enable the systemd service:
```bash
sudo cp scripts/dbfd.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dbfd
sudo systemctl start dbfd
```

## Configuration
Edit `config.py` to adjust:
- `FRAME_WIDTH`, `FRAME_HEIGHT` (Resolution)
- `MAX_MEMORY_MB` (Memory limit)
- `CAMERA_INDEX`

## Logs & Output
- Logs are saved in `logs/` (if configured) or stdout.
- Detected fire images are saved in `output/`.
