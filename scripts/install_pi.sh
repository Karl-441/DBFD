#!/bin/bash
set -e

# Ensure we are in the scripts directory to make relative paths work
cd "$(dirname "$0")"

echo "Starting DBFD Installation for Raspberry Pi..."

# 1. Update System
echo "Updating apt repositories..."
sudo apt-get update

# 2. Install System Dependencies (Preferred over pip for Pi to avoid compilation)
echo "Installing system libraries..."
# Added python3-skimage (scikit-image) which is heavy to compile
# Added python3-mss if available, otherwise pip will handle it
# Note: Replaced libatlas-base-dev with libopenblas-dev for newer Debian versions (Trixie+)
# Added python3-libcamera (Picamera2) for official RPi cameras on Bullseye/Bookworm
# Added python3-rpi.gpio for alarm module
# Added python3-gpiozero for Raspberry Pi 5 compatibility
sudo apt-get install -y python3-opencv python3-numpy python3-scipy python3-pyqt6 python3-skimage python3-pip python3-venv libopenblas-dev python3-libcamera python3-rpi.gpio python3-gpiozero

# 3. Create Virtual Environment (Portable-ish for same architecture)
echo "Creating Virtual Environment in 'venv'..."
python3 -m venv ../venv --system-site-packages

# Activate venv
source ../venv/bin/activate

# 4. Install Python Dependencies
echo "Installing Python packages into venv..."
# Use pip to install remaining packages (mss, ultralytics)
# --system-site-packages allows us to use the apt-installed cv2, pyqt6, skimage
pip install mss psutil ultralytics --break-system-packages

echo "Permissions..."
chmod +x start_service.sh
chmod +x fix_alarm_boot.sh

echo "Applying GPIO Boot Fix (Prevent Alarm Trigger)..."
./fix_alarm_boot.sh || echo "Warning: Failed to run boot fix script. Please run manually."

echo "Configuring Kernel UDP Buffers (sysctl) for smooth streaming..."
# Increase UDP buffers for reliable streaming (as per rpicam/mediamtx recommendation)
SYSCTL_CONF="/etc/sysctl.conf"
if ! grep -q "net.core.rmem_default=1000000" "$SYSCTL_CONF"; then
    echo "net.core.rmem_default=1000000" | sudo tee -a "$SYSCTL_CONF"
    echo "net.core.rmem_max=1000000" | sudo tee -a "$SYSCTL_CONF"
    sudo sysctl -p
    echo "Sysctl updated."
else
    echo "Sysctl already configured."
fi

echo "Installation Complete!"
echo "--------------------------------------------------------"
echo "A virtual environment has been created in '../venv'."
echo "You can copy the entire 'DBFD-Raspberry' folder to another Pi 4B"
echo "and it should run without reinstalling (provided apt packages match)."
echo "--------------------------------------------------------"
echo "Run './start_service.sh' to test."
