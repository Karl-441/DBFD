#!/bin/bash
# Script to fix Alarm Module always-on issue at boot time
# ST011 is Active Low. We need GPIO 17 to be High (3.3V) at boot.

CONFIG_FILE="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="/boot/config.txt"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Could not find config.txt in /boot or /boot/firmware"
    exit 1
fi

echo "Configuring GPIO 17 to be Output High at boot in $CONFIG_FILE..."

# Check if already configured
if grep -q "gpio=17=op,dh" "$CONFIG_FILE"; then
    echo "Configuration already exists. No changes needed."
else
    # Append configuration
    echo "" | sudo tee -a "$CONFIG_FILE"
    echo "# DBFD Alarm Module Boot State (Prevent Trigger)" | sudo tee -a "$CONFIG_FILE"
    echo "gpio=17=op,dh" | sudo tee -a "$CONFIG_FILE"
    echo "Added 'gpio=17=op,dh' to config.txt."
    echo "Please REBOOT for this to take effect."
fi
