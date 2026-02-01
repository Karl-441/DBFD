# Alarm Module Setup Guide

## Overview
The DBFD project supports the ST011 (or compatible) Sound and Light Alarm Module. When fire is detected, the system triggers the alarm for a configured duration.

## Hardware Connection

### Components
1. **Raspberry Pi 4B**
2. **ST011 Alarm Module** (Has VCC, GND, and Signal pins)
3. **Jumper Wires** (Female-to-Female)

### Wiring Diagram

| ST011 Pin | Raspberry Pi Pin | Description |
|-----------|------------------|-------------|
| **VCC** | **Pin 2 (5V)** | Power Supply (Red) |
| **GND** | **Pin 6 (GND)** | Ground (Black) |
| **IN / SIG** | **Pin 11 (GPIO 17)** | Signal Control (Yellow/White) |

**Note**:
- The ST011 module typically operates on 5V but accepts 3.3V logic signals from the Pi.
- Ensure connections are secure before powering on the Pi.

## Software Configuration

### 1. Install Dependencies
The alarm module requires `RPi.GPIO`. If you haven't run the installer recently, update your dependencies:
```bash
./scripts/install_pi.sh
```

### 2. Configuration (`config.py`)
You can adjust the alarm behavior in `config.py`:

```python
# Alarm Settings (ST011 Module)
ALARM_GPIO_PIN = 17  # BCM numbering (GPIO 17 is Physical Pin 11)
ALARM_ACTIVE_HIGH = True # Set to False if your module triggers on Low signal
ALARM_COOLDOWN = 5.0 # How long the alarm stays on after detection (seconds)
```

## Testing

1. **Connect the hardware** as described above.
2. **Run the system**:
   ```bash
   ./scripts/start_service.sh
   ```
3. **Trigger detection**: Show a fire-like object (red/orange flame) to the camera.
4. **Verify**:
   - The ST011 module should light up and buzz.
   - The terminal log should show `ALARM ON!`.
   - The alarm should turn off automatically after 5 seconds of no fire detection.

## Troubleshooting
- **Alarm always on?** Try changing `ALARM_ACTIVE_HIGH` to `False` in `config.py`.
- **No sound?** Check wiring connections, specifically the GPIO pin number (BCM vs Physical). We use BCM 17.
