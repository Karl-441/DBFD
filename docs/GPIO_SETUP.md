# GPIO Setup Guide (Raspberry Pi 5 / 4)

This project uses the Raspberry Pi GPIO interface to control the **Sound and Light Alarm Module**.
With the update to Raspberry Pi 5, we now use `gpiozero` for better hardware compatibility, but the physical wiring remains standard.

## 1. Pinout Reference

Refer to the standard Raspberry Pi 40-Pin GPIO Header (BCM Mode).

| Physical Pin | Function | BCM GPIO | Note |
| :--- | :--- | :--- | :--- |
| **02** | 5V Power | - | Connect to Alarm VCC |
| **06** | Ground | - | Connect to Alarm GND |
| **11** | Signal | GPIO 17 | Connect to Alarm SIG/IN |

## 2. Wiring Instructions

### Sound & Light Alarm Module (e.g., ST011)

Most modules have 3 pins: **VCC**, **GND**, **I/O** (or SIG).

1.  **VCC (+)**: Connect to **Physical Pin 2** (5V).
    *   *Note: Some smaller buzzers work on 3.3V (Pin 1), but most loud alarms require 5V.*
2.  **GND (-)**: Connect to **Physical Pin 6** (GND).
    *   *Any GND pin works (e.g., Pin 9, 14, 20).*
3.  **SIG (Signal/IO)**: Connect to **Physical Pin 11** (GPIO 17).

### Visual Guide
```
      Raspberry Pi 40-Pin Header
      --------------------------
      3.3V  [1] [2]  5V  <-- Connect VCC here
     GPIO2  [3] [4]  5V
     GPIO3  [5] [6]  GND <-- Connect GND here
     GPIO4  [7] [8]  GPIO14
       GND  [9] [10] GPIO15
   GPIO17  [11] [12] GPIO18 <-- Connect SIG to Pin 11
   GPIO27  [13] [14] GND
   ...
```

## 3. Configuration

The default pin is **GPIO 17**. If you need to change this, edit `config.py`:

```python
# config.py
ALARM_GPIO_PIN = 17  # BCM Numbering
```

### Trigger Level (Active High vs Low)
*   **Active Low (Most Modules)**: The alarm sounds when the signal is LOW (0V). Set `ALARM_ACTIVE_HIGH = False`.
*   **Active High**: The alarm sounds when the signal is HIGH (3.3V). Set `ALARM_ACTIVE_HIGH = True`.

## 4. Independent Operation

The system is designed to be **fault-tolerant**. 
*   If the alarm module is **disconnected** or the GPIO initialization fails (e.g., on a PC or due to permission issues), the software will **log an error but continue running**.
*   The camera and detection algorithms will function normally without the alarm.
*   Check the logs (`logs/` directory) for "Alarm Manager initialized" or "Failed to setup Alarm" messages.

## 5. Troubleshooting (Pi 5 Specifics)

Raspberry Pi 5 uses a new I/O chip (RP1).
*   **Library**: Ensure `python3-gpiozero` is installed (handled by `install_pi.sh`).
*   **Permissions**: Ensure your user is in the `gpio` group (default on Pi OS).
*   **Conflict**: If you see "Pin is already in use", check if another process (like `libcamera` or a legacy script) is holding the pin.
