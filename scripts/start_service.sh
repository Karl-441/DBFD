#!/bin/bash

# 获取脚本所在目录的父目录作为项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "------------------------------------------------"
echo "  DBFD-Raspberry Service Starter"
echo "  Project Root: $PROJECT_ROOT"
echo "------------------------------------------------"

# 1. 环境检查与解释器选择 (Interpreter Selection)
PYTHON_EXEC="python3"

# 检测当前系统类型
IS_WINDOWS=false
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    IS_WINDOWS=true
fi

if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "[INFO] Found 'venv' folder."
    
    if [ "$IS_WINDOWS" = true ]; then
        PYTHON_EXEC="$PROJECT_ROOT/venv/Scripts/python.exe"
    else
        # Linux/树莓派环境
        if [ -f "$PROJECT_ROOT/venv/bin/python3" ]; then
            PYTHON_EXEC="$PROJECT_ROOT/venv/bin/python3"
        elif [ -f "$PROJECT_ROOT/venv/bin/python" ]; then
            PYTHON_EXEC="$PROJECT_ROOT/venv/bin/python"
        else
            echo "[WARN] 'venv' exists but is NOT a Linux virtual environment (possibly copied from Windows)."
            echo "[WARN] Falling back to system python3..."
            PYTHON_EXEC="python3"
        fi
    fi
    echo "[INFO] Using interpreter: $PYTHON_EXEC"
else
    echo "[WARN] 'venv' not found. Using system python3."
    PYTHON_EXEC="python3"
fi

# 2. 路径与环境变量配置 (Paths & Fixes)
# 确保项目根目录在 PYTHONPATH 中
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 修复 OpenCV "co-located POCs unavailable" 错误
# 强制 FFmpeg 禁用不稳定的硬件加速探测，并设置较低的日志级别
export OPENCV_FFMPEG_CAPTURE_OPTIONS="video_codec;h264|rtsp_transport;udp"
export OPENCV_LOG_LEVEL=ERROR
export OPENCV_VIDEOIO_PRIORITY_MSMF=0

# --- 硬件加速优化 (Vulkan & GPU Acceleration) ---
# 1. 强制启用 MESA Vulkan 驱动 (树莓派 4/5 必备)
export MESA_VULKAN_DEVICE_SELECT=1
export LD_LIBRARY_PATH="/usr/lib/arm-linux-gnueabihf/vulkan:$LD_LIBRARY_PATH"

# 2.# 内存优化：限制 glibc 内存池数量，减少树莓派上的内存碎片
export MALLOC_ARENA_MAX=2

# 4. 清理旧进程 (Force cleanup)
pkill -9 rpicam-vid > /dev/null 2>&1
pkill -9 libcamera-vid > /dev/null 2>&1

# 5. 运行程序
# 硬件性能预热 (Power Management)
# 如果是树莓派，尝试强制 GPU 保持高性能 (需 sudo 权限或 root 运行，这里仅建议配置)
# echo "performance" | sudo tee /sys/class/graphics/fb0/device/power_dpm_state > /dev/null 2>&1

if [ -z "$DISPLAY" ]; then
    echo "[INFO] No DISPLAY detected. Starting in Headless Mode..."
    $PYTHON_EXEC main.py --headless "$@"
else
    echo "[INFO] DISPLAY detected ($DISPLAY). Starting GUI Mode..."
    $PYTHON_EXEC main.py "$@"
fi

# 检查运行结果
if [ $? -ne 0 ]; then
    echo "[ERROR] DBFD exited with error code $?."
    exit $?
fi
