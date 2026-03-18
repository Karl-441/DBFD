#!/bin/bash

# 获取脚本所在目录并切换到该目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "===================================================="
echo "  火灾检测模型人工修正与迭代组件 - 一键启动脚本"
echo "  支持系统: Raspberry Pi OS 5 (Bookworm) / Ubuntu / Debian"
echo "===================================================="

# 检查 Python3
if ! command -v python3 &> /dev/null
then
    echo "错误: 未找到 python3。请先运行: sudo apt update && sudo apt install python3 python3-venv python3-pyqt6 python3-opencv python3-pil"
    exit 1
fi

# 1. 检查或创建项目根目录的虚拟环境 (PEP 668 合规)
# 根目录相对于组件目录是 ..
ROOT_DIR="$(cd "$DIR/.." && pwd)"
VENV_DIR="$ROOT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] 正在项目根目录创建虚拟环境 (使用 --system-site-packages 以适配树莓派 PyQt6/OpenCV)..."
    # 使用 --system-site-packages 允许虚拟环境访问 apt 安装的 PyQt6 和 OpenCV
    python3 -m venv --system-site-packages "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "错误: 创建虚拟环境失败。可能需要安装 python3-venv: sudo apt install python3-venv"
        exit 1
    fi
fi

# 2. 激活根目录虚拟环境并安装依赖
echo "[2/3] 正在检查并安装项目依赖 (使用根目录 requirements.txt)..."
source "$VENV_DIR/bin/activate"

# 升级 pip
pip install --quiet --upgrade pip

ROOT_REQ="$ROOT_DIR/requirements.txt"
if [ -f "$ROOT_REQ" ]; then
    pip install -r "$ROOT_REQ"
else
    echo "警告: 未找到根目录 requirements.txt。"
fi

# 3. 设置环境变量以适配 Raspberry Pi OS 5 (Bookworm/Wayland)
echo "[3/3] 配置运行环境 (Wayland/X11 适配)..."

# 检查是否为 Wayland 会话
if [ "$XDG_SESSION_TYPE" == "wayland" ]; then
    # 树莓派 OS 5 默认使用 Wayland
    # 某些版本的 PyQt6 需要指定平台
    export QT_QPA_PLATFORM=wayland
    echo "检测到 Wayland 会话，已设置 QT_QPA_PLATFORM=wayland"
else
    export QT_QPA_PLATFORM=xcb
    echo "检测到 X11 或其他会话，已设置 QT_QPA_PLATFORM=xcb"
fi

# 解决某些 OpenCV 版本在树莓派上的 Qt 库冲突问题
export QT_LOGGING_RULES="*.debug=false;qt.qpa.*=false"

# 4. 启动主程序
echo "----------------------------------------------------"
echo "正在启动组件..."
python3 main.py

# 结束后自动退出虚拟环境
deactivate
echo "===================================================="
echo "组件已关闭。"
