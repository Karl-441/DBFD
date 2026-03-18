# DBFD-Raspberry (树莓派火灾检测系统)

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red.svg)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

DBFD-Raspberry (Drone Based Fire Detector) 是一款专为树莓派 5 优化的实时火灾检测系统。该项目结合了传统的图像处理算法 (PNN) 与先进的深度学习模型 (YOLOv8)，旨在提供低延迟、高可靠性的火灾预警方案。

## 核心特性

-   **双算法引擎**:
    -   **PNN (Pixel-based Neural Network)**: 极低计算开销，利用色彩空间和纹理特征进行快速初筛。
    -   **YOLOv8 (NCNN/Vulkan)**: 利用树莓派 5 的 GPU 加速，通过 Vulkan 接口运行 NCNN 优化模型，提供高精度的火灾识别。
-   **树莓派 5 深度优化**:
    -   **LibCamera 封装**: 通过 `rpicam-vid` 推流，绕过 OpenCV V4L2 的性能瓶颈。
    -   **H.264 Baseline**: 强制使用 Baseline Profile 消除 POC 同步错误，实现毫秒级流传输延迟。
    -   **缓冲区管理**: 智能丢帧逻辑，确保算法处理的始终是最新的传感器画面。
-   **工业级预警**:
    -   **GPIO 硬件控制**: 针对 Pi 5 的 RP1 芯片优化的 `lgpio` 控制，实现毫秒级蜂鸣器响应。
    -   **粘性报警逻辑**: 引入 3.0s 冷却机制，有效防止检测抖动导致的报警器频繁闪烁。
-   **数据闭环**:
    -   **自动截图**: 检测到火灾时，每处理 3 帧自动保存一张带检测框的截图及其 JSON 元数据。
    -   **人工修正兼容**: 输出格式完全兼容 `fire_correction_component`，支持后续的人工校验与模型迭代。

## 目录结构

```text
d:\Github\DBFD-Raspberry\
├── algorithm/            # PNN、YOLO (NCNN) 及融合算法实现
├── core/                 # 系统核心模块 (摄像头封装、报警管理、配置管理)
├── ui/                   # 基于 PyQt6 的图形界面 (主窗口、配置窗口、转换工具)
├── output/               # 数据输出目录 (截图、JSON、运行日志)
├── fire_correction_component/ # (子组件) 人工修正与模型再训练工具
├── models/               # 存放 .pt 和 .ncnn 模型权重
├── scripts/              # 安装与启动脚本
├── config.py             # 动态配置代理
└── main.py               # 程序总入口
```

## 快速开始

### 1. 环境准备
推荐使用树莓派官方 64 位系统 (Debian Trixie)。
```bash
sudo apt update
sudo apt install python3-pyqt6 python3-opencv python3-libcamera python3-venv libcamera-apps-lite
```

### 2. 安装与运行
```bash
# 克隆仓库
git clone https://github.com/user/DBFD-Raspberry.git
cd DBFD-Raspberry

# 运行启动脚本 (自动处理 venv)
chmod +x scripts/start_service.sh
./scripts/start_service.sh
```

## 识别输出说明

检测到火灾时，系统会在 `output/predictions/YYYYMMDD/` 目录下生成：
-   `pred_HHMMSS_ffffff.jpg`: 带有检测框和标签的画面截图。
-   `pred_HHMMSS_ffffff.json`: 包含版本信息、图像尺寸、检测框坐标（bbox: [x, y, w, h]）、置信度及算法来源的元数据。

这些数据可以直接被 `fire_correction_component` 扫描并用于后续的模型优化。

## 性能报告
本项目包含完整的性能测试套件。运行 `test_trixie.sh` 可生成详细的 CPU/GPU 频率、SoC 温度及 FPS 波动报告。

---
**注意**: 请确保在操作 GPIO 时具有相应的权限。建议将当前用户加入 `video` 和 `gpio` 组。
