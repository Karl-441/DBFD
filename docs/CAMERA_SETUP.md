# 摄像头配置与故障排查指南 (Camera Setup Guide)

本项目采用 **UDP 管道流** 方案调用树莓派摄像头，以解决 Python 直接绑定 `libcamera` 库的不稳定问题。

## 1. 技术方案 (The Scheme)

*   **工具链**: `rpicam-vid` (CLI) -> MPEG-TS (LibAV) -> UDP (Localhost) -> OpenCV `VideoCapture`
*   **优势**: 
    *   **低延迟**: 在 Pi 5 上自动启用 `profile=baseline` 禁用 B 帧。
    *   **稳定性**: 进程隔离，摄像头崩溃不会导致主程序退出，且支持自动重启。
    *   **兼容性**: 自动适配 Bookworm (`rpicam-vid`) 和 Bullseye (`libcamera-vid`)。

## 2. 常见问题与修复

### 问题 A: `Failed to call rpicam` 或摄像头无法启动
**现象**: 程序启动后报错，或画面黑屏。
**排查步骤**:
1.  **检查硬件**: 确保排线连接正确（Pi 5 需要专用排线）。
2.  **检查依赖**: 运行 `rpicam-hello` 看看是否能出图。
3.  **查看日志**:
    *   新版程序会将 `rpicam-vid` 的错误输出到临时文件。
    *   在终端中运行程序，观察 "Error log" 输出。
    *   常见错误：
        *   `failed to open camera`: 硬件连接问题或被占用。
        *   `unknown option`: 参数不支持（如旧系统不支持 libav）。

### 问题 B: 画面卡顿或延迟高
**解决方案**:
*   系统已自动为 Pi 5 添加 `profile=baseline`。
*   如果仍有延迟，请检查 `/etc/sysctl.conf` 中的 UDP 缓冲区设置（运行 `install_pi.sh` 会自动配置）。

### 问题 C: 端口冲突
默认使用 UDP 端口 **1234**。如果该端口被占用，请修改 `core/camera_wrapper.py` 中的 `self.udp_port`。

## 3. 手动测试命令

您可以在终端手动运行以下命令来测试摄像头是否正常工作：

```bash
# Pi 5 / Bookworm
rpicam-vid -t 0 --inline --width 640 --height 480 --codec libav --libav-format mpegts -o udp://127.0.0.1:1234

# Pi 4 / Bullseye
libcamera-vid -t 0 --inline --width 640 --height 480 --codec libav --libav-format mpegts -o udp://127.0.0.1:1234
```

然后使用 VLC 播放器打开 `udp://@:1234` 查看画面。
