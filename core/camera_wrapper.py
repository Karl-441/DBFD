import subprocess
import numpy as np
import cv2
import time
import signal
import shutil
import tempfile
from pathlib import Path

"""
    针对树莓派的新版摄像头栈 (libcamera / rpicam-apps) 进行封装。
    由于 OpenCV 的 V4L2 后端在某些 Pi 系统上兼容性不佳，本模块通过
    调用 `rpicam-vid` 或 `libcamera-vid` 命令行工具，将视频流推送到
    本地 UDP 端口，然后使用 OpenCV 读取该 UDP 流。
    这种方法能获得更好的性能和兼容性。
"""

class LibCameraWrapper:
    def __init__(self, width, height, fps):
        """
            width: 帧宽度
            height: 帧高度
            fps: 帧率
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.cap = None
        self.fail_count = 0
        self.max_fail = max(10, int(fps)) # 允许的最大连续读取失败次数
        self.log_file = Path(tempfile.gettempdir()) / "rpicam_stderr.log"
        
        # 确定使用的命令: 优先尝试 rpicam-vid (Bookworm), 然后是 libcamera-vid (Bullseye)
        self.cmd_base = None
        if shutil.which("rpicam-vid"):
            self.cmd_base = "rpicam-vid"
        elif shutil.which("libcamera-vid"):
            self.cmd_base = "libcamera-vid"
        
        if not self.cmd_base:
            # 开发环境 fallback
            print("Warning: rpicam-vid/libcamera-vid not found. Mocking camera or failing. (未找到 libcamera 工具)")
            raise ImportError("rpicam-vid or libcamera-vid not found. Please install rpicam-apps.")

        print(f"Using CLI Camera Tool: {self.cmd_base}")

        # 检测 Pi 5 以应用特定修复
        self.is_pi5 = self._check_pi5()
        if self.is_pi5:
            print("Detected Raspberry Pi 5. Applying 'profile=baseline' to disable B-frames. (检测到 Pi 5)")

        self.udp_port = 1234
        self._start_pipeline()

    def _check_pi5(self):
        """检查是否运行在 Raspberry Pi 5 上"""
        try:
            model_file = Path('/proc/device-tree/model')
            if model_file.exists():
                with model_file.open('r') as f:
                    model = f.read()
                    return "Raspberry Pi 5" in model
        except:
            pass
        return False

    def _build_command(self):
        """构建 libcamera 命令行参数"""
        cmd = [
            self.cmd_base,
            "-t", "0", # 无限时长
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.fps),
            "--nopreview", # 不显示预览窗口
            "--codec", "libav", # 使用 libav 编码
            "--libav-format", "mpegts", # MPEG-TS 容器格式，适合流传输
            "-o", f"udp://127.0.0.1:{self.udp_port}?pkt_size=1316" # 推流到本地 UDP
        ]
        
        # Pi 5 优化: 强制使用 baseline profile 以避免 B-frames (减少延迟)
        # 目前禁用，因可能与某些 Trixie libav 版本冲突
        # if self.is_pi5:
        #    cmd.extend(["--libav-video-codec-opts", "profile=baseline"])
            
        return cmd

    def _start_pipeline(self):
        """启动摄像推流进程和 OpenCV 读取连接"""
        # 1. 强制清理现有的摄像头进程，避免 "Device busy" 错误
        self._kill_existing_process()
        
        self.cmd = self._build_command()
        
        try:
            print(f"Starting Camera Process: {' '.join(self.cmd)}")
            
            # 使用日志文件记录 stderr，以便调试启动失败
            self.stderr_log = open(self.log_file, "w+")
            
            self.process = subprocess.Popen(
                self.cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=self.stderr_log
            )
            
            # 等待进程启动
            time.sleep(2.0)
            
            # 检查进程是否立即死亡
            if self.process.poll() is not None:
                self.stderr_log.seek(0)
                error_msg = self.stderr_log.read()
                if not error_msg:
                    error_msg = "[Empty Log] - Process exited without stderr output."
                print(f"!!! CAMERA STARTUP FAILED !!!\nCMD: {' '.join(self.cmd)}\nERROR LOG:\n{error_msg}\n-----------------------------")
                raise RuntimeError(f"Camera process failed to start. See log above.")
            
            print(f"Camera streaming to UDP://127.0.0.1:{self.udp_port}")
            
            # 配置 OpenCV 读取 UDP 流
            # 使用 udp://@:1234 绑定所有接口
            # 设置 overrun_nonfatal=1 防止缓冲区溢出导致崩溃
            # 增加 fifo_size 和 buffer_size 以应对网络波动
            udp_url = f"udp://@:{self.udp_port}?overrun_nonfatal=1&fifo_size=50000000&buffer_size=10000000"
            self.cap = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                self.release()
                raise RuntimeError("Failed to connect OpenCV to UDP stream.")
                
            print("OpenCV connected to camera stream.")
            
        except Exception as e:
            self.release()
            raise e

    def _kill_existing_process(self):
        """
        强制杀死任何正在运行的 rpicam-vid/libcamera-vid 进程
        """
        try:
            target = self.cmd_base
            subprocess.run(["pkill", "-x", target], stderr=subprocess.DEVNULL)
            time.sleep(0.5) # 给硬件一点时间释放
        except Exception:
            pass

    def read(self):
        """
            (bool, numpy.ndarray): (成功标志, 图像帧)
        """
        if not self.cap:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            self.fail_count += 1
            # 检查子进程是否已死
            if (self.process and self.process.poll() is not None) or self.fail_count >= self.max_fail:
                print("Camera stream stalled. Restarting pipeline... (视频流中断，正在重启)")
                # 打印最后的错误日志
                if self.process and self.process.poll() is not None:
                     try:
                         self.stderr_log.seek(0)
                         print(f"Last stderr: {self.stderr_log.read()[-500:]}")
                     except: pass
                
                try:
                    self._start_again()
                    self.fail_count = 0
                except Exception as e:
                    print(f"Restart failed: {e}")
            return False, None
        else:
            self.fail_count = 0
        return True, frame

    def release(self):
        """释放资源"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None
            
        if self.process:
            print("Stopping camera process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            
        if hasattr(self, 'stderr_log') and self.stderr_log:
            try:
                self.stderr_log.close()
            except: pass

    def _start_again(self):
        """重启整个流水线"""
        self.release()
        self._start_pipeline()
