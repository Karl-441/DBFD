import subprocess
import os
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
        """构建 libcamera 命令行参数 (H.264 兼容模式)"""
        # 使用最基础的 H.264 参数，确保在所有 Pi 版本上都能启动
        cmd = [
            self.cmd_base,
            "-t", "0", 
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.fps),
            "--nopreview",
            "--codec", "h264",
            "--inline", # 必须内联报头，否则 OpenCV 无法在中途切入流
            "--profile", "baseline", # 强制使用 baseline profile，禁用 B 帧以降低延迟并修复 POC 错误
            "--intra", "15", # 每 15 帧强制一个 I 帧 (0.5s @ 30fps)，解决 "non-existing PPS" 同步问题
            "--denoise", "cdn_off", # 关闭降噪以减少启动和处理延迟
            "--flush", # 强制刷新输出缓冲区
            "-o", f"udp://127.0.0.1:{self.udp_port}?pkt_size=1316" # 优化 UDP 包大小
        ]
        return cmd

    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.process:
            try:
                # 杀死整个进程组，确保彻底清理
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=1.0)
            except:
                pass
            self.process = None
        
        if hasattr(self, 'stderr_log') and self.stderr_log:
            try:
                self.stderr_log.close()
            except:
                pass
        
        self._kill_existing_process()
        # 强制等待一下，确保端口释放
        time.sleep(0.5)

    def _start_pipeline(self):
        """启动摄像推流进程和 OpenCV 读取连接"""
        # 1. 强制清理现有的摄像头进程
        self._kill_existing_process()
        
        self.cmd = self._build_command()
        
        try:
            print(f"Starting Camera Process: {' '.join(self.cmd)}")
            
            # 以写模式打开日志文件，每次清空
            with open(self.log_file, "w") as f:
                f.write(f"--- Camera Log Started at {time.ctime()} ---\n")
            
            self.stderr_log = open(self.log_file, "a")
            self.process = subprocess.Popen(
                self.cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=self.stderr_log,
                preexec_fn=os.setsid # 创建新的进程组，方便彻底杀死
            )
            
            # 2. 等待进程启动并检查状态 (从 2.0s 降低到 0.8s)
            time.sleep(0.8)
            
            if self.process.poll() is not None:
                # 进程已退出，读取错误日志
                self.stderr_log.close()
                with open(self.log_file, "r") as f:
                    error_log = f.read()
                print(f"!!! CAMERA ERROR LOG !!!\n{error_log}\n------------------------")
                raise RuntimeError(f"Camera process failed to start. See log above.")
            
            # 3. 尝试连接 OpenCV (针对 H.264 优化探测)
            # 降低 probesize 和 analyzeduration 进一步减少启动时间
            udp_url = f"udp://127.0.0.1:{self.udp_port}?overrun_nonfatal=1&fifo_size=1000000"
            
            # 环境变量优化：极致降低探测时间
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "probesize;128|analyzeduration;0|fpsprobesize;1"
            
            max_retries = 5
            for i in range(max_retries):
                print(f"Connecting to H.264 stream (Attempt {i+1}/{max_retries})...")
                self.cap = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
                
                if self.cap.isOpened():
                    print("OpenCV connected to camera stream.")
                    return
                
                time.sleep(1.0)
            
            raise RuntimeError("Failed to connect OpenCV to H.264 stream after retries.")
            
        except Exception as e:
            self.release()
            raise e

    def _kill_existing_process(self):
        """
        强制杀死任何正在运行的 rpicam-vid/libcamera-vid 进程
        """
        try:
            target = self.cmd_base
            # 使用更彻底的 kill 方式
            subprocess.run(["pkill", "-9", "-x", target], stderr=subprocess.DEVNULL)
            time.sleep(0.3) # 给硬件和端口一点时间释放
        except Exception:
            pass

    def _handle_failure(self):
        """处理读取失败，必要时重启链路"""
        self.fail_count += 1
        if (self.process and self.process.poll() is not None) or self.fail_count >= self.max_fail:
            print("Camera stream stalled. Restarting pipeline... (视频流中断，正在重启)")
            if self.process and self.process.poll() is not None:
                 try:
                     self.stderr_log.seek(0)
                     print(f"Last stderr: {self.stderr_log.read()[-500:]}")
                 except: pass
            
            try:
                self._start_pipeline()
                self.fail_count = 0
            except Exception as e:
                print(f"Restart failed: {e}")
        return False

    def grab(self):
        """代理 cv2.VideoCapture.grab()"""
        if not self.cap: return False
        res = self.cap.grab()
        if not res:
            self._handle_failure()
        else:
            self.fail_count = 0
        return res

    def retrieve(self):
        """代理 cv2.VideoCapture.retrieve()"""
        if not self.cap: return False, None
        ret, frame = self.cap.retrieve()
        if not ret:
            self._handle_failure()
        else:
            self.fail_count = 0
        return ret, frame

    def read(self):
        """
            (bool, numpy.ndarray): (成功标志, 图像帧)
        """
        if not self.cap:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            self._handle_failure()
            return False, None
        else:
            self.fail_count = 0
        return True, frame

    def _start_again(self):
        """重启整个流水线"""
        self.release()
        self._start_pipeline()
