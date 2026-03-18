import psutil
import time
import sys
import os
import subprocess
import json
import threading
import signal

class SysMonitor:
    def __init__(self, output_dir, sampling_hz=1):
        self.output_dir = output_dir
        self.interval = 1.0 / sampling_hz
        self.running = False
        self.data_files = {
            'cpu': os.path.join(output_dir, 'csv/cpu_stats.csv'),
            'gpu': os.path.join(output_dir, 'csv/gpu_stats.csv'),
            'temp': os.path.join(output_dir, 'csv/temp_stats.csv')
        }
        self._ensure_dirs()
        # 注册信号处理，确保 kill 命令能正常停止并保存数据
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"Monitor received signal {signum}, stopping...", file=sys.stderr)
        self.running = False

    def _ensure_dirs(self):
        os.makedirs(os.path.join(self.output_dir, 'csv'), exist_ok=True)

    def get_sys_info(self):
        """抓取准确的系统硬件信息"""
        info = {
            "hw": {
                "SoC": "Unknown",
                "RAM": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                "Cores": psutil.cpu_count(logical=False),
                "Threads": psutil.cpu_count(logical=True),
                "Cooling": "Detected"
            },
            "sw": {
                "OS": "Linux",
                "Kernel": os.uname().release,
                "Python": sys.version.split()[0]
            }
        }
        
        # 尝试获取 SoC 型号
        try:
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    info["hw"]["SoC"] = f.read().replace('\x00', '').strip()
        except: pass

        # 尝试获取固件信息
        try:
            res = subprocess.check_output(["vcgencmd", "version"], stderr=subprocess.DEVNULL).decode()
            info["sw"]["Firmware"] = res.split('\n')[0]
        except: pass

        return info

    def monitor_loop(self):
        # 预热并设置基准
        psutil.cpu_percent(interval=None)
        psutil.cpu_times_percent(interval=None)
        
        try:
            # 确保文件所在的目录存在
            os.makedirs(os.path.dirname(self.data_files['cpu']), exist_ok=True)
            
            with open(self.data_files['cpu'], 'w') as f_cpu, \
                 open(self.data_files['gpu'], 'w') as f_gpu, \
                 open(self.data_files['temp'], 'w') as f_temp:
                
                f_cpu.write("timestamp,user,system,idle,iowait,cpu_total_percent\n")
                f_gpu.write("timestamp,gpu_percent,gpu_freq,gpu_mem\n")
                f_temp.write("timestamp,temp_c\n")

                while self.running:
                    ts = time.time()
                    
                    # 使用指定的 interval 进行阻塞采样，这比单独使用 sleep 更准确
                    cpu_p = psutil.cpu_percent(interval=self.interval)
                    cpu_t = psutil.cpu_times_percent(interval=None)
                    f_cpu.write(f"{ts},{cpu_t.user},{cpu_t.system},{cpu_t.idle},{cpu_t.iowait},{cpu_p}\n")
                    f_cpu.flush()
                    
                    # 温度采样
                    temp = 0.0
                    try:
                        temp_path = "/sys/class/thermal/thermal_zone0/temp"
                        if os.path.exists(temp_path):
                            with open(temp_path, "r") as ft:
                                temp = int(ft.read().strip()) / 1000.0
                    except: pass
                    f_temp.write(f"{ts},{temp}\n")
                    f_temp.flush()

                    # GPU 采样
                    gpu_freq = "0"
                    gpu_mem = "0"
                    if os.path.exists("/usr/bin/vcgencmd"):
                        try:
                            # 频率采样
                            res_f = subprocess.check_output(["vcgencmd", "measure_clock", "v3d"], stderr=subprocess.DEVNULL).decode()
                            if '=' in res_f:
                                gpu_freq = res_f.split('=')[1].strip()
                            
                            # 内存采样
                            res_m = subprocess.check_output(["vcgencmd", "get_mem", "gpu"], stderr=subprocess.DEVNULL).decode()
                            if '=' in res_m:
                                gpu_mem = res_m.split('=')[1].strip().replace('M', '')
                        except: pass
                    f_gpu.write(f"{ts},0,{gpu_freq},{gpu_mem}\n")
                    f_gpu.flush()
                    
        except Exception as e:
            print(f"Monitor error in loop: {e}", file=sys.stderr)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)

if __name__ == "__main__":
    # 作为独立脚本运行时的简单逻辑
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--hz", type=int, default=1)
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()

    monitor = SysMonitor(args.dir, args.hz)
    
    if args.info:
        print(json.dumps(monitor.get_sys_info()))
        sys.exit(0)

    monitor.start()
    print("Monitor started. Press Ctrl+C to stop.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("Monitor stopped.")
