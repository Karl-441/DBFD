import psutil
import time
import threading
import logging
import os
import gc
import ctypes

"""
内存监控器
    后台运行的守护线程，实时监控当前进程和系统的内存使用情况。
    当内存使用超过阈值时，自动触发 Python 垃圾回收和系统级内存释放,防止程序因OOM被系统杀掉。
"""

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryMonitor:
    def __init__(self, threshold_mb=400, check_interval=5):
        """
        初始化内存监控器
        参数:
            threshold_mb (int): 触发清理的内存阈值 (MB)
            check_interval (int): 检查间隔 (秒)
        """
        self.threshold_mb = threshold_mb
        self.check_interval = check_interval
        self.running = False
        self.thread = None

    def start(self):
        """启动监控线程"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Memory monitor started. (内存监控已启动)")

    def stop(self):
        """停止监控线程"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Memory monitor stopped. (内存监控已停止)")

    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024 # 转换为 MB

            # 检查当前进程内存
            if rss_mb > self.threshold_mb:
                logger.warning(f"High memory usage detected: {rss_mb:.2f} MB (Threshold: {self.threshold_mb} MB)")
                self._attempt_cleanup()
            
            # 检查系统总内存
            sys_mem = psutil.virtual_memory()
            if sys_mem.percent > 90:
                 logger.warning(f"System memory critical: {sys_mem.percent}% used. (系统内存告急)")
                 self._attempt_cleanup()

            time.sleep(self.check_interval)

    def _attempt_cleanup(self):
        """
        执行内存清理,调用 Python 的 gc.collect() 回收循环引用的对象,调用 libc.malloc_trim(0) 强制归还空闲内存给操作系统 
        """
        logger.info("Attempting memory cleanup... (正在尝试内存清理)")
        gc.collect()
        
        # Linux 上的激进内存释放
        # Python 的 GC 回收了对象，但 glibc 分配器可能不会立即归还内存给 OS
        # malloc_trim 强制执行这一归还操作
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            logger.info("malloc_trim(0) called.")
        except Exception:
            pass # 非 Linux 系统或找不到 libc，忽略

        logger.info("Cleanup finished. (清理完成)")

if __name__ == "__main__":
    # 测试代码
    monitor = MemoryMonitor(threshold_mb=100) # 设置低阈值用于测试
    monitor.start()
    try:
        data = []
        while True:
            data.append(' ' * 1024 * 1024) # 每次分配 1MB
            time.sleep(0.1)
    except KeyboardInterrupt:
        monitor.stop()
