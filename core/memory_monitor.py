import psutil
import time
import threading
import logging
import os
import gc
import ctypes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryMonitor:
    def __init__(self, threshold_mb=400, check_interval=5):
        self.threshold_mb = threshold_mb
        self.check_interval = check_interval
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Memory monitor started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Memory monitor stopped.")

    def _monitor_loop(self):
        while self.running:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024

            if rss_mb > self.threshold_mb:
                logger.warning(f"High memory usage detected: {rss_mb:.2f} MB (Threshold: {self.threshold_mb} MB)")
                self._attempt_cleanup()
            
            # Also check system wide memory
            sys_mem = psutil.virtual_memory()
            if sys_mem.percent > 90:
                 logger.warning(f"System memory critical: {sys_mem.percent}% used.")
                 self._attempt_cleanup()

            time.sleep(self.check_interval)

    def _attempt_cleanup(self):
        logger.info("Attempting memory cleanup...")
        gc.collect()
        
        # Aggressive memory release for Linux (malloc_trim)
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            logger.info("malloc_trim(0) called.")
        except Exception:
            pass # Not on Linux or libc not found

        # Add other cleanup logic here if needed (e.g. clearing image buffers)
        logger.info("Cleanup finished.")

if __name__ == "__main__":
    # Test
    monitor = MemoryMonitor(threshold_mb=100) # Low threshold for test
    monitor.start()
    try:
        data = []
        while True:
            data.append(' ' * 1024 * 1024) # Allocate 1MB
            time.sleep(0.1)
    except KeyboardInterrupt:
        monitor.stop()
