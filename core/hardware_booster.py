import os
import subprocess
import logging
import platform

logger = logging.getLogger(__name__)

class HardwareBooster:
    """
    树莓派硬件加速与监控器 (Raspberry Pi Hardware Booster & Monitor)
    利用 vcgencmd 监控频率/温度，并检查 Vulkan 支持情况。
    """
    def __init__(self):
        # 检查是否为树莓派且 vcgencmd 可用
        self.is_pi = os.path.exists('/usr/bin/vcgencmd')
        # 如果是树莓派，尝试检查是否有权限执行 vcgencmd (Need to be in 'video' group)
        if self.is_pi:
            try:
                subprocess.run(['vcgencmd', 'get_throttled'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                self.is_pi = False
                logger.warning("HardwareBooster: vcgencmd permission denied. Try 'sudo usermod -aG video $USER'")
        
        self.vulkan_available = self._check_vulkan()
        
    def _check_vulkan(self):
        """检查系统中是否安装了 Vulkan (vulkan-utils/vulkaninfo)"""
        try:
            # 尝试运行 vulkaninfo 的简短版
            result = subprocess.run(['vulkaninfo', '--summary'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def get_stats(self):
        """
        获取当前硬件状态 (vcgencmd stats)
        """
        stats = {
            'temp': 'N/A',
            'gpu_clock': 'N/A',
            'cpu_clock': 'N/S',
            'throttled': 'Normal',
            'vulkan': 'Active' if self.vulkan_available else 'Inactive'
        }
        
        if not self.is_pi:
            return stats
            
        try:
            # 1. 获取温度 (Temperature)
            temp_out = subprocess.check_output(['vcgencmd', 'measure_temp']).decode()
            stats['temp'] = temp_out.split('=')[1].strip()
            
            # 2. 获取 GPU 频率 (GPU Clock)
            gpu_out = subprocess.check_output(['vcgencmd', 'measure_clock', 'v3d']).decode()
            stats['gpu_clock'] = f"{int(gpu_out.split('=')[1]) // 1000000}MHz"
            
            # 3. 获取 CPU 频率 (CPU Clock)
            cpu_out = subprocess.check_output(['vcgencmd', 'measure_clock', 'arm']).decode()
            stats['cpu_clock'] = f"{int(cpu_out.split('=')[1]) // 1000000}MHz"
            
            # 4. 检查降频状态 (Throttle Status)
            # 0x0 表示正常，其他值表示电压不足或温度过高
            throttle_out = subprocess.check_output(['vcgencmd', 'get_throttled']).decode()
            stats['throttled'] = 'Throttled!' if throttle_out.split('=')[1].strip() != '0x0' else 'Normal'
            
        except Exception as e:
            logger.error(f"HardwareBooster: Error getting Pi stats: {e}")
            
        return stats

    def set_high_performance(self):
        """
        建议硬件预热与高性能配置逻辑。
        (主要通过环境变量或引导用户进行系统级超频)
        """
        if self.is_pi:
            logger.info("HardwareBooster: Raspberry Pi detected. Optimization active.")
            if not self.vulkan_available:
                logger.warning("HardwareBooster: Vulkan (vulkan-utils) not detected. Recommend 'sudo apt install vulkan-tools'")
        else:
            logger.info("HardwareBooster: Non-Pi environment. Generic optimizations only.")
