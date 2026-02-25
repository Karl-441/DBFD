import time
import threading
import logging
import sys
import os
import config

"""
    管理硬件报警模块
    支持GPIO控制，兼容gpiozero和 RPi.GPIO
"""

# 尝试导入 GPIO 库，优先使用 gpiozero
try:
    from gpiozero import OutputDevice
    from gpiozero.pins.lgpio import LGPIOFactory
    GPIO_LIB = "gpiozero"
except ImportError:
    # 降级到 RPi.GPIO (Legacy)
    try:
        import RPi.GPIO as GPIO
        GPIO_LIB = "RPi.GPIO"
    except ImportError:
        GPIO_LIB = None

logger = logging.getLogger(__name__)

class AlarmManager:
    def __init__(self):
        """
        初始化报警管理器
        """
        self.pin = config.ALARM_GPIO_PIN
        self.active_high = config.ALARM_ACTIVE_HIGH
        self.cooldown = config.ALARM_COOLDOWN
        
        # 状态标志
        self.is_alarming = False
        self.last_trigger_time = 0
        self.alarm_thread = None
        self.running = True
        self.device = None
        self.gpio_ok = False
        
        self._setup_gpio()

    def _setup_gpio(self):
        """
            根据可用的库初始化 GPIO 引脚。如果初始化失败，系统将继续运行但无报警功能
        """
        if GPIO_LIB == "gpiozero":
            try:
                # 使用 OutputDevice 进行通用控制
                # active_high=False 意味着 on() 会输出低电平 (针对低电平触发模块)
                # initial_value=False 意味着初始状态为 "Off" (即高电平，如果 active_high=False)
                self.device = OutputDevice(
                    self.pin, 
                    active_high=self.active_high, 
                    initial_value=False
                )
                self.gpio_ok = True
                logger.info(f"Alarm Manager initialized using gpiozero on GPIO {self.pin}")
            except Exception as e:
                logger.error(f"Failed to setup Alarm (gpiozero): {e}")
                self.gpio_ok = False
                
        elif GPIO_LIB == "RPi.GPIO":
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.pin, GPIO.OUT)
                
                # 初始化状态
                initial_state = GPIO.LOW if self.active_high else GPIO.HIGH
                GPIO.output(self.pin, initial_state)
                
                self.gpio_ok = True
                logger.info(f"Alarm Manager initialized using RPi.GPIO on GPIO {self.pin}")
            except Exception as e:
                logger.error(f"Failed to setup Alarm (RPi.GPIO): {e}")
                self.gpio_ok = False
        else:
            logger.warning("No GPIO library found. Alarm disabled (Simulation Mode). (未找到 GPIO 库，进入模拟模式)")
            self.gpio_ok = False

    def _turn_on(self):
        """打开报警器 (Turn On Alarm)"""
        if self.gpio_ok:
            try:
                if self.device: # gpiozero
                    self.device.on()
                else: # RPi.GPIO
                    state = GPIO.HIGH if self.active_high else GPIO.LOW
                    GPIO.output(self.pin, state)
            except Exception as e:
                logger.error(f"Error turning alarm ON: {e}")
        logger.info("ALARM ON! (报警开启)")

    def _turn_off(self):
        """关闭报警器"""
        if self.gpio_ok:
            try:
                if self.device: # gpiozero
                    self.device.off()
                else: # RPi.GPIO
                    state = GPIO.LOW if self.active_high else GPIO.HIGH
                    GPIO.output(self.pin, state)
            except Exception as e:
                logger.error(f"Error turning alarm OFF: {e}")
        logger.info("ALARM OFF (报警关闭)")

    def reconfigure(self):
        """重新加载配置并初始化 GPIO"""
        self.cleanup_gpio()
        self.pin = config.ALARM_GPIO_PIN
        self.active_high = config.ALARM_ACTIVE_HIGH
        self.cooldown = config.ALARM_COOLDOWN
        self._setup_gpio()
        logger.info(f"AlarmManager reconfigured: Pin={self.pin}, ActiveHigh={self.active_high}")

    def trigger(self):
        """
            触发报警。如果已经在报警中，则重置冷却计时器，延长报警时间。
            启动后台线程监控报警持续时间。
        """
        # 检查配置是否变更
        if self.pin != config.ALARM_GPIO_PIN or self.active_high != config.ALARM_ACTIVE_HIGH:
            self.reconfigure()

        self.last_trigger_time = time.time()
        
        if not self.is_alarming:
            self.is_alarming = True
            self._turn_on()
            # 启动守护线程监控超时自动关闭
            self.alarm_thread = threading.Thread(target=self._monitor_alarm, daemon=True)
            self.alarm_thread.start()

    def stop(self):
        """
        停止报警 (Stop Alarm)
        检查配置更新，并确保处于非报警状态。
        """
        # 1. 检查配置是否需要更新
        if self.pin != config.ALARM_GPIO_PIN or self.active_high != config.ALARM_ACTIVE_HIGH:
            self.reconfigure()
            # reconfigure 已经重置了 GPIO 状态，不需要再调 _turn_off
            # 但我们需要重置内部状态
            self.is_alarming = False
            return

        # 2. 如果正在报警，强制关闭
        if self.is_alarming:
            self.is_alarming = False
            self._turn_off()
            logger.info("Alarm stopped (Signal Lost)")

    def cleanup_gpio(self):
        """释放 GPIO 资源"""
        if self.gpio_ok:
            try:
                if self.device:
                    self.device.close()
                    self.device = None
                elif GPIO_LIB == "RPi.GPIO":
                    GPIO.cleanup(self.pin)
            except:
                pass
        self.gpio_ok = False

    def cleanup(self):
        """清理资源 (Cleanup)"""
        self.running = False
        self._turn_off()
        self.cleanup_gpio()
