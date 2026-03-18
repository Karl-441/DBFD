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
    # 尝试直接使用 lgpio (树莓派 5 推荐)
    try:
        import lgpio
        GPIO_LIB = "lgpio"
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
        self.device = None # 用于 gpiozero
        self.chip = None   # 用于 lgpio
        self.gpio_ok = False
        
        self._setup_gpio()

    def _setup_gpio(self):
        """
            根据可用的库初始化 GPIO 引脚。如果初始化失败，系统将继续运行但无报警功能
        """
        if GPIO_LIB == "gpiozero":
            try:
                # 显式尝试使用 LGPIO 驱动 (Pi 5 兼容)
                try:
                    factory = LGPIOFactory()
                    self.device = OutputDevice(self.pin, active_high=self.active_high, pin_factory=factory)
                except:
                    self.device = OutputDevice(self.pin, active_high=self.active_high)
                
                self.gpio_ok = True
                logger.info(f"Alarm Manager initialized using gpiozero on GPIO {self.pin}")
            except Exception as e:
                logger.error(f"Failed to setup Alarm (gpiozero): {e}")
                self.gpio_ok = False
        
        elif GPIO_LIB == "lgpio":
            try:
                self.chip = lgpio.gpiochip_open(0) # RP1 chip on Pi 5 is usually 0
                lgpio.gpio_claim_output(self.chip, self.pin)
                self.gpio_ok = True
                logger.info(f"Alarm Manager initialized using lgpio on GPIO {self.pin}")
            except Exception as e:
                logger.error(f"Failed to setup Alarm (lgpio): {e}")
                self.gpio_ok = False
                
        elif GPIO_LIB == "RPi.GPIO":
            try:
                # 只有非 Pi 5 环境下 RPi.GPIO 才能正常确定 SOC 地址
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
            logger.warning("No GPIO library found. Alarm disabled.")
            self.gpio_ok = False

    def _turn_on(self):
        """打开报警器"""
        if self.gpio_ok:
            try:
                if GPIO_LIB == "gpiozero":
                    self.device.on()
                elif GPIO_LIB == "lgpio":
                    val = 1 if self.active_high else 0
                    lgpio.gpio_write(self.chip, self.pin, val)
                elif GPIO_LIB == "RPi.GPIO":
                    state = GPIO.HIGH if self.active_high else GPIO.LOW
                    GPIO.output(self.pin, state)
            except Exception as e:
                logger.error(f"Error turning alarm ON: {e}")
        logger.info("ALARM ON! (报警开启)")

    def _turn_off(self):
        """关闭报警器"""
        if self.gpio_ok:
            try:
                if GPIO_LIB == "gpiozero":
                    self.device.off()
                elif GPIO_LIB == "lgpio":
                    val = 0 if self.active_high else 1
                    lgpio.gpio_write(self.chip, self.pin, val)
                elif GPIO_LIB == "RPi.GPIO":
                    state = GPIO.LOW if self.active_high else GPIO.HIGH
                    GPIO.output(self.pin, state)
            except Exception as e:
                logger.error(f"Error turning alarm OFF: {e}")
        logger.info("ALARM OFF. (报警关闭)")

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
        打开报警器
        """
        # 检查配置是否变更
        if self.pin != config.ALARM_GPIO_PIN or self.active_high != config.ALARM_ACTIVE_HIGH:
            self.reconfigure()

        if not self.is_alarming:
            self.is_alarming = True
            self._turn_on()
            logger.info("Alarm Triggered (ON)")

    def stop(self):
        """
        关闭报警器
        """
        if self.is_alarming:
            self.is_alarming = False
            self._turn_off()
            logger.info("Alarm Stopped (OFF)")

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
        """清理资源"""
        self.running = False
        self._turn_off()
        self.cleanup_gpio()
