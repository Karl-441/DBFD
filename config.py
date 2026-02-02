"""
配置文件 
    为了兼容现有代码并支持单例模式的配置管理器，
    本模块代理了 core.config_manager.cfg 的属性访问。
    使用方式不变:
    import config
    print(config.FRAME_WIDTH)
    也支持访问单例对象进行保存/加载:
    config.cfg.save_config()
"""
from core.config_manager import cfg
import sys

# 导出 cfg 对象，供需要直接操作配置管理器的地方使用
cfg = cfg

# 模块级 __getattr__ (Python 3.7+)
# 当访问 config.SOME_VAR 时，自动重定向到 cfg.SOME_VAR
def __getattr__(name):
    return getattr(cfg, name)

# 为了让 dir(config) 能显示所有属性 (对 IDE 和自动补全友好)
def __dir__():
    return list(cfg.__dict__.keys()) + ['cfg']
