"""
配置代理模块
    为了兼容现有代码并支持单例模式的配置管理器，
    本模块通过 Python 3.7+ 的模块级 __getattr__ 机制，将所有属性访问
    自动代理到 core.config_manager.cfg 单例对象上。

    用法示例:
        import config
        print(config.FRAME_WIDTH)       # 等价于 cfg.FRAME_WIDTH
        config.cfg.save_config()        # 直接操作单例，保存配置到 JSON

    实现原理:
        当 Python 解析 `config.SOME_VAR` 时，若该名称在模块命名空间中不存在，
        就会调用模块级 __getattr__(name)，由它转发到 cfg 对象。
        这样任何模块都可以用 `import config; config.X` 的方式读取配置，
        而无需关心底层单例的存在。
"""
from core.config_manager import cfg

# 导出 cfg 对象，供需要直接操作配置管理器的地方使用（如 cfg.save_config()）
cfg = cfg


def __getattr__(name: str):
    """将模块级属性访问代理到 ConfigManager 单例。"""
    return getattr(cfg, name)


def __dir__() -> list:
    """让 dir(config) 和 IDE 自动补全能枚举出所有配置项。"""
    return list(cfg.__dict__.keys()) + ['cfg']
