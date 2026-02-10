import json
from pathlib import Path

class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._init_defaults()
        return cls._instance

    def _init_defaults(self):
        # ==========================================
        # 硬件约束
        # ==========================================
        self.MAX_MEMORY_MB = 1024
        self.GC_INTERVAL = 30

        # ==========================================
        # 性能优化配置
        # ==========================================
        self.DETECT_INTERVAL = 3
        self.PNN_TARGET_WIDTH = 160
        self.PNN_TARGET_HEIGHT = 120
        self.PNN_MAX_SAMPLES = 100

        # ==========================================
        # 报警设置 
        # ==========================================
        self.ALARM_GPIO_PIN = 17
        self.ALARM_ACTIVE_HIGH = True
        self.ALARM_COOLDOWN = 5.0

        # ==========================================
        # 摄像头设置 
        # ==========================================
        self.USE_LIBCAMERA = True
        self.CAMERA_INDEX = 0
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
        self.FPS = 30

        # ==========================================
        # 算法设置 
        # ==========================================
        self.USE_PNN = True
        self.USE_YOLO = False
        self.DEVICE = 'cpu'  # cpu, 0, mps, auto

        # ==========================================
        # 路径配置pathlib
        # ==========================================
        # 获取 core 目录的父目录作为 BASE_DIR
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.MODEL_PATH = self.BASE_DIR / "model_pnn.pkl"
        self.OUTPUT_DIR = self.BASE_DIR / "output"
        self.LOG_DIR = self.BASE_DIR / "logs"

        # 确保必要的目录存在
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)

    def load_config(self, config_path="config.json"):
        """从 JSON 文件加载配置，覆盖默认值"""
        config_file = self.BASE_DIR / config_path
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self, key):
                            # 如果当前值是 Path 类型，则将加载的字符串转换为 Path
                            current_val = getattr(self, key)
                            if isinstance(current_val, Path):
                                setattr(self, key, Path(value))
                            else:
                                setattr(self, key, value)
                print(f"Config loaded from {config_file}")
            except Exception as e:
                print(f"Failed to load config: {e}")

    def save_config(self, config_path="config.json"):
        """保存当前配置到 JSON 文件"""
        config_file = self.BASE_DIR / config_path
        data = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                if isinstance(value, Path):
                    data[key] = str(value)
                else:
                    data[key] = value
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"Config saved to {config_file}")
        except Exception as e:
            print(f"Failed to save config: {e}")

# 全局单例实例
cfg = ConfigManager()
# 尝试自动加载本地配置
cfg.load_config()
