import sys
import argparse
import time
import os

"""
主程序入口 
    项目的启动脚本。负责解析命令行参数，初始化系统环境
    启动内存监控，并根据参数选择启动 GUI 模式或无头模式 
    如果 GUI 启动失败，会自动尝试进入无头模式。
"""

# 将项目根目录添加到系统路径，确保能导入 core, algorithm 等模块
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

# --- 自动引导虚拟环境 (Auto-venv Bootstrapping) ---
def setup_venv():
    """
    自动检测并加载项目根目录下的 venv，解决依赖检测不到的问题
    """
    import platform
    venv_path = os.path.join(root_dir, "venv")
    if os.path.exists(venv_path):
        # 根据系统确定 site-packages 路径
        if platform.system() == "Windows":
            site_pkg = os.path.join(venv_path, "Lib", "site-packages")
        else:
            # Linux/Pi 路径通常包含 python 版本号
            import glob
            lib_path = os.path.join(venv_path, "lib", "python*", "site-packages")
            matches = glob.glob(lib_path)
            site_pkg = matches[0] if matches else None
        
        if site_pkg and os.path.exists(site_pkg) and site_pkg not in sys.path:
            sys.path.insert(0, site_pkg)
            print(f"[BOOT] Venv site-packages injected: {site_pkg}")

setup_venv()

import config
from core.memory_monitor import MemoryMonitor

def main():
    """
    主函数
        1. 解析命令行参数 (--headless, --camera)
        2. 初始化并启动内存监控器
        3. 根据模式启动应用
        4. 处理异常和清理资源
    """
    # 初始化参数解析器
    parser = argparse.ArgumentParser(description="DBFD Raspberry Pi Edition (树莓派火灾检测系统)")
    # --headless: 不显示界面，适合低内存环境或服务器模式
    parser.add_argument("--headless", action="store_true", help="Run without GUI (recommended for low RAM) / 无界面运行 (低内存推荐)")
    # --camera: 指定摄像头索引
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index / 摄像头索引")
    args = parser.parse_args()

    print(f"DBFD-Raspberry Starting... (正在启动...)")
    print(f"Mode: {'Headless (无头模式)' if args.headless else 'GUI (图形界面模式)'}")
    print(f"Memory Limit: {config.MAX_MEMORY_MB} MB (内存限制)")

    # 启动内存监控器
    # 用于在内存不足时自动执行垃圾回收或警告
    monitor = MemoryMonitor(threshold_mb=config.MAX_MEMORY_MB)
    monitor.start()

    try:
        if args.headless:
            # 无头模式
            try:
                from core.headless_runner import run_headless
                run_headless(args.camera)
            except ImportError as e:
                # 缺少依赖时的错误处理
                print("\nCRITICAL ERROR: Missing dependencies for Headless Mode.")
                print(f"Details: {e}")
                print("Please run: sudo apt install python3-opencv")
                print("Or: pip install opencv-python-headless\n")
                sys.exit(1)
        else:
            # GUI 模式
            try:
                # 延迟导入 PyQt6，节省内存
                from PyQt6.QtWidgets import QApplication
                from ui.gui import MainWindow
                
                app = QApplication(sys.argv)
                window = MainWindow()
                window.show()
                sys.exit(app.exec())
            except ImportError as e:
                # GUI 库缺失时的自动降级处理
                print(f"GUI libraries missing or failed to load: {e}")
                print("Attempting fallback to headless mode... (尝试降级到无头模式)")
                try:
                    from core.headless_runner import run_headless
                    run_headless(args.camera)
                except ImportError as e2:
                    print("\nCRITICAL ERROR: Failed to fallback to headless mode.")
                    print(f"Details: {e2}")
                    print("Please ensure OpenCV is installed: sudo apt install python3-opencv\n")
                    sys.exit(1)
            except Exception as e:
                 print(f"Critical Error in GUI: {e}")
                 # 确保在退出前停止监控线程
                 monitor.stop()
                 sys.exit(1)

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 中断
        print("Stopping... (正在停止)")
    finally:
        # 最终清理
        monitor.stop()

if __name__ == "__main__":
    main()
