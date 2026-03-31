"""
无头模式运行器
    该模块负责在没有 GUI 的情况下运行火灾检测系统。
    适用于服务器环境或资源受限的树莓派设备。
    主要流程：
    1. 初始化摄像头 (LibCamera 或 OpenCV)
    2. 加载 PNN 模型
    3. 进入主循环：读取帧 -> 预处理 -> 连通域分析 -> 特征提取 -> 分类 -> 报警
    4. 内存管理与自动垃圾回收
"""

try:
    import cv2
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)
except ImportError:
    raise ImportError("opencv-python (cv2) 未安装，请运行 'sudo apt install python3-opencv' 或 'pip install opencv-python-headless'。")

import time
import pickle
import sys
import os
import logging
import numpy as np
from pathlib import Path
import config
from core.output_manager import OutputManager
from core.alarm_manager import AlarmManager

import gc

from algorithm.fusion import run_pnn_pipeline

logger = logging.getLogger(__name__)

def _setup_opencv_camera(camera_index: int):
    """辅助函数：初始化标准 OpenCV 摄像头。失败时返回 None。"""
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    if not cap.isOpened():
        logger.error(f"无法打开摄像头索引 {camera_index}")
        return None
    return cap

def run_headless(camera_index: int = 0) -> None:
    """
    参数:
        camera_index: 摄像头设备索引
    """
    logger.info(f"启动无头模式，摄像头索引: {camera_index}")

    # 1. 加载模型 (Load Model)
    pnn_model = None
    model_path = Path(config.MODEL_PATH)
    if model_path.exists():
        with model_path.open('rb') as f:
            pnn_model = pickle.load(f)
        logger.info(f"PNN 模型已加载: {model_path}")
    else:
        logger.error(f"未找到 PNN 模型文件: {model_path}")
        return

    # 2. 初始化摄像头 (Setup Camera)
    if config.USE_LIBCAMERA:
        logger.info("正在使用 LibCamera 初始化摄像头...")
        try:
            from core.camera_wrapper import LibCameraWrapper
            cap = LibCameraWrapper(config.FRAME_WIDTH, config.FRAME_HEIGHT, config.FPS)
            logger.info("LibCamera 初始化成功。")
        except ImportError as e:
            logger.warning(f"LibCamera 加载失败: {e}，降级到 OpenCV VideoCapture...")
            cap = _setup_opencv_camera(camera_index)
            if not cap:
                return
    else:
        cap = _setup_opencv_camera(camera_index)
        if not cap:
            return

    # 初始化管理器
    output_manager = OutputManager()
    alarm_manager = AlarmManager()
    
    last_save_time = 0
    save_interval = 2.0 # 报警图片保存间隔 (秒)，防止磁盘IO过高
    frame_count = 0
    fps_start_time = time.time()

    logger.info("监控已启动，按 Ctrl+C 停止。")
    
    perf_log = os.getenv("DBFD_PERF_TEST") == "1"
    perf_file = None
    if perf_log:
        perf_file = open("csv/perf_stats.csv", "w")
        perf_file.write("timestamp,latency_ms,fps\n")

    try:
        while True:
            start_time = time.time()
            
            # 多态读取 (LibCameraWrapper 和 cv2.VideoCapture 接口一致)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("无法读取视频帧，等待 1 秒后重试...")
                time.sleep(1)
                continue

            # 3. 执行检测 (Detect)
            detections, _ = detect_fire(frame, pnn_model)
            
            # 计算延迟和 FPS
            process_time = time.time() - start_time
            latency_ms = process_time * 1000
            
            if perf_log:
                perf_file.write(f"{time.time()},{latency_ms:.2f},0\n")

            if detections:
                logger.warning(f"检测到火情！共 {len(detections)} 个区域。")
                alarm_manager.trigger()

                # 保存证据图片
                if time.time() - last_save_time > save_interval:
                    # 绘制检测框
                    vis = frame.copy()
                    for (x, y, w, h) in detections:
                        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(vis, "FIRE", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    path = output_manager.save_prediction(vis, detections, filename=f"fire_alert_{int(time.time())}.jpg")
                    logger.info(f"报警图片已保存: {path}")
                    last_save_time = time.time()

                    # 显式删除大对象
                    del vis
            
            # 显式删除帧对象
            del frame
            
            # 4. 周期性垃圾回收 (Periodic GC)
            frame_count += 1
            if frame_count % config.GC_INTERVAL == 0:
                gc.collect()
                # 计算并记录 FPS
                elapsed = time.time() - fps_start_time
                fps = config.GC_INTERVAL / elapsed
                logger.info(f"当前 FPS: {fps:.2f}")
                if perf_log:
                    perf_file.write(f"{time.time()},0,{fps:.2f}\n")
                fps_start_time = time.time()

            # 5. 帧率控制 (FPS Control)
            process_time = time.time() - start_time
            sleep_time = max(0, (1.0/config.FPS) - process_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("正在停止无头模式...")
    finally:
        if perf_file:
            perf_file.close()
        cap.release()
        alarm_manager.cleanup()

def detect_fire(img: np.ndarray, pnn_model) -> tuple:
    """
    单帧火灾检测（仅 PNN，无头模式专用）。

    参数:
        img:        输入帧
        pnn_model:  PNN 模型实例

    返回:
        tuple(list, np.ndarray | None)
            - detections: [(x, y, w, h), ...]
            - mask:       预处理后的掩膜（调试用），出错时为 None
    """
    pnn_results = run_pnn_pipeline(
        img, pnn_model,
        config.PNN_TARGET_WIDTH,
        config.PNN_TARGET_HEIGHT,
    )
    detections = [tuple(r['box']) for r in pnn_results]

    # 单独生成 mask 供调试展示
    try:
        target_w, target_h = config.PNN_TARGET_WIDTH, config.PNN_TARGET_HEIGHT
        small = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        from algorithm.preprocess import preprocess_image
        mask = preprocess_image(small)
    except Exception:
        mask = None

    return detections, mask
