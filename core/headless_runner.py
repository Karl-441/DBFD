try:
    import cv2
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)
except ImportError:
    # 这里的异常会被 main.py 捕获，但为了模块独立性保留提示
    raise ImportError("opencv-python (cv2) is not installed. Please run 'sudo apt install python3-opencv' or 'pip install opencv-python-headless'.")

import time
import pickle
import sys
import os
import numpy as np
import config
from algorithm.preprocess import preprocess_image
from algorithm.features import extract_features
from core.output_manager import OutputManager
from core.alarm_manager import AlarmManager

import gc

"""
    该模块负责在没有GUI的情况下运行火灾检测系统。
    适用于服务器环境或资源受限的树莓派设备。
    主要流程：
    1. 初始化摄像头 (LibCamera 或 OpenCV)
    2. 加载 PNN 模型
    3. 进入主循环：读取帧 -> 预处理 -> 连通域分析 -> 特征提取 -> 分类 -> 报警
    4. 内存管理与自动垃圾回收
"""

def _setup_opencv_camera(camera_index):
    """辅助函数：初始化标准 OpenCV 摄像头"""
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    return cap

def run_headless(camera_index=0):
    """
    参数:
        camera_index: 摄像头设备索引
    """
    print(f"Starting Headless Mode on Camera {camera_index}... (正在启动无头模式)")
    
    # 1. 加载模型 (Load Model)
    pnn_model = None
    if os.path.exists(config.MODEL_PATH):
        with open(config.MODEL_PATH, 'rb') as f:
            pnn_model = pickle.load(f)
        print("PNN Model loaded. (模型已加载)")
    else:
        print(f"Error: Model not found at {config.MODEL_PATH} (未找到模型文件)")
        return

    # 2. 初始化摄像头 (Setup Camera)
    if config.USE_LIBCAMERA:
        print("Using Libcamera (Picamera2)... (使用 Libcamera)")
        try:
            from core.camera_wrapper import LibCameraWrapper
            cap = LibCameraWrapper(config.FRAME_WIDTH, config.FRAME_HEIGHT, config.FPS)
            print("Libcamera initialized.")
        except ImportError as e:
            print(f"Failed to load Libcamera: {e}")
            print("Falling back to OpenCV VideoCapture... (降级到 OpenCV)")
            cap = _setup_opencv_camera(camera_index)
            if not cap: return
    else:
        cap = _setup_opencv_camera(camera_index)
        if not cap: return

    # 初始化管理器
    output_manager = OutputManager()
    alarm_manager = AlarmManager()
    
    last_save_time = 0
    save_interval = 2.0 # 报警图片保存间隔 (秒)，防止磁盘IO过高
    
    # 优化: 预分配变量 (虽然 Python 是动态类型，但保持良好的变量管理习惯有助于内存)
    frame_count = 0
    fps_start_time = time.time()

    print("Monitoring started. Press Ctrl+C to stop. (监控已启动，按 Ctrl+C 停止)")
    
    try:
        while True:
            start_time = time.time()
            
            # 多态读取 (LibCameraWrapper 和 cv2.VideoCapture 接口一致)
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame. (无法读取帧)")
                time.sleep(1)
                continue

            # 3. 执行检测 (Detect)
            detections, _ = detect_fire(frame, pnn_model)
            
            if detections:
                print(f"FIRE DETECTED! {len(detections)} regions. (发现火情!)")
                alarm_manager.trigger()
                
                # 保存证据图片
                if time.time() - last_save_time > save_interval:
                    # 绘制检测框
                    vis = frame.copy()
                    for (x, y, w, h) in detections:
                        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(vis, "FIRE", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    path = output_manager.save_prediction(vis, detections, filename=f"fire_alert_{int(time.time())}.jpg")
                    print(f"Alert saved: {path}")
                    last_save_time = time.time()
                    
                    # 显式删除大对象
                    del vis
            
            # 显式删除帧对象
            del frame
            
            # 4. 周期性垃圾回收 (Periodic GC)
            frame_count += 1
            if frame_count % config.GC_INTERVAL == 0:
                gc.collect()
                # 计算并打印 FPS
                elapsed = time.time() - fps_start_time
                fps = config.GC_INTERVAL / elapsed
                print(f"Current FPS: {fps:.2f}")
                fps_start_time = time.time()

            # 5. 帧率控制 (FPS Control)
            process_time = time.time() - start_time
            sleep_time = max(0, (1.0/config.FPS) - process_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping headless runner... (正在停止)")
    finally:
        cap.release()
        alarm_manager.cleanup()

def detect_fire(img, pnn_model):
    """
    单帧火灾检测逻辑
        1. 缩小图像以提高处理速度
        2. 预处理 (颜色分割)
        3. 连通组件分析
        4. ROI 特征提取与分类
        img: 输入帧
        pnn_model: PNN 模型实例
        detections: 检测框列表 [(x, y, w, h), ...]
        mask: 预处理后的掩膜 (调试)
    """
    try:
        # 缩小处理分辨率
        target_w, target_h = config.PNN_TARGET_WIDTH, config.PNN_TARGET_HEIGHT
        h0, w0 = img.shape[:2]
        small = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # 预处理
        mask = preprocess_image(small)
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 坐标缩放比例
        sx = w0 / float(target_w)
        sy = h0 / float(target_h)
        
        detections = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 12: # 过滤噪点
                continue
                
            # 提取 ROI
            component_mask = np.zeros_like(mask)
            component_mask[labels == i] = 255
            roi = small[y:y+h, x:x+w]
            roi_mask = component_mask[y:y+h, x:x+w]
            
            try:
                # 特征提取与分类
                feats = extract_features(roi, roi_mask)
                pred = pnn_model.predict(feats)[0]
                if pred == 1:
                    # 还原坐标到原图
                    xr = int(x * sx)
                    yr = int(y * sy)
                    wr = int(w * sx)
                    hr = int(h * sy)
                    detections.append((xr, yr, wr, hr))
            except:
                continue
        return detections, mask
    except Exception:
        return [], None
