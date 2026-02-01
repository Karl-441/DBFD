import cv2
import time
import argparse
import sys
import os

"""
摄像头调试工具
    用于测试和调试摄像头的简单脚本
    可以列出可用分辨率，测试帧率，并验证 LibCamera 或 OpenCV 的连接性
"""

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def test_camera(index=0, width=640, height=480):
    """测试摄像头"""
    print(f"Testing Camera {index} at {width}x{height}...")
    
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Failed to open camera.")
        return
        
    print("Camera opened. Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        cv2.imshow('Camera Test', frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"FPS: {30/elapsed:.2f}")
            start_time = time.time()
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Width")
    parser.add_argument("--height", type=int, default=480, help="Height")
    args = parser.parse_args()
    
    test_camera(args.index, args.width, args.height)
