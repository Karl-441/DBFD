import cv2
import sys
import os
import time

"""
UDP 流测试工具
    用于验证 LibCameraWrapper 推送的 UDP 视频流是否正常
    模拟接收端，连接到本地 UDP 端口并显示视频
"""

def test_udp_stream(port=1234):
    """测试 UDP 视频流"""
    # 构造 UDP URL
    # udp://@:1234 监听本地端口
    udp_url = f"udp://@:{port}?overrun_nonfatal=1&fifo_size=5000000"
    
    print(f"Connecting to {udp_url}...")
    
    cap = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("Failed to open UDP stream.")
        return
        
    print("Stream opened. Waiting for frames... (Press 'q' to quit)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received. Retrying...")
            time.sleep(0.5)
            continue
            
        cv2.imshow('UDP Stream Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    port = 1234
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    test_udp_stream(port)
