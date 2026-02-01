from ultralytics import YOLO
import cv2
import argparse
import os
import sys

"""
YOLOv8 检测工具 
    用于使用 YOLOv8 模型对单张图片进行火灾检测的命令行工具。
    主要用于模型效果测试和验证，非实时运行的主程序。
    支持自动搜索本地已训练的最佳模型权重。
"""

# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.output_manager import OutputManager

def main():
    """
    主函数 (Main Function)
        1. 解析命令行参数 (--image, --model)
        2. 自动查找可能的模型权重文件
        3. 加载 YOLO 模型
        4. 执行推理并保存带标注的结果图片
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Fire Detection Tool")
    parser.add_argument('--image', type=str, required=True, help="Path to image / 图片路径")
    
    # 默认模型名称
    # Ultralytics 默认将训练结果保存在 'runs/detect/train/weights/best.pt'
    default_model = 'yolov8n.pt'
    
    # 检查常见的训练权重路径
    # 优先使用本地训练好的最佳权重，如果找不到则使用预训练模型或传入的参数
    possible_paths = [
        r'runs/detect/train/weights/best.pt',
        r'd:/Github/DBFD/runs/detect/train/weights/best.pt',
        r'd:/Github/DBFD/output/models/yolo_auto_train/exp/weights/best.pt'
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            default_model = p
            break
        
    parser.add_argument('--model', type=str, default=default_model, help="Path to model .pt file / 模型路径")
    args = parser.parse_args()

    print(f"Loading model {args.model}... (正在加载模型)")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return

    print(f"Predicting on {args.image}... (正在预测)")
    results = model(args.image)
    
    output_manager = OutputManager()
    
    # 可视化并保存结果
    for i, r in enumerate(results):
        # plot() 返回一个 BGR 格式的 numpy 数组，包含绘制了边界框的图像
        im_array = r.plot()  
        
        # 使用 OutputManager 保存结果
        output_path = output_manager.save_prediction(im_array, [], filename=f"yolo_result_{os.path.basename(args.image)}")
        print(f"Result saved to {output_path} (结果已保存)")

if __name__ == "__main__":
    main()
