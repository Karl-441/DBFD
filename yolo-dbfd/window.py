# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: yolov5-jungong
File Name: window.py.py
Author: chenming
Create Date: 2021/11/8
Description：无界面摄像头检测功能
-------------------------------------------------
"""
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import LoadStreams
from utils.general import (LOGGER, check_img_size, check_imshow, check_requirements,
                           non_max_suppression, print_args, scale_coords)
import time

# 替代time_sync函数
def time_sync():
    # 返回当前时间戳，与标准time.time()功能相同
    return time.time()
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


def load_model(weights="runs/train/exp_yolov5s/weights/best.pt", device='cpu', half=False, dnn=False):
    """
    加载YOLOv5模型
    """
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    
    # 加载模型
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    
    # 配置精度
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
    
    print("模型加载完成!")
    return model, device, stride, names


def detect_camera():
    """
    无界面摄像头检测主函数
    """
    # 配置参数
    weights = "runs/train/exp_yolov5s/weights/best.pt"  # 模型路径
    device = 'cpu'  # 设备选择
    source = '0'  # 摄像头源
    imgsz = [640, 640]  # 推理尺寸
    conf_thres = 0.25  # 置信度阈值
    iou_thres = 0.45  # NMS IOU阈值
    max_det = 1000  # 最大检测数量
    classes = None  # 类别过滤
    agnostic_nms = False  # 类别无关NMS
    augment = False  # 增强推理
    visualize = False  # 可视化特征
    line_thickness = 3  # 边界框厚度
    hide_labels = False  # 隐藏标签
    hide_conf = False  # 隐藏置信度
    half = False  # 使用FP16半精度推理
    dnn = False  # 使用OpenCV DNN进行ONNX推理
    
    # 加载模型
    model, device, stride, names = load_model(weights, device, half, dnn)
    
    # 检查图像大小
    imgsz = check_img_size(imgsz, s=stride)
    
    # 设置为显示结果
    view_img = check_imshow()
    
    # 加载摄像头流
    cudnn.benchmark = True  # 设置为True以加速固定图像大小的推理
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
    bs = len(dataset)  # batch_size
    
    # 预热模型
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))
    
    print("开始摄像头检测，按ESC键退出...")
    
    # 推理循环
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        
        # 推理
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        
        # 处理预测结果
        for i, det in enumerate(pred):  # 每个图像
            seen += 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '
            
            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            
            # 绘制标注
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 调整边界框从img_size到im0大小
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 每个类的检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # 整数类
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
            
            # 打印时间（仅推理）
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            
            # 流结果
            im0 = annotator.result()
            
            # 使用OpenCV显示结果
            if view_img:
                cv2.imshow(str(p), im0)
                # 按ESC键退出
                if cv2.waitKey(1) == 27:
                    dataset.running = False
                    break
    
    # 清理
    cv2.destroyAllWindows()
    print(f'检测完成，共处理 {seen} 帧')


if __name__ == "__main__":
    try:
        # 检查依赖
        check_requirements(exclude=('tensorboard', 'thop'))
        # 运行摄像头检测
        detect_camera()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序出错: {e}")
    finally:
        # 确保清理资源
        cv2.destroyAllWindows()
