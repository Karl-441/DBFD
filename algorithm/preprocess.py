import cv2
import numpy as np

"""
图像预处理模块
    该模块负责对输入图像进行预处理，提取疑似火焰区域。
    主要采用颜色空间阈值法（RGB 和 HSI）来生成候选掩膜，
    并通过高斯模糊、加权融合和 Otsu 阈值分割来优化结果，
    最终输出二值化的火焰候选区域掩膜。
"""

def rgb_fire_detection(img):
    """
    基于 RGB 颜色空间的火焰检测
        利用火焰在 RGB 空间中的颜色特性（红色分量占主导）来提取候选区域。
        规则: R > G > B 且 R > 180
    参数:
        img (numpy.ndarray): 输入的 BGR 图像
        
    返回值:
        numpy.ndarray: 二值掩膜 (0 或 255)，255 表示符合条件的区域
    """
    # img is BGR (OpenCV 默认格式)
    B, G, R = cv2.split(img)
    
    # 规则1: 红色通道必须大于绿色，且绿色大于蓝色 (R > G > B)
    cond1 = (R > G) & (G > B)
    # 规则2: 红色通道必须足够亮 (R > 180)
    cond2 = (R > 180)
    
    mask = cond1 & cond2
    return mask.astype(np.uint8) * 255

def hsi_fire_detection(img):
    """
    基于 HSI 颜色空间的火焰检测
    功能:
        将图像转换到 HSI (Hue, Saturation, Intensity) 空间，
        利用火焰的色调、饱和度和亮度范围进行提取。
        规则: 
          0 < H < 60 (红色到黄色区间)
          40 < S < 100 (饱和度适中)
          127 < I < 255 (高亮度)
    
    参数:
        img (numpy.ndarray): 输入的 BGR 图像
        
    返回值:
        numpy.ndarray: 二值掩膜 (0 或 255)
    """
    rows, cols, channels = img.shape
    # 归一化到 0-1 范围进行计算
    img_float = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)
    
    # 计算亮度 (Intensity)
    i = (r + g + b) / 3.0
    
    # 计算饱和度 (Saturation)
    min_rgb = np.minimum(np.minimum(r, g), b)
    s = 1 - (3 / (r + g + b + 1e-6) * min_rgb)
    s[i == 0] = 0
    
    # 计算色调 (Hue)
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b) * (g - b))
    theta = np.arccos(num / (den + 1e-6))
    
    h = theta.copy()
    h[b > g] = 2 * np.pi - h[b > g]
    h = h / (2 * np.pi) * 360 # 转换到 0-360 度
    
    # 转换回 0-255 范围以便于与阈值比较 (根据论文/经验值)
    s_255 = s * 255
    i_255 = i * 255
    
    # 应用检测规则
    cond1 = (h >= 0) & (h < 60)         # 色调在红黄之间
    cond2 = (s_255 > 40) & (s_255 < 100) # 饱和度范围
    cond3 = (i_255 > 127) & (i_255 < 255)# 高亮度区域
    
    mask = cond1 & cond2 & cond3
    return mask.astype(np.uint8) * 255

def preprocess_image(img):
    """
    图像预处理主流程
        结合 RGB 和 HSI 检测结果，通过高斯模糊和 Otsu 阈值分割生成最终的火焰候选区域掩膜。
        包含形态学操作以去噪和填充孔洞。
    
    参数:
        img (numpy.ndarray): 输入的 BGR 图像
        
    返回值:
        numpy.ndarray: 最终的二值掩膜 (Otsu 分割结果)
    
    异常:
        ValueError: 如果输入图像为 None
    """
    if img is None:
        raise ValueError("Image is None")
        
    # 1. 粗提取 (Coarse Extraction)
    mask_rgb = rgb_fire_detection(img)
    mask_hsi = hsi_fire_detection(img)
    
    # 2. 模糊与融合 (Blur and Fuse)
    # 提取前景区域
    fg_rgb = cv2.bitwise_and(img, img, mask=mask_rgb)
    fg_hsi = cv2.bitwise_and(img, img, mask=mask_hsi)
    
    # 高斯模糊，平滑噪声
    blur_rgb = cv2.GaussianBlur(fg_rgb, (5,5), 0)
    blur_hsi = cv2.GaussianBlur(fg_hsi, (5,5), 0)
    
    # 加权融合 RGB 和 HSI 的结果 (各占 50%)
    w1, w2 = 0.5, 0.5
    fusion = cv2.addWeighted(blur_rgb, w1, blur_hsi, w2, 0)
    
    # 3. Otsu 阈值分割 (Otsu Segmentation)
    fusion_gray = cv2.cvtColor(fusion, cv2.COLOR_BGR2GRAY)
    
    # Otsu 算法自动计算最佳阈值
    # 注意: 如果图像大部分是黑色的，Otsu 阈值可能会很低，但这是符合预期的
    ret, otsu_mask = cv2.threshold(fusion_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. 形态学处理 (Morphological Processing)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # 闭运算: 填充前景物体内部的小孔
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)
    # 开运算: 去除背景中的噪点 (小的白色区域)
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)
    
    return otsu_mask
