"""
特征提取模块
    该模块负责从图像的感兴趣区域中提取特征向量，供分类器使用。
    提取的特征包括：
    1. YCbCr 颜色空间统计特征（均值和标准差）
    2. GLCM（灰度共生矩阵）纹理特征（能量、熵、对比度、相关性）
    总特征维度为 12 维。
"""

import cv2
import numpy as np
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    graycomatrix = None
    graycoprops = None
    print("Warning: scikit-image not found. Texture features (GLCM) will be zeroed.")

def extract_ycbcr_features(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    提取 YCbCr 颜色特征。

    计算 ROI 区域在 Cb 和 Cr 通道上的均值和标准差。

    参数:
        img:  原始 BGR 图像
        mask: ROI 掩膜

    返回:
        np.ndarray: [Mean_Cb, Std_Cb, Mean_Cr, Std_Cr]（4 维）
    """
    # 转换为 YCrCb（OpenCV 默认格式）
    # 注意: OpenCV 使用 Y-Cr-Cb 顺序，而不是 Y-Cb-Cr
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    if np.count_nonzero(mask) == 0:
        return np.zeros(4, dtype=np.float64)

    pixels_cb = Cb[mask > 0]
    pixels_cr = Cr[mask > 0]

    return np.array([
        np.mean(pixels_cb),
        np.std(pixels_cb),
        np.mean(pixels_cr),
        np.std(pixels_cr),
    ])

def extract_glcm_features(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    提取 GLCM 纹理特征。

    计算灰度共生矩阵并提取 ASM、Entropy、Contrast、Correlation，
    对 4 个方向（0°、45°、90°、135°）计算均值和标准差。

    参数:
        img:  原始 BGR 图像
        mask: ROI 掩膜

    返回:
        np.ndarray: [Mean_ASM, Mean_ENT, Mean_CON, Mean_COR,
                     Std_ASM,  Std_ENT,  Std_CON,  Std_COR]（8 维）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coords = cv2.findNonZero(mask)
    if coords is None:
        return np.zeros(8, dtype=np.float64)

    x, y, w, h = cv2.boundingRect(coords)
    roi = gray[y:y+h, x:x+w]

    if roi.shape[0] < 2 or roi.shape[1] < 2:
        return np.zeros(8, dtype=np.float64)

    if graycomatrix is None:
        return np.zeros(8, dtype=np.float64)

    # 计算 GLCM
    # 距离=1, 角度=[0, 45, 90, 135] 度
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # levels=256 (灰度级)
    glcm = graycomatrix(roi, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # 提取属性
    # 1. ASM (能量/角二阶矩)
    asm = graycoprops(glcm, 'ASM')[0]
    
    # 2. 熵 (Entropy) - skimage 没有直接提供，需手动计算
    ent = []
    for i in range(4): # 遍历 4 个角度
        p = glcm[:, :, 0, i]
        mask_p = p > 0
        entropy = -np.sum(p[mask_p] * np.log(p[mask_p]))
        ent.append(entropy)
    ent = np.array(ent)
    
    # 3. 对比度 (Contrast)
    con = graycoprops(glcm, 'contrast')[0]
    
    # 4. 相关性 (Correlation)
    cor = graycoprops(glcm, 'correlation')[0]
    
    # 计算均值和标准差，顺序：ASM, ENT, CON, COR
    means = [np.mean(asm), np.mean(ent), np.mean(con), np.mean(cor)]
    stds  = [np.std(asm),  np.std(ent),  np.std(con),  np.std(cor)]

    return np.array(means + stds)

def extract_features(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    综合特征提取。

    结合 YCbCr 颜色特征（4 维）和 GLCM 纹理特征（8 维），
    共返回 12 维特征向量。

    参数:
        img:  原始图像（BGR）
        mask: ROI 掩膜

    返回:
        np.ndarray: 12 维特征数组
    """
    f1 = extract_ycbcr_features(img, mask)
    f2 = extract_glcm_features(img, mask)
    return np.concatenate([f1, f2])
