import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_ycbcr_features(img, mask):
    """
    Extracts Mean and Std of Cb and Cr channels in the mask region.
    Returns [Mean_Cb, Std_Cb, Mean_Cr, Std_Cr]
    """
    # Convert to YCrCb (OpenCV default)
    # Note: OpenCV uses Y-Cr-Cb ordering
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    
    # Extract pixels in ROI
    if np.count_nonzero(mask) == 0:
        return [0.0, 0.0, 0.0, 0.0]
        
    pixels_cb = Cb[mask > 0]
    pixels_cr = Cr[mask > 0]
    
    mean_cb = np.mean(pixels_cb)
    std_cb = np.std(pixels_cb)
    mean_cr = np.mean(pixels_cr)
    std_cr = np.std(pixels_cr)
    
    return [mean_cb, std_cb, mean_cr, std_cr]

def extract_glcm_features(img, mask):
    """
    Extracts GLCM features: ASM, Entropy, Contrast, Correlation.
    Returns Mean and Std over 4 directions (0, 45, 90, 135).
    Vector: [Mean_ASM, Mean_ENT, Mean_CON, Mean_COR, Std_ASM, Std_ENT, Std_CON, Std_COR]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Crop to bounding box to reduce computation and background influence
    coords = cv2.findNonZero(mask)
    if coords is None:
        return [0.0] * 8
    
    x, y, w, h = cv2.boundingRect(coords)
    roi = gray[y:y+h, x:x+w]
    
    # If ROI is too small, return 0
    if roi.shape[0] < 2 or roi.shape[1] < 2:
        return [0.0] * 8

    # Compute GLCM
    # Distances=1, Angles=[0, 45, 90, 135] degrees
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # levels=256
    glcm = graycomatrix(roi, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # Extract properties
    # 1. ASM (Energy)
    asm = graycoprops(glcm, 'ASM')[0]
    
    # 2. Entropy (ENT) - Manual computation
    ent = []
    for i in range(4): # 4 angles
        p = glcm[:, :, 0, i]
        mask_p = p > 0
        entropy = -np.sum(p[mask_p] * np.log(p[mask_p]))
        ent.append(entropy)
    ent = np.array(ent)
    
    # 3. Contrast (CON)
    con = graycoprops(glcm, 'contrast')[0]
    
    # 4. Correlation (COR)
    cor = graycoprops(glcm, 'correlation')[0]
    
    # Calculate Mean and Std
    # Order: ASM, ENT, CON, COR
    means = [np.mean(asm), np.mean(ent), np.mean(con), np.mean(cor)]
    stds = [np.std(asm), np.std(ent), np.std(con), np.std(cor)]
    
    return means + stds

def extract_features(img, mask):
    """
    Combines YCbCr and GLCM features.
    Total 12 dimensions.
    """
    f1 = extract_ycbcr_features(img, mask)
    f2 = extract_glcm_features(img, mask)
    return np.array(f1 + f2)
