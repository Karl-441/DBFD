import cv2
import numpy as np

def rgb_fire_detection(img):
    """
    RGB Color Space Rule:
    R > G > B
    R > 180
    """
    # img is BGR
    B, G, R = cv2.split(img)
    
    cond1 = (R > G) & (G > B)
    cond2 = (R > 180)
    
    mask = cond1 & cond2
    return mask.astype(np.uint8) * 255

def hsi_fire_detection(img):
    """
    HSI Color Space Rule:
    0 < H < 60
    40 < S < 100
    127 < I < 255
    """
    rows, cols, channels = img.shape
    # Normalize to 0-1 for calculation
    img_float = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)
    
    # Intensity
    i = (r + g + b) / 3.0
    
    # Saturation
    min_rgb = np.minimum(np.minimum(r, g), b)
    s = 1 - (3 / (r + g + b + 1e-6) * min_rgb)
    s[i == 0] = 0
    
    # Hue
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b) * (g - b))
    theta = np.arccos(num / (den + 1e-6))
    
    h = theta.copy()
    h[b > g] = 2 * np.pi - h[b > g]
    h = h / (2 * np.pi) * 360 # 0-360 degrees
    
    # Convert to 0-255 scale for thresholding comparisons as per prompt numbers
    s_255 = s * 255
    i_255 = i * 255
    
    # Rules
    cond1 = (h >= 0) & (h < 60)
    cond2 = (s_255 > 40) & (s_255 < 100)
    cond3 = (i_255 > 127) & (i_255 < 255)
    
    mask = cond1 & cond2 & cond3
    return mask.astype(np.uint8) * 255

def preprocess_image(img):
    """
    Returns the binary mask of candidate fire regions.
    """
    if img is None:
        raise ValueError("Image is None")
        
    # 1. Coarse Extraction
    mask_rgb = rgb_fire_detection(img)
    mask_hsi = hsi_fire_detection(img)
    
    # 2. Blur and Fuse
    # Extract foregrounds
    fg_rgb = cv2.bitwise_and(img, img, mask=mask_rgb)
    fg_hsi = cv2.bitwise_and(img, img, mask=mask_hsi)
    
    # Blur
    blur_rgb = cv2.GaussianBlur(fg_rgb, (5,5), 0)
    blur_hsi = cv2.GaussianBlur(fg_hsi, (5,5), 0)
    
    # Weighted Fusion
    w1, w2 = 0.5, 0.5
    fusion = cv2.addWeighted(blur_rgb, w1, blur_hsi, w2, 0)
    
    # 3. Otsu Segmentation
    fusion_gray = cv2.cvtColor(fusion, cv2.COLOR_BGR2GRAY)
    
    # Otsu
    # We only care about non-zero regions, but Otsu works on the whole histogram.
    # If the image is mostly black (0), Otsu threshold might be very low.
    # However, standard Otsu is requested.
    ret, otsu_mask = cv2.threshold(fusion_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological Processing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # Close: fill holes
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)
    # Open: remove noise
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)
    
    return otsu_mask
