import numpy as np
import cv2

def normalized(pixellist):
    pixellist = np.array(pixellist)
    result = np.zeros(len(pixellist), dtype=np.float32)
    cv2.normalize(pixellist, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # 归一化处理

    return result