import cv2 as cv
import numpy as np

def originalPPG(img):
    b, g, r = cv.split(img)  # 分割后单独显示
    sum1 = np.sum(b)
    x1 = sum1 / (b.shape[0] * b.shape[1])  # 该帧通道的一阶矩代表光照强度 求均值
    pixelblue = x1
    sum2 = np.sum(g)
    x2 = sum2 / (g.shape[0] * g.shape[1])
    pixelgreen = x2
    sum3 = np.sum(r)
    x3 = sum3 / (r.shape[0] * r.shape[1])
    pixelred = x3
    return pixelblue ,pixelgreen,pixelred