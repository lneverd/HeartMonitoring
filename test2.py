import numpy as np
import cv2 as cv

def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')

# 示例数据
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 使用窗口大小为3的移动窗口平均
window_size = 3
result = moving_average(data, window_size)

print(result)

img = cv.imread('./data/zhanghao.png')
b, g, r = cv.split(img)  # 分割后单独显示
sum1 = np.sum(b)
print(sum1)
x1 = sum1 / (b.shape[0] * b.shape[1])  # 该帧通道的一阶矩代表光照强度 求均值
pixelblue = x1
sum2 = np.sum(g)
x2 = sum2 / (g.shape[0] * g.shape[1])
pixelgreen = x2
sum3 = np.sum(r)
x3 = sum3 / (r.shape[0] * r.shape[1])
pixelred = x3

print(x1,x2,x3)

newimg = cv.resize(img,(1000,1000),interpolation=cv.INTER_CUBIC)

b, g, r = cv.split(newimg)  # 分割后单独显示
sum1 = np.sum(b)
print(sum1)
x1 = sum1 / (b.shape[0] * b.shape[1])  # 该帧通道的一阶矩代表光照强度 求均值
pixelblue = x1
sum2 = np.sum(g)
x2 = sum2 / (g.shape[0] * g.shape[1])
pixelgreen = x2
sum3 = np.sum(r)
x3 = sum3 / (r.shape[0] * r.shape[1])
pixelred = x3
cv.imshow('newimg',newimg)
cv.waitKey(0)
print(x1,x2,x3)