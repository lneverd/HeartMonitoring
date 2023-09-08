# -*- coding: utf-8 -*-
"""
Created on Thu May  6 22:51:50 2021

@author: 李国祥
"""

import cv2
from cv2 import dnn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import math
from scipy.fftpack import fft
from matplotlib.pylab import mpl
from scipy import signal
import time
import warnings

warnings.filterwarnings("ignore")
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号


############### 延时函数  ###############
def delay():
    startTime = time.time()
    while (1):
        if time.time() - startTime > 3.5:
            break
        ############### 获取视频函数  ###############


def video_access():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("./data/face.mp4", fourcc, 30.0, (640, 480))
    startTime = time.time()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 1)
            cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
            out.write(frame)  # 保存视频
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        if time.time() - startTime > 34:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


############### 视频处理函数  ###############
def video_process():
    videoCapture = cv2.VideoCapture()
    videoCapture.open("./data/face.mp4")
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    # fps是帧率，意思是每一秒刷新图片的数量，frames是一整段视频中总的图片数量。
    print("fps=", fps, "frames=", frames)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        cv2.imwrite("./data/face/face(%d).jpg" % i, frame)


############### 感兴趣区域（额头）提取函数  ###############
def roi():
    config_file = "opencv_face_detector.pbtxt";  ###########################绝对路径
    model_file = "opencv_face_detector_uint8.pb";  ####################绝对路径
    net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    inWidth = 300
    inHeight = 300
    confThreshold = 0.95
    for j in range(0, 900):
        w = 0
        h = 0
        frame = cv2.imread('./data/face/face(%d).jpg' % j)
        cols = frame.shape[1]
        rows = frame.shape[0]
        net.setInput(dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False))
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)
                w = xRightTop - xLeftBottom
                h = yRightTop - yLeftBottom
                w1 = 2 * w
                h1 = 2.7 * h
                print(w1)
        crops = frame[yLeftBottom + int(h / 10):yLeftBottom + int(h1 / 10),
                xLeftBottom + int(w / 3):xLeftBottom + int(w1 / 3)]
        cv2.imwrite("./data/ROI/roi(%d).jpg" % j, crops)


############### 皮尔森相关系数分析函数  ###############
def pearson(vector1, vector2):
    n = len(vector1)
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    p_sum = sum([vector1[i] * vector2[i] for i in range(n)])
    num = p_sum - (sum1 * sum2 / n)
    den = math.sqrt((sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))
    if den == 0:
        return 0.0
    return num / den


############### 原始心率提取（未除噪）函数  ###############
def original_HR():
    blue = []
    green = []
    red = []
    for i in range(0, 900):
        img = cv2.imread('./data/ROI/roi(%d).jpg' % i)
        b, g, r = cv2.split(img)  # 分割后单独显示
        sum1 = np.sum(b)
        x1 = sum1 / (b.shape[0] * b.shape[1])  # 该帧通道的一阶矩代表光照强度 求均值
        blue.append(x1)
        sum2 = np.sum(g)
        x2 = sum2 / (g.shape[0] * g.shape[1])
        green.append(x2)
        sum3 = np.sum(r)
        x3 = sum3 / (r.shape[0] * r.shape[1])
        red.append(x3)
    blue = np.array(blue)
    green = np.array(green)
    red = np.array(red)
    result_b = np.zeros(len(blue), dtype=np.float32)
    cv2.normalize(blue, result_b, alpha=-5, beta=5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # 归一化处理
    result_g = np.zeros(len(green), dtype=np.float32)
    cv2.normalize(green, result_g, alpha=-5, beta=5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    result_r = np.zeros(len(red), dtype=np.float32)
    cv2.normalize(red, result_r, alpha=-5, beta=5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    xy = np.arange(900) #画刻度的
    b_x = plt.subplot(411) #四行一列当前位置是1
    b_x1 = plt.subplot(412) #四行一列当前位置是2
    b_x2 = plt.subplot(413) #当前位置是3
    b_x.plot(xy, result_b)
    b_x1.plot(xy, result_g)
    b_x2.plot(xy, result_r)
    plt.show()
    s = np.array([blue, green, red]) #3行n列的矩阵
    ran = 2 * np.random.random([3, 3])
    mix = ran.dot(s)
    b_x = plt.subplot(411)
    b_x1 = plt.subplot(412)
    b_x2 = plt.subplot(413)
    b_x.plot(xy, mix[0, :])
    b_x1.plot(xy, mix[1, :])
    b_x2.plot(xy, mix[2, :])
    plt.show()
    ica = FastICA(n_components=3)
    mix = mix.T
    u = ica.fit_transform(mix)
    u = u.T
    b_x = plt.subplot(411)
    b_x1 = plt.subplot(412)
    b_x2 = plt.subplot(413)
    b_x.plot(xy, u[0, :])
    b_x1.plot(xy, u[1, :])
    b_x2.plot(xy, u[2, :])
    plt.show()
    # 计算相关性，与g信号相比
    print(pearson(u[0, :], result_g))
    print(pearson(u[1, :], result_g))
    print(pearson(u[2, :], result_g))
    max_num = 0
    rate_original = 0
    if abs(pearson(u[0, :], result_g)) > abs(pearson(u[1, :], result_g)):
        max_num = abs(pearson(u[0, :], result_g))
        if max_num > abs(pearson(u[2, :], result_g)):
            rate_original = u[0, :]
        else:
            rate_original = u[2, :]
    else:
        max_num = abs(pearson(u[1, :], result_g))
        if max_num > abs(pearson(u[2, :], result_g)):
            rate_original = u[1, :]
        else:
            rate_original = u[2, :]
    return rate_original


############### 运行函数  ###############
def run():
    org_hr = original_HR()
    b, a = signal.butter(4, [0.04, 0.27], 'bandpass')
    filtedData = signal.filtfilt(b, a, org_hr)  # data为要过滤的信号
    fft_y = fft(filtedData)  # 快速傅里叶变换------------------时域到频域的转化
    N = 900
    x = np.arange(N)  # 频率个数
    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    angle_y = np.angle(fft_y)  # 取复数的角度
    plt.figure()
    plt.plot(x, abs_y)
    plt.title('双边振幅谱（未归一化）')

    plt.figure()
    plt.plot(x, angle_y)
    plt.title('双边相位谱（未归一化）')
    plt.show()

    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    print(np.argmax(abs_y))
    print('当前您的心率为:')
    print((np.argmax(abs_y) / 30) * 60)


if __name__ == '__main__':
    print('3秒后开启摄像头，请在自然光充足的地方脸部正对摄像头，并露出额头！')
    delay()
    video_access()
    video_process()
    roi()
    run()
    input('Press <Enter>')

cv2.waitKey(0)
cv2.destroyAllWindows()