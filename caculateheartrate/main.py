import cv2 as cv
import averagepixel
import normalize
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import related
from scipy import signal
from scipy.fftpack import fft
blue_list = list()
green_list = list()
red_list = list()


def ffttransomer(rate_original):
    fps = 20
    frame_count = 300
    b, a = signal.butter(6, [0.06, 0.5], 'bandpass')
    filtedData = signal.filtfilt(b, a, rate_original, axis=0)  # data为要过滤的信号
    fft_y = fft(filtedData)  # 快速傅里叶变换------------------时域到频域的转化
    x = np.arange(frame_count)  # 频率个数
    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    plt.figure()
    plt.plot(x, abs_y)
    plt.show()
    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    print('当前您的心率为:')
    print((np.argmax(abs_y) / (frame_count / fps)) * 60)

if __name__ == '__main__':

    for i in range(300):
        img = cv.imread("../data/zxyROI/ROI1." + str(i + 1) + ".jpg")
        blue_mean, green_mean, red_mean = averagepixel.originalPPG(img)
        blue_list.append(blue_mean)
        green_list.append(green_mean)
        red_list.append(red_mean)
    blue_normalized = normalize.normalized(blue_list)
    green_normalized = normalize.normalized(green_list)
    red_normalized = normalize.normalized(red_list)

    plt.figure()
    plt.plot(blue_normalized, color='blue')
    plt.plot(green_normalized, color='green')
    plt.plot(red_normalized, color='red')


    s = np.array([blue_normalized, green_normalized, red_normalized])  # 3行n列的矩阵

    # ran = 2 * np.random.random([3, 3])
    # mix = ran.dot(s)

    fast_ica = FastICA(n_components=3)
    Sr = fast_ica.fit_transform(s)
    S = np.dot(Sr.T, s)
    plt.figure()
    plt.title("ICA used FastICA")
    plt.plot(S[0, :].T,color = 'blue')
    plt.plot(S[1, :].T,color = 'green')
    plt.plot(S[2, :].T,color = 'red')
    plt.show()

    rate_original = 0
    if abs(related.pearson(S[0, :], green_normalized)) > abs(related.pearson(S[1, :], green_normalized)):
        max_num = abs(related.pearson(S[0, :], green_normalized))
        if max_num > abs(related.pearson(S[2, :], green_normalized)):
            rate_original = S[0, :]
        else:
            rate_original = S[2, :]
    else:
        max_num = abs(related.pearson(S[1, :], green_normalized))
        if max_num > abs(related.pearson(S[2, :], green_normalized)):
            rate_original = S[1, :]
        else:
            rate_original = S[2, :]
    plt.figure()
    plt.title("original ppg signal")
    plt.plot(rate_original,color = 'blue')
    plt.show()
    ffttransomer(rate_original)



