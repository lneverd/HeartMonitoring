import cv2 as cv
import averagepixel
import normalize
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import related
from scipy import signal
from scipy.fftpack import fft
import wave
import main
import decomposition as dec
import mainfreq


def ffttransomer(rate_original):
    frame_count = 300
    fps = 20


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


blue_list = list()
green_list = list()
red_list = list()
if __name__ == '__main__':

    for i in range(300):
        img = cv.imread("../data/pyramidzxy/pyramid.ROI1." + str(i + 1) + ".jpg")
        blue_mean, green_mean, red_mean = averagepixel.originalPPG(img)
        blue_list.append(blue_mean)
        green_list.append(green_mean)
        red_list.append(red_mean)

    blue_normalized = normalize.normalized(blue_list)
    green_normalized = normalize.normalized(green_list)
    red_normalized = normalize.normalized(red_list)

    plt.figure()
    # plt.plot(blue_normalized, color='blue')
    plt.plot(green_normalized, color='green')
    # plt.plot(red_normalized, color='red')
    plt.show()
    '''

    # 这是使用小波分解 需要欧拉放大 为什么不用欧拉放大效果更好？
    newPPG = wave.smallwave(green_normalized)
    plt.figure()
    plt.title('newPPG')
    # plt.plot(blue_normalized, color='blue')
    plt.plot(newPPG, color='green')
    # plt.plot(red_normalized, color='red')
    plt.show()
    ffttransomer(newPPG)
    
    '''
    









    
    # ceemd分解 IImfs总共六个元素 每个元素都是一个数组300
    data,IImfs1 = dec.ceemdan_decompose_res(blue_normalized)
    data, IImfs2 = dec.ceemdan_decompose_res(green_normalized)
    data, IImfs3 = dec.ceemdan_decompose_res(red_normalized)
    
    
    # frequency1 = mainfreq.maxfreq(IImfs[0],len(IImfs[0]))
    # frequency2 = mainfreq.maxfreq(IImfs[1], len(IImfs[1]))
    # frequency3 = mainfreq.maxfreq(IImfs[2], len(IImfs[2]))
    # frequency4 = mainfreq.maxfreq(IImfs[3], len(IImfs[3]))
    # frequency5 = mainfreq.maxfreq(IImfs[4], len(IImfs[4]))
    # frequency6 = mainfreq.maxfreq(IImfs[5], len(IImfs[5]))
    # print('mainfreq1',frequency1,'freq2',frequency2,'freq3',frequency3,'freq4',frequency4,'freq5',frequency5)
    # 绿色freq1 9.733333333333333 freq2 3.1333333333333333 freq3 1.2666666666666666 freq4 0.4666666666666667 freq5 0.13333333333333333 freq6 0.0
    # 红色freq1 6.733333333333333 freq2 2.1333333333333333 freq3 1.0 freq4 0.13333333333333333 freq5 -0.06666666666666667
    # 蓝色freq1 6.666666666666667 freq2 -3.3333333333333335 freq3 1.2666666666666666 freq4 0.13333333333333333 freq5 0.0
    # 选择2，3，4作为重构信号
    
    
    
    
    restruct_b = IImfs1[1]+IImfs1[2]
    restruct_g = IImfs2[1]+IImfs2[2]
    restruct_r = IImfs3[1]+IImfs3[2]

    plt.figure()
    plt.title('restruct')
    plt.plot(restruct_b,color = 'blue')
    plt.plot(restruct_g,color = 'green')
    plt.plot(restruct_r,color = 'red')
    plt.show()

    s = np.array([restruct_b, restruct_g, restruct_r])  # 3行n列的矩阵

    fast_ica = FastICA(n_components=3)
    Sr = fast_ica.fit_transform(s)
    S = np.dot(Sr.T, s)


    ffttransomer(S[0, :])
    ffttransomer(S[1, :])
    ffttransomer(S[2, :])
    



