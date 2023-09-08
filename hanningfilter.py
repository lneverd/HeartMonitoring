import numpy as np
from scipy import signal

def filter(input_signal):
    # 定义滤波器参数
    order = 128  # 滤波器阶数
    fs = 20  # 采样率
    nyquist = 0.5 * fs  # 奈奎斯特频率
    low_cutoff = 0.83 / nyquist  # 低截止频率
    high_cutoff = 2.0 / nyquist  # 高截止频率

    # 创建汉宁窗口
    window = np.hamming(order)

    # 创建带通滤波器
    b = signal.firwin(order, [low_cutoff, high_cutoff], window='hamming', pass_zero=False)

    # 打印滤波器系数
    # print("滤波器系数：", b)
    # 应用滤波器
    filtered_signal = signal.lfilter(b, 1, input_signal)

    return filtered_signal
