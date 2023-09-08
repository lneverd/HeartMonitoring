from scipy.signal import lfilter,butter,iirfilter,filtfilt

import numpy as np


"""
各个频率值以截止频率作归一化后，频率都是截止频率的相对值，没有了量纲。
信号处理工具箱中经常使用的是nyquist频率，它被定义为采样频率的二分之一，
在滤波器的阶数选择和设计中的截止频率均使用nyquist频率进行归一化处理。
例如对于一个采样频率为1000hz的系统，400hz的归一化频率就为400/500=0.8，
归一化频率范围在[0,1]之间。如果将归一化频率转换为角频率，则将归一化频率乘以2*pi，
如果将归一化频率转换为hz，则将归一化频率乘以采样频率的一半
本论文中视频的采样频率是30fps
"""
def fliter(orisginal):
    fs = 20
    lowcut = 0.83
    highcut = 2.0
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b,a = butter(6,[low,high],"bandpass")
    # b, a = iirfilter(14, [2 * np.pi * 0.5, 2 * np.pi * 4], rs=50, btype='band', analog=True, ftype='cheby2')
    ressignal = filtfilt(b, a, orisginal,axis=0)


    return ressignal
