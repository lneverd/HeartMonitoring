from scipy.fft import fft, fftfreq
import numpy as np

def maxfreq(filtedData,frame_count):
    fps = 20
    fft = np.fft.fft(filtedData)

    # 计算心率（频域中幅度最大的频率）
    freqs = np.fft.fftfreq(frame_count, d=1 / fps)
    max_freq = freqs[np.argmax(np.abs(fft))]

    return max_freq

