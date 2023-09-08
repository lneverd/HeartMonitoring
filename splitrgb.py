import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import signalprocess as process
from scipy import signal
from scipy.fftpack import fft
import jadeR as jade
from sklearn.decomposition import FastICA
from caculateheartrate.related import pearson
import hanningfilter as hann



def person_related(S,green_normalized):
    rate_original = 0
    if abs(pearson(S[0, :], green_normalized)) > abs(pearson(S[1, :], green_normalized)):
        max_num = abs(pearson(S[0, :], green_normalized))
        if max_num > abs(pearson(S[2, :], green_normalized)):
            rate_original = S[0, :]
        else:
            rate_original = S[2, :]
    else:
        max_num = abs(pearson(S[1, :], green_normalized))
        if max_num > abs(pearson(S[2, :], green_normalized)):
            rate_original = S[1, :]
        else:
            rate_original = S[2, :]
    return rate_original

#默认视频帧数是30fps 数组长度是200
def buildtime(frmecount,fps):
    t=[(i)/fps for i in range(0,frmecount)] #time in seconds
    return t
# 去趋势化
def detrendRGB(originalsginal):
    new_sginal = signal.detrend(originalsginal)
    return new_sginal


def pixel_opt(img):
    # b,g,r = cv.split(img)
    #
    #
    # b_mean,b_dev = cv.meanStdDev(b)
    # g_mean,g_dev = cv.meanStdDev(g)
    # r_mean, r_dev = cv.meanStdDev(r)

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
    return pixelblue, pixelgreen, pixelred

def normalize(list):
    mean = np.mean(list)
    std = np.std(list)
    list = np.array(list)
    normal_list = np.zeros(len(list), dtype=np.float32)#生成一个全是0长度为list长度大小的数组用来存放归一化之后的结果
    cv.normalize(list, normal_list, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return normal_list


# 快速傅里叶变换
def ffttransform(filtedData,frame_count):
    fps = 20
    fft = np.fft.fft(filtedData)

    # 计算心率（频域中幅度最大的频率）
    freqs = np.fft.fftfreq(frame_count, d=1 / fps)
    max_freq = freqs[np.argmax(np.abs(fft))]
    heart_rate = abs(max_freq * 60)

    print(heart_rate)
# 选出分离的信号 jade使用
def selectband(data):

    # 采样率
    fs = 20
    # 计算功率谱密度
    frequencies, power_spectrum = signal.welch(data, fs)

    # 绘制功率谱
    plt.plot(frequencies, power_spectrum)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度')
    plt.title('功率谱')
    plt.grid(True)
    plt.show()

    # 计算峰值
    peak_freq = frequencies[np.argmax(power_spectrum)]

    return peak_freq

def comparefreq(A,B,C):
    if selectband(A)>selectband(B):
        if selectband(A)>selectband(C):
            return A
        else:
            return C
    else:
        if selectband(B)>selectband(C):
            return B
        else:
            return C


# 移动滤波 (平滑处理)
def moving_average(data, smooth_interval=2):
    if (smooth_interval > len(data)):
        print("Smooth interval > lenght of data")
        return

    sum = 0
    new_data = np.zeros(len(data))

    for i in range(smooth_interval):
        for j in range(smooth_interval):
            sum += data[i + j]
        average = sum / smooth_interval
        new_data[i] = average
        sum = 0

    for i in range(len(data) - smooth_interval):
        for j in range(smooth_interval):
            sum += data[i + j]
        average = sum / smooth_interval
        new_data[i + smooth_interval] = average
        sum = 0
    return new_data

# 通过ptt计算的心率
def heartrate(peaks):
    interal = 0
    for i in range(len(peaks) - 1):
        interal += peaks[i + 1] - peaks[i]

    interal = interal / (len(peaks) - 1)
    t = interal / 20
    hr = 1 / t * 60
    return round(hr)
def blood_pressure(peaklist):
    # 求出ptt的平均值
    ptt_mean = np.mean(peaklist) / 20 * 1000
    # 计算收缩压公式
    blood_sbp = (-0.686) * ptt_mean + 156.6
    print('收缩血压是:', blood_sbp)

# 接收三个分离的量roi1
blue_list = list()
green_list = list()
red_list = list()
# 第二个list用来接收roi2
blue_list2 = list()
green_list2 = list()
red_list2 = list()
t=buildtime(300,20)

for i in range(300):
    img = cv.imread("./data/zxy/pyramid.ROI1."+str(i+1)+".jpg")
    blue_mean, green_mean, red_mean = pixel_opt(img)
    blue_list.append(blue_mean)
    green_list.append(green_mean)
    red_list.append(red_mean)


for i in range(300):
    img2 = cv.imread("./data/zxy/pyramid.ROI2."+str(i+1)+".jpg")
    blue_mean2, green_mean2, red_mean2 = pixel_opt(img2)
    blue_list2.append(blue_mean2)
    green_list2.append(green_mean2)
    red_list2.append(red_mean2)
# roi1的图像
plt.figure()
plt.title("ROI1-original-sginal")
plt.plot(t,blue_list,color='blue')
plt.plot(t,green_list,color = 'green')
plt.plot(t,red_list,color='red')

plt.show()
# roi2的图像
plt.figure()
plt.title("ROI2-original-sginal")
plt.plot(t,blue_list2,color='blue')
plt.plot(t,green_list2,color = 'green')
plt.plot(t,red_list2,color='red')

plt.show()

# 去趋势化
r_detrended = detrendRGB(red_list)
g_detrended = detrendRGB(green_list)
b_detrended = detrendRGB(blue_list)

r_detrended2 = detrendRGB(red_list2)
g_detrended2 = detrendRGB(green_list2)
b_detrended2 = detrendRGB(blue_list2)

# 去趋势化的图像
# plt.figure()
# plt.title("detrend-sginal-1")
# plt.plot(t,b_detrended,color='blue')
# plt.plot(t,g_detrended,color = 'green')
# plt.plot(t,r_detrended,color='red')
# plt.show()
#
# plt.figure()
# plt.title("detrend-sginal-2")
# plt.plot(t,b_detrended2,color='blue')
# plt.plot(t,g_detrended2,color = 'green')
# plt.plot(t,r_detrended2,color='red')
# plt.show()


# 归一化之后的结果
normal_blue_list = normalize(b_detrended)
normal_green_list = normalize(g_detrended)
normal_red_list = normalize(r_detrended)


normal_blue_list2 = normalize(b_detrended2)
normal_green_list2 = normalize(g_detrended2)
normal_red_list2 = normalize(r_detrended2)



# 准备ICA
mix = np.array([normal_red_list,normal_green_list,normal_blue_list])
mix2 = np.array([normal_red_list2,normal_green_list2,normal_blue_list2])


'''
# jade 分离矩阵
ICA = jade.jadeR(mix)
S = np.dot(ICA,mix)


ICA2 = jade.jadeR(mix2)
S2 = np.dot(ICA2,mix2)

'''



# ran = 2 * np.random.random([3, 3])
# mix = ran.dot(mix)
# mix2 = ran.dot(mix2)

# fastica
fast_ica = FastICA(n_components=3)

mix = mix.T
Sr = fast_ica.fit_transform(mix)

S = Sr.T
mix2 = mix2.T

Sr2 = fast_ica.fit_transform(mix2)
S2 = Sr2.T

signal1 = comparefreq(np.ravel(S[0, :]),np.ravel(S[1, :]),np.ravel(S[2, :]))
signal2 = comparefreq(np.ravel(S2[0, :]),np.ravel(S2[1, :]),np.ravel(S2[2, :]))

# signal1 = person_related(S,normal_green_list)
# signal2 = person_related(S2,normal_green_list2)


result1 = process.fliter(signal1)
result2 = process.fliter(signal2)

# result1 = hann.filter(signal1)
# result2 = hann.filter(signal2)




'''

# 滤波器 这个是先滤波后选信号
array_dest_b = process.fliter(S[0, :].T)
array_dest_g = process.fliter(S[1, :].T)
array_dest_r = process.fliter(S[2, :].T)

array_dest_b2 = process.fliter(S2[0, :].T)
array_dest_g2 = process.fliter(S2[1, :].T)
array_dest_r2 = process.fliter(S2[2, :].T)
plt.figure()

y1 = plt.subplot(411)
y2 = plt.subplot(412)
y3 = plt.subplot(413)
y1.plot(array_dest_b)
y2.plot(array_dest_g)
y3.plot(array_dest_r)
plt.show()





#jade
result1 = comparefreq(np.squeeze(array_dest_b),np.squeeze(array_dest_g),np.squeeze(array_dest_r))
result2 = comparefreq(np.squeeze(array_dest_b2),np.squeeze(array_dest_g2),np.squeeze(array_dest_r2))
'''

# 通过论文[16]得知是选择size为40
window_size = 40

averagesginal = moving_average(result1,window_size)
averagesginal2 = moving_average(result2,window_size)

plt.figure()
plt.title("Smoothed sginal")
plt.plot(averagesginal,color = 'blue')
plt.plot(averagesginal2,color = 'red')
plt.show()


# findpeaks
peaks,properties = signal.find_peaks(averagesginal,distance=6) #蓝线
print("peaks",peaks)
peaks2,properties2 = signal.find_peaks(averagesginal2,distance=6) #红线
print("peaks2",peaks2)


# 人的收缩压最低不能低于50mmHg
peaklist = list()
for i in peaks:
    for j in peaks2:
        if abs(i-j)<5:
            peaklist.append(abs(i-j))
            break
print('pttlist:',peaklist)

blood_pressure(peaklist)

#TODO 计算心率
print(heartrate(peaks))
print(heartrate(peaks2))
print('心率是：',int((heartrate(peaks)+heartrate(peaks2))/2))


# (300,) to  (300, 1)  np.squeeze(array_dest_g)




cv.waitKey(0)
cv.destroyAllWindows()