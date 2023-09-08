import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack

def sameSize(imgA: np.ndarray, imgB: np.ndarray):
    """
    两张尺寸不同的图片
    :param imgA: 图片 A
    :param imgB: 图片 B
    :return: 统一尺寸的两张图像: tuple
    """
    row1, col1 = imgA.shape[0:2]
    row2, col2 = imgB.shape[0:2]
    row = min(row1, row2)
    col = min(col1, col2)
    imgA = imgA[0:row, 0:col]
    imgB = imgB[0:row, 0:col]
    return imgA, imgB

#convert RBG to YIQ


#Build Gaussian Pyramid
def build_gaussian_pyramid(src,level=3):
    s=src.copy()
    pyramid=[s]
    for i in range(level):
        s=cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid

#Build Laplacian Pyramid
def build_laplacian_pyramid(src,levels=3):
    gaussianPyramid = build_gaussian_pyramid(src, levels)
    pyramid=[]
    for i in range(levels,0,-1):
        GE=cv2.pyrUp(gaussianPyramid[i])
        gaussianPyramid[i-1],GE = sameSize(gaussianPyramid[i-1],GE)
        L=cv2.subtract(gaussianPyramid[i-1],GE)
        pyramid.append(L)
    return pyramid

#把截取的ROI区域的图片加载进来
def load_video(filename,videofps,roi):

    # 需要修改的地方：图片的height * width 图片路径 滤波高频和低频 自行决定 保存路径的名称
    frame_count = 300
    width = 90 #宽
    height = 90 #高
    fps = int(videofps)
    video_tensor=np.zeros((frame_count,height,width,3),dtype='float')
    # cv2.INTER_AREA适合缩小图片，INTER_CUBIC适合放大图片
    for i in range(300):
        img = cv2.imread(f"./data/{filename}/{roi}." + str(i + 1) + ".jpg")
        try:
            newimg = cv2.resize(img,(width,height))
        except:
            continue
        video_tensor[i] = newimg
    return video_tensor,fps

# apply temporal ideal bandpass filter to gaussian video
def temporal_ideal_filter(tensor,low,high,fps,axis=0):
    fft=fftpack.fft(tensor,axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    # 是否取绝对值?有待考证 如果改成np.real(iff)那么速率就与原视频一样了
    return np.abs(iff)

# build gaussian pyramid for video
def gaussian_video(video_tensor,levels=3):
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_gaussian_pyramid(frame,level=levels)
        gaussian_frame=pyr[-1]
        if i==0:
            vid_data=np.zeros((video_tensor.shape[0],gaussian_frame.shape[0],gaussian_frame.shape[1],3))
        vid_data[i]=gaussian_frame
    return vid_data

#amplify the video 放大倍数
def amplify_video(gaussian_vid,amplification=20):
    return gaussian_vid*amplification

#reconstract video from original video and gaussian video
def reconstract_video(amp_video,origin_video,levels=3):
    final_video=np.zeros(origin_video.shape)
    for i in range(0,amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img=cv2.pyrUp(img)
        img,origin_video[i] = sameSize(img,origin_video[i])
        img=img+origin_video[i]
        final_video[i]=img
    return final_video

#保存路径名称
def save_video(video_tensor,roi):
    for i in range(300):
        img = video_tensor[i]
        img = cv2.convertScaleAbs(img)
        cv2.imwrite("./data/zxh/pyramid."+str(roi)+"."+str(i+1)+".jpg",img)


#magnify color 放大颜色 放大倍数 初始是20倍 50倍试试
def magnify_color(roi,video_fps,low,high,levels=3,amplification=50):
    t,f=load_video(video_fps,roi) #返回 图片列表和fps
    gau_video=gaussian_video(t,levels=levels)
    filtered_tensor=temporal_ideal_filter(gau_video,low,high,f)
    amplified_video=amplify_video(filtered_tensor,amplification=amplification)
    final=reconstract_video(amplified_video,t,levels=3)
    save_video(final,roi)

# build laplacian pyramid for video
def laplacian_video(video_tensor,levels=3):
    tensor_list=[]
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_laplacian_pyramid(frame,levels=levels)
        if i==0:
            for k in range(levels):
                tensor_list.append(np.zeros((video_tensor.shape[0],pyr[k].shape[0],pyr[k].shape[1],3)))
        for n in range(levels):
            tensor_list[n][i] = pyr[n]
    return tensor_list

#butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y

#reconstract video from laplacian pyramid
def reconstract_from_tensorlist(filter_tensor_list,levels=3):
    final=np.zeros(filter_tensor_list[-1].shape)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(levels-1):
            up=cv2.pyrUp(up)#可以改为up=cv2.pyrUp(up)
        final[i],up = sameSize(final[i],up)
        final[i]=up
    return final


# manify motion roi,video_fps,low,high,levels=3,amplification=50
def magnify_motion(filename,roi,video_fps,low,high,levels=3,amplification=50):
    t,f=load_video(filename,video_fps,roi)
    lap_video_list=laplacian_video(t,levels=levels)
    filter_tensor_list=[]
    for i in range(levels):
        filter_tensor=butter_bandpass_filter(lap_video_list[i],low,high,f)
        filter_tensor*=amplification
        filter_tensor_list.append(filter_tensor)
    recon=reconstract_from_tensorlist(filter_tensor_list)
    final=t+recon
    save_video(final,roi)

if __name__=="__main__":
    fps = 20
    freq_min = 0.83  #1  0.75hz 来自文献 bennett2016  0.83~1hz 放大率100 mit论文 有人说图像金字塔选择6层效果最好
    freq_max = 1.6   # 1.8  1 hz
    filename = 'zxhROI'
    # magnify_color('ROI1',fps,freq_min,freq_max)
    magnify_motion(filename,'ROI2',fps,0.5,4)
