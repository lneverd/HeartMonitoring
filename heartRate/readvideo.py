import cv2 as cv
import pyramids
import numpy as np
import eulerian
import heartrate

video_frames = list()
amplified_video_pyramid = []

frame_count = 300
fps = 30
freq_min = 1
freq_max = 1.8

# 为什么必须设置高宽一样 否则就报错
imgwidth = 200
imgheight = 200



for i in range(300):
    img = cv.imread("../data/zxyROI-0/ROI1."+str(i+1)+".jpg")

    img = cv.resize(img,(imgwidth,imgheight))

    frame = np.ndarray(shape=img.shape, dtype="float")

    frame[:] = img * (1. / 255)


    video_frames.append(frame)

lap_video = pyramids.build_video_pyramid(video_frames)


for i,video in enumerate(lap_video):
    # if i == 0 or i == len(lap_video)-1:
    #     continue
    # video是每一帧

    # Eulerian magnification with temporal FFT filtering fft_filter里面自带放大
    print("Running FFT and Eulerian magnification...")
    result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
    # 放大之后的图像与拉普拉斯图像合成
    lap_video[i] += result

    # Calculate heart rate
    print("Calculating heart rate...")
    heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)


print(heart_rate)

cv.waitKey(0)
cv.destroyAllWindows()
