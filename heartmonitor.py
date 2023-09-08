#导入cv2包
import cv2
import os
import dlib
import numpy as np
from matplotlib import pyplot as plt
#调用包
predictor_path = "./site-package/shape_predictor_81_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# 输入的文件路径
filepath = "./data/0612_2_zxh_79.mp4"
video = cv2.VideoCapture(filepath)
# face_detector = cv2.CascadeClassifier(r'./data/haarcascade_frontalface_default.xml')
# 给每一个测试对象都创建一个文件夹
face_id = input("输入测试的名称:")
dir_path = "./data/"+str(face_id)+"ROI"
if os.path.exists(dir_path):
    print("你要创建的文件夹已经存在")
else:
    os.mkdir(dir_path)



def getroi(frame):


    a = np.arange(81)
    b = np.arange(81)
    # 取灰度
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    dets = detector(img_gray, 0)
    if (len(dets) != 0):
        # 找到脸颊区域
        for i in range(len(dets)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dets[i]).parts()])
            for idx, point in enumerate(landmarks):
                # 68点的坐标
                pos = (point[0, 0], point[0, 1])
                # ROI：脸颊区域
                a = a.tolist()
                b = b.tolist()
                a[idx] = point[0, 0]
                b[idx] = point[0, 1]
                a = np.array(a)
                b = np.array(b)

        # 额头区域
        x1 = int(a[76])
        y1 = int(b[19])
        x2 = int(a[73])
        y2 = int(b[73])
        if (x1 < 0):  # 防止人脸转动，x1位置超出图像
            x1 = 0
        forehead_ROI = frame[y2:y1, x1:x2]

        # 脸颊区域
        x1 = int((a[2] + a[4]) / 2)
        y1 = int((b[2] + b[4]) / 2)
        y2 = int((b[2] + b[23]) / 2)
        x2 = int((a[32] + a[40]) /2)
        if (x1 < 0):
            x1 = 0
        cheek_ROI = frame[y2:y1, x1:x2]

        # 检测整个人脸区域
        for k, d in enumerate(dets):
            [x1, x2, y1, y2] = [d.left(), d.right(), d.top(), d.bottom()]
        face_ROI = frame[y1:y2, x1:x2]
        return forehead_ROI,cheek_ROI


def readvideo():
    count = 1
    interval = 1
    while (video.isOpened()):

        ret, frame = video.read()
        # if interval > 430:
        # 取灰度
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 人脸数rects
        rects = detector(img_gray, 0)
        # for i in range(len(rects)):
        #     landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rects[i]).parts()])
        #     for idx, point in enumerate(landmarks):
        #         # 81点的坐标
        #         pos = (point[0, 0], point[0, 1])
        #         # 利用cv2.circle给每个特征点画一个圈，共81个
        #         cv2.circle(frame, pos, 2, color=(0, 255, 0))
        #         # 利用cv2.putText输出1-81
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         cv2.putText(frame, str(idx + 1), pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        #     cv2.imshow("video",frame)

        try:
            forehead_ROI, cheek_ROI = getroi(frame)
            cv2.imwrite(f"./data/{face_id}ROI/" +"ROI1"+ "." + str(count) + ".jpg",forehead_ROI)
            cv2.imwrite(f"./data/{face_id}ROI/" +"ROI2"+ "." + str(count) + ".jpg",cheek_ROI)
            count+=1
        except:
            continue
        # key = cv2.waitKey(33)
        # if key == 27:
        #     break
        if count>300: #需要截图的数目
            break
        print(interval)
        interval+=1

readvideo()



video.release()


cv2.destroyAllWindows()
