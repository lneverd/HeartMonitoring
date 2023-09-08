import cv2 as cv
import numpy as np
import maxArea

def draw(x,y,width,height):
    # 左上点
    x1 = int(x-0.5*width)
    y1 = int(y-0.5*height)
    # 右上点
    x0 = x + 0.5 * width
    y0 = y1
    # 左下点
    x2 = x1
    y2 = y + 0.5 * height
    # 右下点
    x3 = int(x0)
    y3 = int(y2)
    return x1,y1,x3,y3
    # return img[y1:y3,x1:x3]

# 转化成二值图像 0或者255
def threshold_demo(image):
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY) #把输入图像灰度化
    # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    return binary


#     计算二值图像白色像素点个数表示面积
def getarea(image):
    sum=0
    h,w=image.shape[0],image.shape[1]
    for i in range(h):
        for j in range(w):
                if image[i][j].all()>0:
                    sum+=1
    return sum

def getframe(img,imgwidth,imgheight,lower,upper):
    hsvImage = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    skinMask = cv.inRange(hsvImage, lower, upper)
    kernal = np.ones((5, 5), np.uint8)  # 建立一个核心
    opening = cv.morphologyEx(skinMask, cv.MORPH_OPEN, kernal)  # 开启
    dilation = cv.dilate(opening, kernal, iterations=1)  # 膨胀
    image, contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # 取轮廓,这个dilation是个二值图像

    cnt = contours[0]  # 轮廓集合中的第二个轮廓

    M = cv.moments(cnt)  # 图像矩
    # 重心 基本上没问题
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print('cx:', cx, 'cy:', cy)
    # y1:y2 x1:x2
    # 只显示皮肤其余地方都是黑色
    newimg = cv.bitwise_and(img, img, mask=skinMask)
    # cv.imshow('forhead',img)
    # cv.imshow('skin',newimg)
    # x1,y1,x2,y2 = draw(cx,cy,10,5)
    binary = threshold_demo(newimg)
    # cv.imshow('binary',binary)
    # area = getarea(binary[y1:y2,x1:x2])
    # print('binary:',area)
    width = imgwidth
    height = imgheight
    flag = False
    for i in range(imgwidth, 0, -10):

        for j in range(imgheight, 0, -10):
            # i表示宽度，j表示高度
            x1, y1, x2, y2 = draw(cx, cy, i, j)
            area = getarea(binary[y1:y2, x1:x2])
            if area >= i * j:
                flag = True
                width, height = i, j
                print('----------------------->', i, j)
                break
        if flag is True:
            break

    x1, y1, x2, y2 = draw(cx, cy, width, height)
    return x1,y1,x2,y2



# path = './data/lgrROI/ROI1.1.jpg'
# img = cv.imread(path)
# imgheight = img.shape[0]
# imgwidth = img.shape[1]


# 定义皮肤的范围，具体自己调整
lower = np.array([0, 50, 100], dtype="uint8")
upper = np.array([25, 255, 255], dtype="uint8")


for n in range(300):
    img = cv.imread("./data/zxhROI/ROI2."+str(n+1)+".jpg")
    new_img = maxArea.execute(img)
    cv.imwrite("./data/zxhROI/ROI2."+str(n+1)+".jpg",new_img)














# cv.imshow('final',img[y1:y2,x1:x2])

# skin = cv.bitwise_and(img, img, mask=skinMask)
# cv.imshow("images", skin)
# cv.imshow("imgage2",image)





cv.destroyAllWindows()