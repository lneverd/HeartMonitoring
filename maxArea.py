from datetime import datetime
import cv2
import numpy as np


def maximalRectangle(matrix, mask=255):
    """二值矩阵获取最大矩形"""
    n, m = matrix.shape
    left = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            if matrix[i, j] == mask:
                if j == 0:
                    left[i, j] = 1
                else:
                    left[i, j] = left[i, j - 1] + 1
    ret = 0
    ans_x = 0
    ans_y = 0
    ans_w = 0
    ans_h = 0
    for i in range(m):
        tmp_ret, y, w, h = largestRectangleArea(left[:, i])
        if tmp_ret > ret:
            ret = tmp_ret
            ans_x = i - w + 1
            ans_y = y
            ans_w = w
            ans_h = h
    return ans_x, ans_y, ans_w, ans_h


def largestRectangleArea(heights):
    """直方图获取最大矩形"""
    n = len(heights)
    left, right = [0] * n, [n] * n

    mono_stack = list()
    for i in range(n):
        while mono_stack and heights[mono_stack[-1]] >= heights[i]:
            right[mono_stack[-1]] = i
            mono_stack.pop()
        left[i] = mono_stack[-1] if mono_stack else -1
        mono_stack.append(i)
    ans = 0
    y = 0
    w = 0
    h = 0
    for i in range(n):
        tmp = (right[i] - left[i] - 1) * heights[i]
        if tmp > ans:
            ans = tmp
            y = left[i] + 1
            w = int(heights[i])
            h = right[i] - left[i] - 1
    return ans, y, w, h


def get_binary_pic(img, threshold=0, mask=255):
    """获取二值图像"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, mask, cv2.THRESH_BINARY)
    return thresh


def cut_picture(img, threshold=0):
    """返回最大有效内接矩形的图片"""
    print("cut_picture start...", datetime.now())
    ans_x, ans_y, ans_w, ans_h = maximalRectangle(get_binary_pic(img,threshold))
    print("cut_picture end...", datetime.now())
    return img[ans_y:ans_y + ans_h, ans_x:ans_x + ans_w, :]


def execute(img):
    lower = np.array([0, 50, 100], dtype="uint8")
    upper = np.array([25, 255, 255], dtype="uint8")
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(hsvImage, lower, upper)
    skin = cv2.bitwise_and(img, img, mask=skinMask)
    new_img = cut_picture(skin)
    return new_img


if __name__ == '__main__':

    lower = np.array([0, 50, 100], dtype="uint8")
    upper = np.array([25, 255, 255], dtype="uint8")

    img = cv2.imread('./data/zxyROI/ROI1.1.jpg')
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(hsvImage, lower, upper)

    skin = cv2.bitwise_and(img,img,mask=skinMask)
    new_img = cut_picture(skin)
    cv2.imshow('skin',skin)
    cv2.imshow('cut',new_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()