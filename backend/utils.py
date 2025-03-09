import random
from skimage import morphology
import matplotlib.pyplot as plt
import math
import cv2 as cv
import base64
import numpy as np
from PIL import Image


#####################################################################
# Jitendra-Sampling相关
#####################################################################

################################################
# 快排
################################################
def partition(li, left, right):
    tmp = li[left]
    while left < right:
        while left < right and li[right] >= tmp:  # 从右边找比tmp小的数
            right -= 1  # 继续从右往左查找
        li[left] = li[right]  # 把右边的值写到左边空位上

        while left < right and li[left] <= tmp:
            left += 1
        li[right] = li[left]  # 把左边的值写到右边空位上

    li[left] = tmp  # 把tmp归位
    return left


def quick_sort(li, left, right):
    if left < right:  # 至少两个元素
        mid = partition(li, left, right)
        quick_sort(li, left, mid - 1)
        quick_sort(li, mid + 1, right)
        return True
    else:
        return False


################################################
# Jitendra’s Sampling采样(cv2)
################################################
def Jitendra(points, final_num, thresh=3):
    # print('计时开始：')
    # begTime = time.perf_counter()

    # 若轮廓点数量过大，则只取thresh*final_num以减小复杂度
    if len(points) > thresh*final_num:
        random.shuffle(points)
        points = points[:thresh*final_num]

    length = len(points)
    distance = []
    for i in range(length):
        for j in range(length):
            if i !=j:
                dist = math.sqrt(pow(points[i][0] - points[j][0], 2) + pow(points[i][1] - points[j][1], 2))
                distance.append({'index1' : i, 'index2' : j, 'dist' : dist})

    # node1Time = time.perf_counter()
    #　print('计时结点1-完成距离计算：' + str(node1Time - begTime) + '秒')

    distance.sort(key=lambda x: x['dist'])   # 用封装好的函数根据距离进行排序
    # print(re)

    # node2Time = time.perf_counter()
    # print('计时结点2-完成快排：' + str(node2Time - begTime) + '秒')

    need_remove_num = length - final_num
    removed = np.zeros((length), dtype = np.int8)
    i = 0
    while need_remove_num > 0:
        if removed[distance[i]['index1']] == 0 and removed[distance[i]['index2']] == 0:
            removed[distance[i]['index1']] = 1
            need_remove_num -= 1
        i += 1

    re = []
    for i in range(length):
        if removed[i] == 0:
            re.append(points[i])

    # node3Time = time.perf_counter()
    # print('计时结点3-完成挑选：' + str(node3Time - begTime) + '秒')

    # endTime = time.perf_counter()
    # print('计时结束：' + str(endTime - begTime) + '秒')
    return re


################################################
# 求骨架散点
################################################
def get_ske_sample_points(image, num, show=False):
    tmp = 1 - image.astype(np.uint8) / 255  # skeletonize无法直接识别cv库图片，需转换
    ske = morphology.skeletonize(tmp)  # 细化
    points = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if ske[x][y] == 1:  # 获得的ske图像中白点(1)是前景点
                points.append([x, y])
    re = Jitendra(points, num)  # 提取轮廓点

    if show:
        # 简单验证一下处理结果
        finalimg = np.zeros((256, 256, 1), np.uint8)
        for item in re:
            finalimg[item[0]][item[1]] = 255

        plt.set_cmap('binary')
        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.rcParams['axes.unicode_minus'] = False
        plt.imshow(finalimg)
        plt.title('skeleton_points_based_on_Jitendra_Sampling', fontsize=16)
        plt.show()

    return re

################################################
# 提取轮廓
################################################
def get_contour(img):
    # get contour
    contour_img = np.zeros(shape=img.shape, dtype=np.uint8)
    contour_list = []
    contour_img += 255
    h = img.shape[0]
    w = img.shape[1]
    for i in range(1, h-1):
        for j in range(1, w-1):
            if img[i][j] == 0:
                contour_img[i][j] = 0
                sum = 0
                sum += img[i - 1][j + 1]
                sum += img[i][j + 1]
                sum += img[i + 1][j + 1]
                sum += img[i - 1][j]
                sum += img[i + 1][j]
                sum += img[i - 1][j - 1]
                sum += img[i][j - 1]
                sum += img[i + 1][j - 1]
                if sum == 0:
                    contour_img[i][j] = 255
                else:
                    contour_list.append([i, j])
    return contour_img, contour_list


################################################
# 求字形轮廓点
################################################
def get_contour_sample_points(image, num, show=False):
    _, points = get_contour(image)
    re = Jitendra(points, num)  # 提取轮廓点

    if show:
        # 简单验证一下处理结果
        finalimg = np.zeros((256, 256), np.uint8)
        for item in re:
            finalimg[item[0]][item[1]] = 255

        plt.set_cmap('binary')
        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.rcParams['axes.unicode_minus'] = False
        plt.imshow(finalimg)
        plt.title('Contour_points_based_on_Jitendra_Sampling', fontsize=16)
        plt.show()

    return re


################################################
# 求前景背景点
################################################
def fg_points_percent(img, centerx, centery, r):
    fg_cnt = 0  # 前景点个数
    total_cnt = 0     # 圆内点个数
    for i in range(centerx-r, centerx+r+1):
        for j in range(centery-r,centery+r+1):
            if i < img.shape[0] and j < img.shape[1]:
                if pow(i-centerx, 2) + pow(j-centery, 2) < pow(r, 2):
                    total_cnt += 1
                    if img[i, j]==0:
                        fg_cnt += 1
    return fg_cnt/total_cnt


################################################
# 求点集中心（cv2）
################################################
def G_points(img,points):
    xfenzi = 0
    yfenzi = 0
    fenmu = 0
    for [x,y] in points:
        fenmu += 1
        xfenzi += x
        yfenzi += y
    fx = xfenzi / fenmu
    fy = yfenzi / fenmu
    print('重心：[' + str(fx) + ',' + str(fy) + ']')
    return fx, fy

def ostu(img0, thresh1=0):
    gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)  # 灰度化
    box = cv.boxFilter(gray, -1, (5, 5), normalize=True)  # 去噪
    _, binarized = cv.threshold(box, thresh1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值化
    return binarized

################################################
# base64编码转为编码头、图片格式、cv2库格式图片
################################################
def base64_to_cv2(base64_data, save=False):
    parts = base64_data.split(',')
    base64_head = parts[0]
    base64_body = parts[1]
    img_type = parts[0].split(':')[1].split(';')[0].rsplit('/', 1)[1]  # 图片类型，如"png"=

    # python binascii.Error: Incorrect padding
    # 以下对base64进行补全
    missing_padding = 4 - len(base64_body) % 4
    if missing_padding:
        base64_body += '=' * missing_padding
    img_data = base64.b64decode(base64_body)

    nparr = np.frombuffer(img_data, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
    if save:
        cv.imwrite('resize_image.' + img_type, img)  # 保存输入的图片
    return base64_head, img_type ,img

################################################
# cv2转PIL
################################################
def cv2_to_PIL(img):
    img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    return img


################################################
# PIL转cv2
################################################
def PIL_to_cv2(img):
    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    return img
