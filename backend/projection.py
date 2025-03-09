from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math


# PIL库输入输出
# 获得旋转图片
def get_rotate(pilim, angle):
    # 转换为有alpha层
    im2 = pilim.convert('RGBA')
    # 旋转
    rot = im2.rotate(angle, expand=True)  # 注意此时会改变图像大小
    rot = rot.resize((256, math.floor(rot.height * 256 / rot.width)))  # 等比例压缩将宽度变成256
    # 创建一个与旋转图像大小相同的白色图像
    fff = Image.new('RGBA', rot.size, (255,) * 4)
    # 使用alpha层的rot作为掩码创建一个复合图像
    re = Image.composite(rot, fff, rot)
    return re


# PIL库输入，cv库输出
# 获得垂直投影数组和投影图（宽度为256）
def get_vert_proj(img):
    # 求每列像素个数
    img1 = img.convert('1')
    image = np.array(img1)
    pixels = []
    for i in range(img.width):
        black = 0
        for j in range(img.height):
            if image[j][i] == 0:
                black += 1
        pixels.append(black)
    # print(pixels)
    vert_proj = np.ones((img.height, img.width), dtype=np.uint8)  # 定义与图像规格相同的二维数组
    vert_proj *= 255
    for i in range(img.width):
        for j in range(pixels[i]):
            j = img.height - 1 - j  # 翻转投影图
            vert_proj[j][i] = 0  # 纵向赋0值变黑，营造直方图的意思
    return pixels, vert_proj


# PIL库输入，cv库输出
# 获得水平投影数组和投影图（高度为256）
def get_hori_proj(img):
    # 求每列像素个数
    img1 = img.convert('1')
    image = np.array(img1)
    pixels = []
    for i in range(img.height):
        black = 0
        for j in range(img.width):
            if image[i][j] == 0:
                black += 1
        pixels.append(black)

    vert_proj = np.ones((img.height, img.width), dtype=np.uint8)  # 定义与图像规格相同的二维数组
    vert_proj *= 255
    for i in range(img.height):
        for j in range(pixels[i]):
            vert_proj[i][j] = 0  # 纵向赋0值变黑，营造直方图的意思
    return pixels, vert_proj


# 求f4和f5
def get_f4_f5(ver_pix1,ver_pix2):
    if len(ver_pix1) != len(ver_pix2):
        return -1
    # 以下为f5数据
    aver1 = sum(ver_pix1) / len(ver_pix1)
    aver2 = sum(ver_pix2) / len(ver_pix2)
    fenzi=0
    fenmu1=0
    fenmu2=0
    # 以下为f4数据
    mini=0
    maxi=0
    for i in range(len(ver_pix1)):
        # 以下为f5数据
        fenzi += (ver_pix1[i]-aver1)*(ver_pix2[i]-aver2)
        fenmu1 += pow(ver_pix1[i] - aver1, 2)
        fenmu2 += pow(ver_pix2[i] - aver2, 2)
        # 以下为f4数据
        mini += min(ver_pix1[i], ver_pix2[i])
        maxi += max(ver_pix1[i], ver_pix2[i])

    f5 = fenzi / math.sqrt(fenmu1*fenmu2)
    f4 = mini/maxi
    return f4,f5


# 返回该角度下的f4,f5值
def get_rotation_projection_similarity(img1, img2, angle, show = True, save = False):
    out1 = get_rotate(img1, angle)
    out1 = out1.convert(img1.mode)
    if save:
        out1.save('rotate1.jpg')  # 保存
    vert_pixel1, vert_img1 = get_vert_proj(out1)
    if save:
        cv.imwrite("vertical_projection1.jpg", vert_img1)  # 保存

    out2 = get_rotate(img2, angle)
    out2 = out2.convert(img2.mode)
    if save:
        out2.save('rotate2.jpg')  # 保存
    vert_pixel2, vert_img2 = get_vert_proj(out2)
    if save:
        cv.imwrite("vertical_projection2.jpg", vert_img2)  # 保存

    f4, f5 = get_f4_f5(vert_pixel1, vert_pixel2)

    if show:
        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.rcParams['axes.unicode_minus'] = False
        plt.set_cmap('binary')

        plt.subplot(2, 2, 1)
        plt.imshow(out1)
        plt.title('重叠比例f4 = {:.5f}'.format(f4), fontsize=14)
        # plt.title('f4 = {}'.format(f4), y=-0.2)

        plt.subplot(2, 2, 2)
        plt.imshow(out2)
        plt.title('相关性f5 = %.5f' % f5, fontsize=14)

        plt.subplot(2, 2, 3)
        plt.imshow(vert_img1)

        plt.subplot(2, 2, 4)
        plt.imshow(vert_img2)

        plt.suptitle('旋转角度 {:d}°'.format(angle), fontsize=18)
        plt.show()

    return f4, f5


""""
# http://py3study.com/Article/details/id/20074.html
pylab.rcParams['font.sans-serif'] = ['KaiTi']
pylab.rcParams['axes.unicode_minus'] = False
pylab.hist(pixels, bins=131, range=(0, 120), color='g')
pylab.xlabel('像素值', size=20)
pylab.ylabel('频率', size=20)
pylab.title('像素值的直方图', size=20)
pylab.show()
"""
