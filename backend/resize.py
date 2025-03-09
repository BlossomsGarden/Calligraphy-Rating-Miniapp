import cv2 as cv
import numpy as np

# resize总规则：重心对准结果图中心
# 当宽与高相比更大时，以宽为准，使像素点分布在水平6~250之间
# 当宽与高相比更小时，以高为准，使像素点分布在竖直25~230之间

def grid_split(img):
    # 纵向重新切割，避免有些字整体纵切时留下较多空白
    H,W = img.shape
    # print('切割前图像的宽*高为:' + str(W) + '*' + str(H))

    imgL = W  # 最左
    imgR = 0  # 最右
    for i in range(H):
        for j in range(W):
            if img[i][j] == 0:
                if j < imgL:
                    imgL = j
                if j > imgR:
                    imgR = j

    imgH = H
    imgW = imgR-imgL
    # print('调整横向间距的切割区间为:', imgL, imgR)
    img = img[0:imgH, imgL:imgR]
    # print('切割后图像的宽*高为:' + str(imgW) + '*' + str(imgH))
    return img

def resize(img):
    img=grid_split(img)
    height, width = img.shape

    # 计算宽高比
    aspect_ratio = width / height
    # 创建目标图像
    if aspect_ratio >= 1:
        # 当宽与高相比更大时，以宽为准，使像素点分布在水平16~240之间
        new_width = 240-16
        new_height = int(new_width / aspect_ratio)
    else:
        # 当宽与高相比更小时，以高为准，使像素点分布在竖直35~220之间
        new_height = 220-35
        new_width = int(new_height * aspect_ratio)

    # 重设图像大小
    img = cv.resize(img, (int(new_width), int(new_height)))
    return img


def offset_resize(img):
    img = resize(img)
    # 指定目标尺寸
    target_width, target_height = 256, 256
    # 创建一个白色背景图像
    white_bk = np.ones((target_height, target_width), dtype=np.uint8) * 255
    # 计算需要添加的边框宽度
    left_border = (target_width - img.shape[1]) // 2
    top_border = (target_height - img.shape[0]) // 2
    # 将原始图像复制到白色背景中央
    white_bk[top_border:top_border + img.shape[0], left_border:left_border + img.shape[1]] = img
    # # 保存扩充后的图像
    # cv.imwrite('expanded_image.jpg', white_bk)
    return white_bk
