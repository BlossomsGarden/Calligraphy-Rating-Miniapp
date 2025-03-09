import os.path
from datetime import datetime

import cv2 as cv
import matplotlib.pyplot as plt
from projection import get_hori_proj, get_vert_proj
from utils import cv2_to_PIL
from resize import offset_resize
from functools import cmp_to_key

# 参考：https://www.cnblogs.com/xuxianshen/p/10257380.html
def COMP(zone1, zone2):
    if zone1[1] -zone1[0] < zone2[1] - zone2[0]:
        return -1
    elif zone1[1] -zone1[0] > zone2[1] - zone2[0]:
        return 1
    else:
        return 0


################################################
# 名称：专属于split的功能附属函数
# 功能：找第一行所有字（或字间空白）的平均值
# 输入：像素数组；切割阈值；“字还是字间空白”参数
# 返回：本帖中书法字的平均长度或宽度
################################################
def mean_space(pix_array3, thresh3, flag='BLACK'):
    k2 = 0
    start = 0
    end = 0
    record = []
    while True:
        while k2 < len(pix_array3):
            if pix_array3[k2] > thresh3:
                start = k2
                break
            k2 += 1
        while k2 < len(pix_array3):
            if pix_array3[k2] <= thresh3:
                end = k2 - 1
                break
            k2 += 1
        record.append([start, end])

        if k2 >= len(pix_array3)-1:
            break

    total_space = 0
    if flag == 'BLACK':         # 黑像素值，求字宽
        record.sort(key=cmp_to_key(COMP))
        if len(record) > 6:
            record.pop(0)
            record.pop(0)
            record.pop(-1)
            record.pop(-1)
        for sub_record in record:
            total_space += sub_record[1] - sub_record[0]
        total_space /= len(record)

    else:                       # 白像素值，求字间宽度
        for k3 in range(1, len(record)):
            total_space += record[k3][0] - record[k3-1][1]
        total_space /= (len(record) - 1)
    return total_space


################################################
# 名称：获取图像切割位点函数
# 功能：从像素列表中寻找哪里到哪里该被切开
# 输入：输入像素值数组；切割阈值
# 返回：含待切割区间始末点键-值对的字典
################################################
def split_zone(pix_array2, thresh2):
    dic1 = {}
    k1 = 0
    sp_start = 0
    sp_end = 0

    # 粗切割
    while True:
        while k1 < len(pix_array2):
            if pix_array2[k1] > thresh2:
                sp_start = k1
                break
            k1 += 1

        while k1 < len(pix_array2):
            if pix_array2[k1] <= thresh2:
                sp_end = k1 - 1
                dic1[sp_start] = sp_end
                break
            k1 += 1

        if k1 >= len(pix_array2):                       # 在k跑遍所有行（或列）之前，继续循环向下寻找是否还有一行（或一列）字
            break

    # print('粗切割：', dic1)

    # 细切割（扔噪点、保证卿、兆、大写宝正常输出）
    std_ch_width = mean_space(pix_array2, thresh2, 'BLACK')*3/2    # 取“标准字长”
    std_blank_width = mean_space(pix_array2, thresh2, 'BLANK')        # 取“标准空白长”
    std_gap = min(std_blank_width, std_ch_width)                      # 取“标准切割间距”
    # print('切割标准字宽：', str(std_ch_width))
    # print('切割标准空白长：', str(std_blank_width))
    # print()

    # 1.记录被切开的字（不足字宽的字）
    dic2 = {}
    zone_start1 = []
    for key in dic1.keys():
        zone_start1.append(key)

    for sp_start1 in zone_start1:
        if dic1[sp_start1] - sp_start1 <= 4:
            del dic1[sp_start1]                                 # 分割长度不足3，认定是噪点，舍去

        elif dic1[sp_start1] - sp_start1 < std_ch_width:        # 宁可错杀一字，不能放过“被切开但是另一部分没进来”的情况（如“忘”）
            dic2[sp_start1] = dic1[sp_start1]                   # 非噪点，但区间宽度不到“标准字长”，认为是字被切开了，记录并删除
            del dic1[sp_start1]

    # print('不足标准字宽：', end='')
    # print(dic2)

    # 2.重新处理被切开的字，处理完毕与dic1合并
    zone_start2 = []      # 记录切割区间的起点
    for key in dic2.keys():
        zone_start2.append(key)

    # 定义flag标签数组，用以对应zone_start数组，记录这2个或3个区间是同一个字被割开了——0表示该字开始，1表示该字的分割区间，直到遇见下一个0
    combine_flag = [0] * len(zone_start2)

    k2 = 1
    while k2 < len(zone_start2):
        # 考虑到字宽与字间距之间的大小关系不是固定的，判断是否是同一个字被切开而不是两个相距不大的字（或列）贴在一起
        # 引入“标准量”来做比较，即
        # 1.前后两切割区间的距离应满足：字宽大于字距时，要比字距小；字距大于字宽时，要比字宽小
        # 2.切割区间合并后，总长不会超过“标准字长”
        if zone_start2[k2] - dic2[zone_start2[k2-1]] < std_gap\
                 and ((dic2[zone_start2[k2]]-zone_start2[k2]) < std_ch_width * 1/2 or  (dic2[zone_start2[k2-1]]-zone_start2[k2-1]) < std_ch_width * 1/2):
            combine_flag[k2] = 1
        k2 += 1

    # print(zone_start2)
    # print(combine_flag)

    k3 = 0
    while k3 < len(zone_start2):
        while combine_flag[k3] == 0:
            sp_start = zone_start2[k3]
            if k3 == len(zone_start2) - 1:
                sp_end = dic2[zone_start2[k3]]        # 其后已经没有下一个数了，说明这个区间里存的是一个不足字长的字
                dic1[sp_start] = sp_end
            else:                                     # 即if k3 < len(zone_start2) - 1
                if combine_flag[k3 + 1] == 0:
                    sp_end = dic2[zone_start2[k3]]    # 后面那个也是0，说明这个区间里存的是一个不足字长的字
                    dic1[sp_start] = sp_end
            k3 += 1
            if k3 >= len(zone_start2):
                break

        if k3 >= len(zone_start2):
            break

        while combine_flag[k3] == 1:
            sp_end = dic2[zone_start2[k3]]
            if k3 <= len(zone_start2) - 1:
                dic1[sp_start] = sp_end
            k3 += 1
            if k3 >= len(zone_start2):
                break
    # end of while
    #
    # print('最终切割区间：', dic1)
    # print()
    return dic1


################################################
# 名称：图像切割函数
# 功能：实现图像切割
# 输入：像素数组；切割阈值；“横纵切割”参数；“是否打印”参数；“是否保存”参数
# 返回：分割结果的图片列表
################################################
def split_images(inti_image, dic, flag1='HORIZONAL_CUT', show=True, save=False):
    c = inti_image.shape                                # 方便读取输入图片的宽高
    result_images = []                                  # 处理的结果图像将被储存在这里面

    if flag1 == 'HORIZONAL_CUT':
        i1 = 1
        for x1, x2 in dic.items():
            final_image = inti_image[x1:x2, 0:c[1]]     # 水平方向切开，竖直方向不变
            result_images.append(final_image)
            if save:
                cv.imwrite('col-result/Column_' + str(i1) + '.jpg', final_image)
            i1 += 1
    else:                                               # 使用else而非'VERTICAL_CUT'的原因同第52行
        i2 = 1
        for y1, y2 in dic.items():
            final_image = inti_image[0:c[0], y1:y2]     # 竖直方向切开，水平方向不变
            result_images.append(final_image)
            if save:
                cv.imwrite('col-result/Column_' + str(i2) + '.jpg', final_image)
            i2 += 1

    if show:
        sp_zone_num = len(dic)  # 记录字典内键值对（待分割图像区间的个数）
        for n in range(sp_zone_num):
            if flag1 == 'HORIZONAL_CUT':
                plt.subplot(sp_zone_num, 1, n + 1)
            else:
                plt.subplot(1, sp_zone_num, n + 1)
            plt.imshow(result_images[n], 'gray')
            plt.xticks([])
            plt.yticks([])

        plt.show()

    return result_images

# 如果把“开始纵切”的代码去掉，可以直接用下面这段话代替，在col-result中放好图片，然后开启横切
# col_split_re=read_col_split_results("col-result")
def read_col_split_results(path):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    files.sort(key=lambda x: int(x[:-4]))
    img_list=[]
    for file in reversed(files):  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            t=cv.imread(path+"/"+file)
            img_list.append(t)
    return img_list


###############################################################################################
# MAIN函数1：纵切
###############################################################################################
SAVE_DIR='split-result'
def take_characters_apart(img, resize=False):
    # 构造结果数组
    characters = []

    # 构造保存文件夹
    now = datetime.now()    # 获取当前日期和时间
    date_folder = now.strftime("%Y-%m-%d")
    time_folder = now.strftime("%H-%M-%S")
    folder_path = os.path.join(SAVE_DIR, date_folder, time_folder)      # 构建文件夹路径
    if not os.path.exists(folder_path):     # 检查文件夹是否存在
        os.makedirs(folder_path)

    print('开始纵切：')
    pix1, _ = get_vert_proj(cv2_to_PIL(img))    # 垂直投影像素总量
    cv.imwrite("col-result/pro.png", _)  # 保存一下投影图
    # 这行代码是用来给pix1作平滑的，这样子图上不会出现过多的波峰波谷，不过留给后人调试吧
    # from scipy.ndimage import uniform_filter1d
    # pix1 = uniform_filter1d(pix1, size=math.ceil(len(pix1)/80)).astype(int)
    col_split = split_zone(pix1, 0)                      # 横向切割的所有分割区间键值对
    col_split_re = split_images(img, col_split, 'VERTICAL_CUT', show=False, save=True)  # 获取横向切割的结果图片数组并打印

    print('开始横切：')
    for k in range(len(col_split_re)):
        pix2, _ = get_hori_proj(cv2_to_PIL(col_split_re[k]))     # 开一个数组记录每列的黑像素总量
        row_split = split_zone(pix2, 0)         # 获取纵向切割的所有分割区间键值对
        row_split_re = split_images(col_split_re[k], row_split, 'HORIZONAL_CUT', show=False, save=False)   # 获取纵向切割的结果图片数组并打印

        for i in range(len(row_split_re)):
            img_path = os.path.join(folder_path, ('col' + str(k) + '-row' + str(i) + '.jpg'))
            if resize:
                resize_img = offset_resize(row_split_re[i])
                characters.append(resize_img)
                cv.imwrite(img_path, resize_img)
            else:
                characters.append(row_split_re[i])
                cv.imwrite(img_path, row_split_re[i])

    print("切割完毕")
    return characters
