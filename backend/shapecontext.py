import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
from utils import get_contour_sample_points, get_ske_sample_points
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
import math
from munkres import Munkres
from utils import G_points


# 计算点集的角度矩阵
def compute_angles(points):
    """ compute angles between a set of points """
    angles = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        p1 = points[i, :]
        for j in range(i + 1, len(points)):
            p2 = points[j, :]
            angles[i, j] = math.atan2((p1[1] - p2[1]), (p2[0] - p1[0]))
            angles[j, i] = -angles[i, j]
    # angles between 0, 2*pi
    angles = np.fmod(np.fmod(angles, 2 * pi) + 2 * pi, 2 * pi)
    return angles


# 计算形状上下文直方图
def compute_histogram(points, nbins_theta=12, r1=0.125, r2=2.0):
    """ quantize angles and radious for all points in the image """
    # compute distance between points
    distmatrix = np.sqrt(pdist(points))
    mean_dist = np.mean(distmatrix)
    distmatrix = distmatrix / mean_dist
    distmatrix = squareform(distmatrix)
    # compute angles between points
    angles = compute_angles(points)
    # quantize angles to a bin
    tbins = np.floor(angles / (2 * pi / nbins_theta))
    lg = np.logspace(r1, r2, num=5)
    # quantize radious to bins
    rbins = np.ones(angles.shape) * -1
    for r in lg:
        counts = (distmatrix < r)
        rbins = rbins + counts.astype(int)
    return rbins, tbins


# 求形状上下文直方图矩阵（归一化）
def compute_features(points, num_points, desc_size=60,  nbins_theta=12, nbins_r=5):
    """ construct shape context feature for all sampled points in the image"""
    rbins, tbins = compute_histogram(points)
    inside = (rbins > -1).astype(int)
    features = np.zeros((num_points, desc_size))
    #construct the feature
    for p in range(num_points):
        rows = []
        cols = []
        for i in range(num_points):
            if inside[p,i]:
                rows.append(tbins[p,i])
                cols.append(rbins[p,i])
        bins = np.ones((len(rows)))
        a = csr_matrix((bins,(np.array(rows), np.array(cols))), shape=(nbins_theta, nbins_r)).todense()
        features[p, :] = a.reshape(1, desc_size) / np.sum(a)
    return features


# 求相似性度量矩阵
def CS(num_points,num_points2,features,features2):
    """ Compute cost of matching shape context features between shapes.
        The smaller the cost the most similar they are.
        Cost is the chi square distance between two features"""

    cost = np.zeros((num_points, num_points2))
    a1, b1 = np.array(features).shape
    a2, b2 = np.array(features2).shape
    if a1 > a2:
        sc1 = features
        sc2 = features2
    else:
        sc1 = features2
        sc2 = features
        a1, a2 = a2, a1

    feat1 = np.tile(sc2, (a1, 1, 1))
    feat2 = np.tile(sc1, (a2, 1, 1))
    feat2 = np.transpose(feat2, (2, 0, 1))
    feat1 = np.transpose(feat1)
    cost = 0.5 * np.sum(np.power(feat1 - feat2, 2) / (feat1 + feat2+np.finfo(float).eps), axis=0)
    # sc_cost = np.mean(np.amin(cost, axis=1)) + np.mean(np.amin(cost, axis=0))
    return cost


# sobel算子帮助求梯度
def sobel(img,x,y):
    conv1=[[-1,0,1], [-2,0,2], [-1,0,1]]
    conv2=[[1,0,-1], [2,0,-2], [1,0,-1]]
    Gx=Gy=0
    for i in range(-1,2):
        for j in range(-1,2):
            if x<img.shape[0]-1 and y<img.shape[1]-1:
                Gx+=conv1[i+1][j+1]*img[x+i][y+j]
                Gy+=conv2[i+1][j+1]*img[x+i][y+j]
    return Gx,Gy


# 求局部外观矩阵
def CA(img1, img2, points1,points2):
    length=len(points1)
    Ca = np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            gx1,gy1=sobel(img1, points1[i][0],points1[i][1])
            angle1=math.atan2(gy1,gx1)
            gx2, gy2 = sobel(img2, points2[j][0], points2[j][1])
            angle2 = math.atan2(gy2 , gx2)
            Ca[i][j]=0.5*math.sqrt(pow(math.cos(angle1)-math.cos(angle2), 2)+ pow(math.sin(angle1)-math.sin(angle2), 2))
    return Ca


# 匈牙利算法求最佳匹配
def Hungarian(cost):
    """ Hungarian assigment. It returns the best matches between 2 sets of points """
    rows = cost.shape[0] - cost.shape[1]
    if rows > 0:
        extra = np.zeros((cost.shape[0] , rows))
        cost = np.hstack((cost, extra))
    m = Munkres()
    indexes = m.compute(cost)
    return  indexes


def TPS(source_points, target_points, img_input, N):
    # N对基准控制点

    img_init = np.ones((256, 256), np.uint8)
    for i in range(256):
        for j in range(256):
            img_init[i, j] *= 255
            if i % 8 == 0 and j % 8 == 0:
                img_init[i, j] = 0
    for [x, y] in source_points:
        cv.circle(img_init, [y, x], 1, 0, 3)
    # cv.imshow('init', img_init)

    # 这句话报错，安装opencv-contrib-python库就好了
    tps = cv.createThinPlateSplineShapeTransformer()

    sourceshape = np.array(source_points, np.int32)
    sourceshape = sourceshape.reshape(1, -1, 2)

    matches = []
    for i in range(1, N + 1):
        matches.append(cv.DMatch(i, i, 0))


    img_init2 = np.ones((256, 256), np.uint8)
    for i in range(256):
        for j in range(256):
            img_init2[i, j] *= 255
            if i % 8 == 0 and j % 8 == 0:
                img_init2[i, j] = 0
    for [x, y] in target_points:
        cv.circle(img_init2, [y, x], 1, 0, 3)
    cv.imshow('init2', img_init2)

    targetshape = np.array(target_points, np.int32)
    targetshape = targetshape.reshape(1, -1, 2)
    tps.estimateTransformation(sourceshape, targetshape, matches)


    cv.imshow('input', img_input)
    re2 = tps.warpImage(img_init)

    # cv.imshow('tmp1', re1)
    cv.imwrite('result'+str(N)+'.jpg', re2)
    return re2


################################################################
# 接口函数，第一次需要先初步匹配一下
################################################################
def first_match(img,img2):
    points = get_contour_sample_points(img, 200, False)
    points = np.asarray(points)
    features = compute_features(points, points.shape[0])  # 求得形状上下文直方图矩阵

    points2 = get_contour_sample_points(img2, 200, False)
    points2=np.asarray(points2)     # 将list类型转为array类型
    features2 = compute_features(points2, points2.shape[0])    # 求得形状上下文直方图矩阵

    Cs = CS(points.shape[0], points2.shape[0], features, features2)
    Ca = CA(img,img2,points,points2)

    beta = 0.1
    C = (1 - beta) * Cs + beta * Ca     # 求得总相似度度量矩阵
    return  points, points2, Hungarian(C)


# 经过匈牙利算法之后，更新一下点集的对应关系
def second_match(img, img2, points, points2, index):
    # 修正点集位置，尽可能重合
    gx1, gy1 = G_points(img, points)
    gx2, gy2 = G_points(img2, points2)
    dx = gx2 - gx1
    dy = gy2 - gy1

    reform1 = []
    reform2 = []
    for (i, j) in index:
        x1 = points[i][0]
        x2 = (int)(points2[j][0] - dx)
        y1 = points[i][1]
        y2 = (int)(points2[j][1] - dy)
        if (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) <= 20 * 20:    # 超过20认为匹配出错，遗弃
            reform1.append([x1, y1])
            reform2.append([x2, y2])

    # 轮廓点匹配的图形化
    # plt.title('contour_match', fontsize=24)
    # for i in range(len(reform2)):
    #     x = []
    #     y = []
    #
    #     x1 = reform1[i][1]
    #     x2 = reform2[i][1]
    #     y1 = reform1[i][0]
    #     y2 = reform2[i][0]
    #     x.append(x1)
    #     x.append(x2)
    #     y.append(y1)
    #     y.append(y2)
    #     plt.plot(x, y, color='r')  # 连线，按顺序将
    #     # plt.scatter(x1, y1, color='g')
    #     # plt.scatter(x2, y2, color='b')
    #
    #     plt.scatter(x1, y1, color='g')
    #     plt.scatter(x2, y2, color='b')
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.xlim(left=0, right=256)
    # plt.ylim(bottom=256, top=0)
    # plt.savefig('skeleton_match' + '.jpg')

    # TPS插值的图形化
    # # 构造一个输入图像
    # img_input = np.ones((256, 256), np.uint8)
    # for i in range(256):
    #     for j in range(256):
    #         img_input[i, j] *= 255
    #         if i % 8 == 0 and j % 8 == 0:
    #             img_input[i, j] = 0
    # for k in range(3,15):
    #     result = TPS(points, points2,img_input,k)
    #
    #     plt.figure("result")
    #     plt.set_cmap('binary')
    #     plt.imshow(1-result)
    #     for [x,y] in reform2:
    #         plt.scatter(y, x, marker='+',color='r')
    #     # plt.show()
    #     plt.savefig('re'+str(k)+'.jpg')

    return reform1, reform2


def sc(img, img2):
    # 第一次匹配的结果
    points, points2, first_corresponse = first_match(img, img2)
    # 第二次匹配，获得一一对应的点集
    points, points2 = second_match(img, img2, points, points2, first_corresponse)
    # 重新计算SC矩阵
    points = np.asarray(points)
    points2 = np.asarray(points2)
    features = compute_features(points, points.shape[0])
    features2 = compute_features(points2, points2.shape[0])
    Cs = CS(points.shape[0], points2.shape[0], features, features2)
    Ca = CA(img,img2,points,points2)
    beta = 0.1
    C = (1 - beta) * Cs + beta * Ca
    # 获取SC差异值
    return  np.sum(np.min(C, axis=1))/C.shape[0] + np.sum(np.min(C, axis=0))/C.shape[1]

