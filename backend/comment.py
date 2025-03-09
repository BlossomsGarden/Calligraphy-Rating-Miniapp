from utils import get_ske_sample_points, fg_points_percent
import cv2 as cv

# 获取平均宽度
def get_mean_stroke(img):
    ske_points = get_ske_sample_points(img, 100)
    d=[]
    for points in ske_points:
        r=1
        while True:
            if fg_points_percent(img, points[0], points[1], r) < 0.95:
                break
            r += 1
        d.append(r)
    return sum(d)/len(d)


# 笔画平均宽度评价函数
def mark_mean_stroke(img1, img2):  # img1为输入图片，img2为参考字图片
    value = (((get_mean_stroke(img1)) - (get_mean_stroke(img2))) / get_mean_stroke(img2))
    if value > 0.15:
        return "笔画平均宽度较大"
    elif value < -0.15:
        return "笔画平均宽度不足"
    else:
        return "笔画平均宽度在合理范围内"


# 轮廓宽高处理
def mark_recognition(img,thresh1=0):
    img = cv.merge([img, img, img])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度化
    box = cv.boxFilter(gray, -1, (3, 3), normalize=True)  # 去噪
    _, binarized = cv.threshold(box, thresh1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值化
    contours, hierarchy = cv.findContours(binarized, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0, 0, 255), 3)
    bounding_boxes = [cv.boundingRect(cnt) for cnt in contours]
    for bbox in bounding_boxes:
        [x, y, w, h] = bbox
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return w,h


#轮廓宽高比评价函数
def mark_HWrate(img1, img2):		#img1为输入图片，img2为参考字图片
    w1,h1=mark_recognition(img1)
    w2,h2=mark_recognition(img2)
    ww=(abs(w1-w2))/w1
    hh=(abs(h1-h2))/h1
    if (w1 > w2)and(ww > 0.1):
        return "字体偏宽"
    elif (w1 < w2)and(ww > 0.1):
        return "字体偏窄"
    elif((h1>h2)and(hh>0.1)):
        return("字体偏长")
    elif((h1<h2)and(hh>0.1)):
        return("字体偏扁")
    else:
        return("字体高度适中")


#黑白像素比例获取
def mark_get(img,thresh1=0):
    img = cv.merge([img, img, img])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度化
    box = cv.boxFilter(gray, -1, (3, 3), normalize=True)  # 去噪
    _, binarized = cv.threshold(box, thresh1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值化
    contours, hierarchy = cv.findContours(binarized, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv.boundingRect(cnt) for cnt in contours]
    for bbox in bounding_boxes:
        [x, y, w, h] = bbox
        cv.rectangle(img, (x, y), ( w,  h), (0, 255, 0), 2)
        cv.imshow('1', img)
        return (len(img[img == 255])),(len(img[img == 0]))


def mark_blackwhite(img1, img2):		#img1为输入图片，img2为参考字图片
    w1,b1=mark_get(img1)
    w2,b2=mark_get(img2)
    rate1=float(w1)/b1
    rate2=float(w2)/b2
    diff=rate1-rate2
    if(diff>0.1):
        return("用笔太重")
    elif(diff<-0.1):
        return("用笔太轻")
    else:
        return("用笔力度适当")


#重心位置获取
def center(img,thresh1=0):
    img = cv.merge([img, img, img])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度化
    box = cv.boxFilter(gray, -1, (3, 3), normalize=True)  # 去噪
    _, binarized = cv.threshold(box, thresh1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值化
    contours, hierarchy = cv.findContours(binarized, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return(cx,cy)


# 重心位置评价函数
def mark_G(img1, img2):
    x1, y1 = center(img1)
    x2, y2 = center(img2)
    xx=(x1-x2)/x2
    yy=(y1-y2)/y2
    if((yy>0.1)and(yy<0.2)):
        if(xx>0.15):
            return("重心略低且向右偏离")
        elif(xx<-0.15):
            return ("重心略低且向左偏离")
        else:
            return("重心略低")
    elif((yy<-0.1)and(yy>-0.2)):
        if (xx > 0.15):
            return ("重心略高且向右偏离")
        elif (xx < -0.15):
            return ("重心略高且向左偏离")
        else:
            return ("重心略高")
    else:
        if (xx > 0.15):
            return ("重心向右偏离")
        elif (xx < -0.15):
            return ("重心向左偏离")
        else:
            return ("重心在合理范围内")