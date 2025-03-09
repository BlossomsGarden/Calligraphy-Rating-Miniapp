# -*- coding: utf-8 -*-
# https://blog.csdn.net/shifengboy/article/details/114274271
import base64
import os
import numpy as np
from flask import Flask, request, json, jsonify
import cv2 as cv
from datetime import datetime
import requests
import urllib.request
from urllib.parse import quote, urlencode

app = Flask(__name__)

# 解决跨域问题
from flask_cors import CORS
CORS(app, resources=r'/*', origins='*', allow_headers='Content-Type', supports_credentials=True)

def http_post(url, data):
    res = urllib.request.urlopen(url, data)
    return res.read().decode('utf-8')


@app.route('/hello')  # 这里不能为空
def hello_world():
    return 'Hello World!'


from io import BytesIO
# UPLOAD_PATH='/var/www/html/images'
UPLOAD_PATH = './images'
@app.route('/upload', methods=['POST'])
def upload_pic():
    if 'imgs' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['imgs']
    # 保存到公网访问上去，但是这样加载出来太慢了，所以改用base64编码
    # if file:
    #     # 将上传的文件保存到指定路径
    #     filename = os.path.join(UPLOAD_PATH, file.filename)
    #     file.save(filename)
    #     # 返回
    #     return jsonify({'url': "http://139.196.197.42:81/images/"+file.filename})

    # 检查是否有文件
    if file:
        # 将文件读取为字节流
        in_memory_file = BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        # 转换为base64编码
        b64_encoded_string = base64.b64encode(in_memory_file.read()).decode('utf-8')
        # 构造base64数据URL
        mime_type = "image/jpeg" if ".jpg" in file.filename or ".jpeg" in file.filename else "image/png"
        base64_data_url = f"data:{mime_type};base64,{b64_encoded_string}"
        # 返回base64编码的图片数据
        return jsonify({'base64': base64_data_url})
    # 如果文件不存在
    return jsonify({'error': 'No file provided'})


@app.route('/unicode', methods=['POST'])
def get_unicode():
    data = request.get_data()  # 接收json数据
    jsondata = json.loads(data)  # json数据解析

    #####################################################
    base64_data = jsondata["data"]
    parts = base64_data.split(',')
    img_type = parts[0].split(':')[1].split(';')[0].rsplit('/', 1)[1]  # 图片类型，如"png"
    img_data = parts[1]  # base64编码本体
    img_data = base64.b64decode(img_data)

    nparr = np.frombuffer(img_data, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
    cv.imwrite('resize_image.' + img_type, img)  # 保存输入的图片

    ####################
    # 笑死我来
    # 你识别模型是只能识别test.jpg
    # 你得先将图片转码为 JPG 格式
    # 才能正常识别
    ####################
    # 将img转为jpg格式的图像数据
    _, img_encoded = cv.imencode('.jpg', img)
    # 将jpg格式的图像数据编码为base64字符串
    jpg_base64_code = base64.b64encode(img_encoded).decode('utf-8')
    # 将base64字符串添加到data URI中
    base64_data = "data:image/jpeg;base64," + jpg_base64_code
    jsondata["data"] = base64_data
    #######################################################

    # 使用urlencode将字典参数序列化成字符串
    data_string = urllib.parse.urlencode(jsondata)
    # 将序列化后的字符串转换成二进制数据，因为post请求携带的是二进制参数
    last_data = bytes(data_string, encoding='utf-8')
    print("准备向服务器发送请求", last_data)
    # 向服务器发送请求
    unicode = 0
    ch = ''
    try:
        res = http_post("http://202.120.188.3:21789/api/recognize", last_data)
        jsondata = json.loads(res)
        unicode = jsondata['unicode']
        ch = jsondata['character']
        code = 200
    except urllib.error.HTTPError as e:
        print("HTTPERROR", e)
        code = e.code
    except urllib.error.URLError as e:
        print("URLERROR", e)
        error_code = None
        code = error_code

    return jsonify({
        'code': code,
        'unicode': unicode,
        'ch': ch
    })


from split import take_characters_apart
from utils import base64_to_cv2, ostu
@app.route('/split', methods=['POST'])
def get_split():
    #  保存输入的图片
    data = request.get_data()  # 接收json数据
    jsondata = json.loads(data)  # json数据解析
    base64_data = jsondata["data"]
    base64_head, img_type, img=base64_to_cv2(base64_data)

    # 开始分割
    img=ostu(img, 10)
    cv.imwrite("ostu.png",img)
    res = take_characters_apart(img, resize=True)

    # 逐一恢复为base64数据
    result=[]
    for r in res:
        _, img_encoded = cv.imencode( ('.'+img_type), r)
        base64_code = base64.b64encode(img_encoded).decode('utf-8')
        # 将base64字符串添加到data URI中
        base64_data = base64_head+ "," + base64_code
        result.append(base64_data)
    return jsonify({
        'scrap': result,
        'code':200
    })


from comment import mark_mean_stroke, mark_HWrate, mark_blackwhite, mark_G
@app.route('/comment', methods=['POST'])
def get_comment():
    data = request.get_data()  # 接收json数据
    jsondata = json.loads(data)  # json数据解析
    img = jsondata["selected"]
    demo=jsondata["demo"]
    _, __, img=base64_to_cv2(img)
    _, __, demo=base64_to_cv2(demo)

    # 获取评论
    try:
        mean_stroke=mark_mean_stroke(img, demo)     # 平均笔画宽度
        h_w_rate=mark_HWrate(img, demo)         # 宽高比
        b_w_rate=mark_blackwhite(img, demo)       # 用笔
        G=mark_G(img, demo)                    # 重心
        # return jsonify({
        #     'MeanStroke': mean_stroke,
        #     'HWRate': h_w_rate,
        #     'BWRate':b_w_rate,
        #     'G':G,
        #     'code':200
        # })
        comments=[]
        comments.append(mean_stroke)
        comments.append(h_w_rate)
        comments.append(b_w_rate)
        comments.append(G)
        return jsonify({
            'comment':comments,
            'code':200
        })

    except Exception as e:
        return jsonify({
            'error': e,
            'code':500
        })


from utils import cv2_to_PIL
from projection import get_rotation_projection_similarity
from mmd import mmd_main
from shapecontext import sc
@app.route('/score', methods=['POST'])
def get_score():
    # 从接口获得img和demo
    data = request.get_data()  # 接收json数据
    jsondata = json.loads(data)  # json数据解析=
    img = jsondata["selected"]
    demo=jsondata["demo"]
    _, __, img=base64_to_cv2(img)
    _, __, demo=base64_to_cv2(demo)

    try:
        # 像素投影
        o0, s0=get_rotation_projection_similarity(cv2_to_PIL(img), cv2_to_PIL(demo), 0, show=False, save=False)
        o90, s90=get_rotation_projection_similarity(cv2_to_PIL(img), cv2_to_PIL(demo), 90, show=False, save=False)
        o45, s45=get_rotation_projection_similarity(cv2_to_PIL(img), cv2_to_PIL(demo), 45, show=False, save=False)
        o_45, s_45=get_rotation_projection_similarity(cv2_to_PIL(img), cv2_to_PIL(demo), -45, show=False, save=False)

        print('求得0°（倾斜投影重叠百分比，相似程度）：', o0, s0)
        print('求得90°（倾斜投影重叠百分比，相似程度）：', o90, s90)    # 可以说横太短了
        print('求得45°（倾斜投影重叠百分比，相似程度）：', o45, s45)
        print('求得-45°（倾斜投影重叠百分比，相似程度）：', o_45, s_45)
        o=(o0+o90+o45+o_45)/4
        s=(s0+s90+s45+s_45)/4
        s1=o*0.8+s*(1-0.8)
        print('针对二者投影相似性进行评分：', s1*100,'%')

        # MMD取图像差异
        s2=(float)(mmd_main(img, demo))
        print("\nMMD差异计算如下:", s2)

        # 获取ShapeContext差异
        s3= sc(img, demo)
        print("\nShapeContext差异计算如下:",s3)

        # 书法风格神韵相似度评估
        s4=100

        print(s1,s2,s3,s4)
        return jsonify({
            's1':s1,
            's2':s2,
            's3':s3,
            's4':s4,
            'score': 0.2*s1+0.4*s2+0.2*s3+0.2*s4,
            'code':200
        })

    except Exception as e:
        return jsonify({
            'error': e,
            'code':500
        })


if __name__ == '__main__':
    # app.add_url_rule('/', 'hello', hello_world)   # 与@app二选一
    app.run(host='127.0.0.1', port=9999, debug=True)

