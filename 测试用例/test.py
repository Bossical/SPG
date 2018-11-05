# coding=utf-8
import dlib
import cv2
import os
import glob
import numpy
import sys
import hmac, hashlib, base64
import time, threading
import xlwt
import random
import string

from matplotlib import pyplot as plt
import numpy as np


predictor_path = r"C:\Python27\shape_predictor_68_face_landmarks.dat"
face_rec_model_path = r"C:\Python27\dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)


def findpicture(img_path,data_left,data_right,data_bottom,data_top):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    dets = detector(img, 1)
    if(dets==None):
        data_left.append(0)
        data_top.append(0)
        data_right.append(0)
        data_bottom.append(0)
        print(data_left)
        return 0
    else:
        for index, face in enumerate(dets):
            print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
                                                                         face.bottom()))

            data_left.append(face.left())
            data_top.append(face.top())
            data_right.append(face.right())
            data_bottom.append(face.bottom())
            print(data_left)

            shape = shape_predictor(img2, face)  # 提取68个特征点
            for i, pt in enumerate(shape.parts()):
                # print('Part {}: {}'.format(i, pt))
                pt_pos = (pt.x, pt.y)
                cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)  # 对图片标点
                # print(type(pt))
            # print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
            # cv2.namedWindow(img_path+str(index), cv2.WINDOW_AUTOSIZE)  #显示框名称
            # cv2.imshow(img_path+str(index), img)   #显示图片

            face_descriptor = face_rec_model.compute_face_descriptor(img2, shape)  # 计算人脸的128维的向量
            # print(face_descriptor)
            return face_descriptor



def comparePersonData(data1, data2):
    diff = 0
    # for v1, v2 in data1, data2:
    # diff += (v1 - v2)**2
    # print data1
    if data1 == None or data2 == None:
        print("0")
        return 0
    for i in xrange(len(data1)):
        diff += (data1[i] - data2[i]) ** 2
    diff = numpy.sqrt(diff)
    print(diff)
    return diff

def comparacc(data1):
    str3 = bytes(data1)
    h = hmac.new(str3)
    # h_str = h.hexdigest()
    hc = hmac.new('hello',h.hexdigest())
    hc.update(str3)
    # hash_bytes = hc.digest()
    hash_bytes = hmac.new('key', hc.hexdigest())
    hash_str = hash_bytes.hexdigest()
    # print(hash_bytes.hexdigest())
    base64_str = base64.urlsafe_b64encode(hash_str)
    # print("It's the same person")
    return base64_str

def dif1comparacc(data1):
    str3 = bytes(data1)
    h = hmac.new(str3)
    # h_str = h.hexdigest()
    salt = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    hc = hmac.new(salt, h.hexdigest())
    hc.update(str3)
    # hash_bytes = hc.digest()
    hash_bytes = hmac.new(salt, hc.hexdigest())
    hash_str = hash_bytes.hexdigest()
    # print(hash_bytes.hexdigest())
    base64_str = base64.urlsafe_b64encode(hash_str)
    # print("It's the same person")
    return base64_str

def dif2comparacc(data1):
    str3 = bytes(data1)
    h = hmac.new(str3)
    # h_str = h.hexdigest()
    salt = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    hc = hmac.new(salt, h.hexdigest())
    hc.update(str3)
    # hash_bytes = hc.digest()
    salt1 = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    hash_bytes = hmac.new(salt1, hc.hexdigest())
    hash_str = hash_bytes.hexdigest()
    # print(hash_bytes.hexdigest())
    base64_str = base64.urlsafe_b64encode(hash_str)
    # print("It's the same person")
    return base64_str

if __name__ == '__main__':
    data_left=[]
    data_right=[]
    data_top=[]
    data_bottom=[]

    data_dis=[]
    mks=0

    for i in range(40):
        print("第"+str(i)+"组图片测试")
        for j in range(4):
            if j+7 ==10:
                break
            data1=findpicture(r"C:\Users\jaska\Desktop\face\test_face\s"+str(i+1)+'\\'+str(j+7)+'.jpg', data_left,
                              data_right, data_bottom, data_top)
            print(data1)
            for k in range(4):
                if k <= j :
                    continue
                else:
                    data2 = findpicture(r"C:\Users\jaska\Desktop\face\test_face\s" + str(i+1) + '\\' + str(k + 7) +
                                        '.jpg', data_left, data_right, data_bottom, data_top)
                    print(data2)
                    print(r"第" + str(j + 7) + "张与" + "第" + str(k + 7) + "张的测试数据")
                    dif = comparePersonData(data1, data2)
                    print("mks="+str(mks))
                    data_dis.append(dif)
                    mks = mks + 1
    print(data_dis)

    data_left = []
    data_right = []
    data_top = []
    data_bottom = []

    data_dis_all=[]
    data5=findpicture(r'C:\Users\jaska\Desktop\face\test_face\all'+'\\'+str(0)+'.jpg', data_left,
                              data_right, data_bottom, data_top)
    for i in range(159):
        data4=findpicture(r'C:\Users\jaska\Desktop\face\test_face\all'+'\\'+str(i)+'.jpg', data_left,
                              data_right, data_bottom, data_top)
        dif1=comparePersonData(data4,data5)
        data_dis_all.append(dif1)

    new_data_left = []
    new_data_right = []
    new_data_top = []
    new_data_bottom = []
    print(len(data_left), len(data_right), len(data_bottom), len(data_top))
    k = 0
    for i in range(len(data_bottom)):
        print(k)
        if k + 1 >= len(data_left):
            break
        a = data_left[k + 1] - data_left[k]
        new_data_left.append(a)
        a = data_right[k + 1] - data_right[k]
        new_data_right.append(a)
        a = data_top[k + 1] - data_top[k]
        new_data_top.append(a)
        a = data_bottom[k + 1] - data_bottom[k]
        new_data_bottom.append(a)
        k = k + 2

    comacc=[]
    comaccd1=[]
    comaccd2=[]
    for i in range(159):
        data3=findpicture(r'C:\Users\jaska\Desktop\face\test_face\all'+'\\'+str(i)+'.jpg', data_left,
                              data_right, data_bottom, data_top)
        comacc.append(comparacc(data3))

    for i in range(159):
        comaccd1.append(dif1comparacc(data3))
        comaccd2.append(dif2comparacc(data3))


    bigs=[]
    for i in range(len(comacc)):
        print(comacc[i])
        for j in range(len(comacc)):
            if i >= j:
                continue
            print(comacc[j])
            k = 0
            m=0
            big = 0
            bigp=0
            while k < 44 and m < 44:
                if comacc[i][m] == comacc[j][k]:
                    print("1", comacc[i][m], comacc[j][k])
                    k = k + 1
                    m = m + 1
                    big = big + 1
                else:
                    print("0", comacc[i][m], comacc[j][k])
                    k = 0
                    if big > bigp:
                        bigp = big
                    big = 0
                    m = m + 1
            bigs.append(bigp)
    print(bigs)

    bigs2=[]
    for i in range(len(comaccd1)):
        print(comaccd1[i])
        for j in range(len(comaccd1)):
            if i >= j:
                continue
            print(comaccd1[j])
            k = 0
            m=0
            big = 0
            bigp=0
            while k < 44 and m < 44:
                if comaccd1[i][m] == comaccd1[j][k]:
                    print("1", comaccd1[i][m], comaccd1[j][k])
                    k = k + 1
                    m = m + 1
                    big = big + 1
                else:
                    print("0", comaccd1[i][m], comaccd1[j][k])
                    k = 0
                    if big > bigp:
                        bigp = big
                    big = 0
                    m = m + 1
            bigs2.append(bigp)
    print(bigs2)

    bigs3 = []
    for i in range(len(comaccd2)):
        print(comaccd2[i])
        for j in range(len(comaccd2)):
            if i >= j:
                continue
            print(comaccd2[j])
            k = 0
            m = 0
            big = 0
            bigp = 0
            while k < 44 and m < 44:
                if comaccd2[i][m] == comaccd2[j][k]:
                    print("1", comaccd2[i][m], comaccd2[j][k])
                    k = k + 1
                    m = m + 1
                    big = big + 1
                else:
                    print("0", comaccd2[i][m], comaccd2[j][k])
                    k = 0
                    if big > bigp:
                        bigp = big
                    big = 0
                    m = m + 1
            bigs3.append(bigp)
    print(bigs3)

    wb = xlwt.Workbook()
    ws = wb.add_sheet('A test sheet')

    ws.write(0, 0, "相同人的欧拉距离".decode('utf-8'))
    ws.write(0, 1, "不相同人的top数据".decode('utf-8'))
    ws.write(0, 2, "不相同人的bottom数据".decode('utf-8'))
    ws.write(0, 3, "不相同人的left数据".decode('utf-8'))
    ws.write(0, 4, "不相同人的right数据".decode('utf-8'))
    ws.write(0, 5, "不同人的欧拉距离".decode('utf-8'))
    ws.write(0, 6, "密码数据".decode('utf-8'))
    ws.write(0, 7, "不同人脸数据相同口令密码串之间最大字串".decode('utf-8'))
    ws.write(0, 8, "同人脸数据（单次弱口令和网址）相同密码串之间最大字串".decode('utf-8'))
    ws.write(0, 9, "同人脸数据（单次弱口令和网址）不相同密码串之间最大字串".decode('utf-8'))

    for i in range(len(data_dis)):
        ws.write(i+1, 0, data_dis[i])

    for i in range(len(new_data_top)):
        ws.write(i+1, 1, new_data_top[i])
        ws.write(i+1, 2, new_data_bottom[i])
        ws.write(i+1, 3, new_data_left[i])
        ws.write(i+1, 4, new_data_right[i])

    for i in range(len(data_dis_all)):
        ws.write(i+1, 5, data_dis_all[i])

    for i in range(len(comacc)):
        ws.write(i+1, 6, comacc[i])


    for i in range(len(bigs)):
        ws.write(i+1,7,bigs[i])

    for i in range(len(bigs2)):
        ws.write(i+1,8,bigs2[i])

    for i in range(len(bigs3)):
        ws.write(i+1,9,bigs3[i])

    wb.save(r"C:\Users\jaska\Desktop\face\test.xls")
