# coding=utf-8
import dlib
import cv2
import os
import glob
import numpy
import sys
import hmac, hashlib, base64

predictor_path =r"C:\Python27\shape_predictor_68_face_landmarks.dat"
face_rec_model_path = r"C:\Python27\dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)


def gotpicture():
    cap = cv2.VideoCapture(1)
    while (1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(r"C:\Users\jaska\Desktop\face\fangjian2.jpeg", frame)
            break
    cap.release()
    cv2.destroyAllWindows()


def findpicture(img_path):
    print("hello")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print("hello")
    b, g, r = cv2.split(img)
    print("hello")
    img2 = cv2.merge([r, g, b])
    print("hello")
    dets = detector(img, 1)
    print("hello")
    for index, face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))

        shape = shape_predictor(img2, face)   # 提取68个特征点
        for i, pt in enumerate(shape.parts()):
            print('Part {}: {}'.format(i, pt))
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)  #对图片标点
            print(type(pt))
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
        #cv2.namedWindow(img_path+str(index), cv2.WINDOW_AUTOSIZE)  #显示框名称
        #cv2.imshow(img_path+str(index), img)   #显示图片

        face_descriptor = face_rec_model.compute_face_descriptor(img2, shape)   # 计算人脸的128维的向量
        #print(face_descriptor)
        return face_descriptor

def comparePersonData(data1, data2):
    diff = 0
    # for v1, v2 in data1, data2:
        # diff += (v1 - v2)**2
    for i in xrange(len(data1)):
        diff += (data1[i] - data2[i])**2
    diff = numpy.sqrt(diff)
    #print diff
    if(diff < 0.6):
        str3 = bytes(data2)
        h = hmac.new(str3)
        #h_str = h.hexdigest()
        hc = hmac.new(b"key")
        hc.update(str3)
        #hash_bytes = hc.digest()
        hash_bytes = hmac.new(b"key")
        hash_str = hash_bytes.hexdigest()
        # print(hash_str)
        base64_str = base64.urlsafe_b64encode(hash_str)
        #print("It's the same person")
        return base64_str
    else:
        print "It's not the same person"

if __name__ == '__main__':
    dist=findpicture(r"C:\Users\jaska\Desktop\face\Recvfangjian2.jpeg")
    #print(dist)
    gotpicture()
    dist2=findpicture(r"C:\Users\jaska\Desktop\face\fangjian2.jpeg")
    #print(dist2)
    #dist=findpicture(r"C:\Users\jaska\Desktop\b1.jpg")
    #dist2=findpicture(r"C:\Users\jaska\Desktop\b2.jpg")
    #dist=findpicture(sys.argv[1])
    #dist2=findpicture(sys.argv[2])
    str=comparePersonData(dist,dist2)
    print(str)