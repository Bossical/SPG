# coding=utf-8
import sys
import dlib
import cv2
import os
import glob
import numpy

# 模型路径
predictor_path =r"C:\Python27\shape_predictor_68_face_landmarks.dat"
face_rec_model_path = r"C:\Python27\dlib_face_recognition_resnet_model_v1.dat"
#测试图片路径
faces_folder_path = r"C:\Users\jaska\Desktop\face"

# 读入模型
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

face_descriptor=[0,0]

k=0

for img_path in glob.glob(os.path.join(faces_folder_path, "*.jpeg")):
    print("Processing file: {}".format(img_path))
    # opencv 读取图片，并显示
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # opencv的bgr格式图片转换成rgb格式
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])

    dets = detector(img, 1)   # 人脸标定
    print("Number of faces detected: {}".format(len(dets)))


    for index, face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))

        shape = shape_predictor(img2, face)   # 提取68个特征点
        for i, pt in enumerate(shape.parts()):
            #print('Part {}: {}'.format(i, pt))
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
            #print(type(pt))
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
        cv2.namedWindow(img_path+str(index), cv2.WINDOW_AUTOSIZE)
        cv2.imshow(img_path+str(index), img)

        face_descriptor[k] = face_rec_model.compute_face_descriptor(img2, shape)   # 计算人脸的128维的向量
        print(face_descriptor[k])
    k+=1

print("Hello")

def comparePersonData(data1, data2):
    diff = 0
    # for v1, v2 in data1, data2:
        # diff += (v1 - v2)**2
    for i in xrange(len(data1)):
        diff += (data1[i] - data2[i])**2
    diff = numpy.sqrt(diff)
    print diff
    if(diff < 0.6):
        print "It's the same person"
    else:
        print "It's not the same person"

comparePersonData(face_descriptor[0],face_descriptor[1])
k = cv2.waitKey(0)
cv2.destroyAllWindows()