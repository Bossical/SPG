import cv2 , dlib
import numpy as np

'''
模型载入
'''
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('C:/Python27/shape_predictor_68_face_landmarks.dat')


img=cv2.imread('C:/Users/jaska/Desktop/2.jpg')
v_file=cv2.flip(img,0)
cv2.imwrite("C:/Users/jaska/Desktop/S01.jpg",v_file)
img1=cv2.imread('C:/Users/jaska/Desktop/3.jpg')
img_gray=cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)     #灰度图转换
rects=detector(img_gray, 0)         #识别出面部
for i in range(len(rects)):     
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])   #对每张脸进行标注
    for idx, point in enumerate(landmarks):     
        pos = (point[0, 0], point[0, 1])
        print("Face"+str(i)+":"+str(pos))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx + 1), pos, font, 0.2, (255, 0, 0), 1, cv2.LINE_AA)

cv2.namedWindow("img1")
cv2.imshow("img1",img1)
cv2.waitKey(0)


