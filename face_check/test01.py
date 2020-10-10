import cv2


def detect():
    # 声明face_cascade变量，该变量为CascadeClassifier对象，负责人脸检测
    face_cascade = cv2.CascadeClassifier(
        r'face_check\haarcascade_frontalface_default.xml')
    # 人的眼睛检测
    eye_cascade = cv2.CascadeClassifier(r'face_check\haarcascade_eye.xml')
    img = cv2.imread(  # 读取图片
        r'face_check/22.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转化为灰度图
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 进行实际的人脸检测
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, 0, (10, 10))  # 进行眼睛检测
    for (x, y, w, h) in faces:          #画出人脸矩阵
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    for (ex, ey, ew, eh) in eyes:       # 画出眼睛矩阵
        img = cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    cv2.imshow('Vikongs Detected!!', img)
    cv2.imwrite(r'face_check/aa.jpg', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    detect()
