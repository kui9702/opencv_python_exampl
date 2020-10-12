import cv2
import os

def detect(face_list_path,face_list_name):
    # 声明face_cascade变量，该变量为CascadeClassifier对象，负责人脸检测
    face_cascade = cv2.CascadeClassifier(
        r'face_check\haarcascade_frontalface_default.xml')
    img = cv2.imread(  # 读取图片
        face_list_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转化为灰度图
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 进行实际的人脸检测
    for (x, y, w, h) in faces:          #画出人脸矩阵
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        f = cv2.resize(gray[y:y+h, x:x+2],(200,200))
        cv2.imwrite(os.path.join(r'face_check_test/pgm',face_list_name+'.pgm'),f)
    cv2.imwrite(os.path.join(r'face_check_test/finish_img',face_list_name+'.jpg'), img)


if __name__ == "__main__":
    people_face_path = 'face_check_test\people_face'    #人脸数据路径
    face_lists = os.listdir(people_face_path)        #所有人脸文件
    print(face_lists)
    print(len(face_lists))
    for face_list in face_lists:
        face_list_name = face_list.split('.')[0]
        face_list_path = os.path.join(people_face_path,face_list)
        detect(face_list_path,face_list_name)
    # detect()

