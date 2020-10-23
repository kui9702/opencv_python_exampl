#检测角点特征
import cv2
import numpy as np

img = cv2.imread(r'image_retrieval\1602490762(1).jpg',1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 23, 0.04)   #用于检测图像的哈里斯角点检测，判断是否能检测出一点是不是图像的角点，第三个参数定义了sobel算子的中孔，是检点检测的敏感度，取值介于3和31之间的奇数
img[dst > 0.01 * dst.max()] = [0, 0, 255]   #将检测到的奇点标记为红色
while(True):
    cv2.imshow('corners', img)
    if cv2.waitKey(int(1000/12)) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
