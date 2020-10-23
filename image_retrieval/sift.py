# from sys import flags
# import cv2
# import sys
# import numpy as np

# # imgpath = sys.argv[1]
# imgpath = r'image_retrieval\1602490762(1).jpg'
# img = cv2.imread(imgpath)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sift = cv2.xfeatures2d.SIFT_create()
# keypoints, descriptor = sift.detectAndCompute(gray,None)

# img = cv2.drawKeypoints(image = img, outImage = img, keypoints = keypoints,
#                         flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINT,
#                         color = (51,163,236))
# cv2.imshow('sift_keypoints', img)
# while(True):
#     if cv2.waitKey(int(1000/12)) & 0xff == ord("q"):
#         break
# cv2.destroyAllWindows()


import cv2
import numpy as np
# from psd_tools import PSDImage

# 灰度图读入图片
img1 = cv2.imread("image_retrieval\img5.jpg",cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("image_retrieval\img6.jpg",cv2.IMREAD_GRAYSCALE)

#sift特征计算
sift = cv2.xfeatures2d.SIFT_create()

img1_kp1, img1_des1 = sift.detectAndCompute(img1, None)
img1_kp2, img1_des2 = sift.detectAndCompute(img2, None)

#flann特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(img1_des1,img1_des2,k=2)
goodMatch = []
for m,n in matches:
    if m.distance < 0.50*n.distance:
        goodMatch.append(m)
#增加一个维度
goodMatch = np.expand_dims(goodMatch,1)
print(goodMatch[:20])
img_out = cv2.drawMatchesKnn(img1,img1_kp1,img2,img1_kp2,goodMatch,None,flags=2)

cv2.imshow("img_out",img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

