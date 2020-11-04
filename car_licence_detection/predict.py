"""
预处理
预测未完待续...
"""

import paddle
import cv2
import skimage.measure as measure
import numpy as np

licence_plate=cv2.imread('./car_licence_test.png')

# h,w,c=licence_plate.shape
# print(h,w,c,licence_plate.shape)
# cv2.imshow('car licence',licence_plate)
# cv2.waitKey(0)

#预处理
#灰度化
gray_plate=cv2.cvtColor(licence_plate,cv2.COLOR_RGB2GRAY)
# cv2.imshow('gary',gray_plate)
# cv2.waitKey(0)

#二值化
ret,binary_plate=cv2.threshold(gray_plate,175,255,cv2.THRESH_BINARY)
# print(binary_plate.shape)
# cv2.imshow('binary',binary_plate)
# print('thresold value',ret)
# cv2.waitKey(0)

labels,labels_num=measure.label(binary_plate,return_num=True,connectivity=2)
print(labels.shape,labels_num)
# print(np.unique(labels))

dst=np.zeros_like(licence_plate,dtype='uint8')

# print((labels==1).shape)
for label in np.unique(labels):
    if label==0:
        continue

    # [b,g,r]=np.random.randint(0,256,size=(3))
    # dst[:,:,0][labels==label],dst[:,:,1][labels==label],dst[:,:,2][labels==label]=[b,g,r]
    dst[:,:,0][labels==label],dst[:,:,1][labels==label],dst[:,:,2][labels==label]=np.random.randint(0,256,size=(3))

    cv2.imshow('dst',dst)
    cv2.waitKey(1000)
    # break

# cv2.imshow('dst',dst)
cv2.waitKey(0)

