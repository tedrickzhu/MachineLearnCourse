# encoding=utf-8

import cv2
import numpy as np

#img = cv2.imread('/home/zzy/Pictures/001.jpg')

def resizebig(img):
    (height, width) = img.shape[:2]

    print (height, width)
    #cv2.imshow("img", img)
    #res1 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    res1 = cv2.resize(img, (width * 5, height * 5), interpolation=cv2.INTER_CUBIC)

    # cv2.imshow("res1", res1)
    (h,w)=res1.shape[:2]
    print (h,w)

    return res1

def resizesmall(img):

    height, width = img.shape[:2]
    res2 = cv2.resize(img, (width / 5, height / 5), interpolation=cv2.INTER_AREA)
    # cv2.imshow('res2', res2)
    (h, w) = res2.shape[:2]
    print (h, w)
    # cv2.waitKey(0)


# res3 = cv2.resize(res2, (w * 5, h * 5), interpolation=cv2.INTER_CUBIC)
# cv2.imshow("res3", res3)
# (h3, w3) = res3.shape[:2]
# print (h3, w3)
#
# cv2.waitKey(0)




