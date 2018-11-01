#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 18-8-28 下午9:07
# @Author  : zhuzhengyi

'''
去除一张图片的水印logo，需要有与图片中水印logo相同大小的水印图片。
'''

import cv2

testimgpath = './sourceimg.jpeg'
watermarkpath = './watermark.jpeg'

testimg = cv2.imread(testimgpath)
wmimg = cv2.imread(watermarkpath)

sift = cv2.SIFT()
kp1,des1 = sift.detectAndCompute(testimg,None)
kp2,des2 = sift.detectAndCompute(wmimg,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

for m,n in matches:
	if m.distance < 0.75*n.distance:
		x1 = kp1[m.queryIdx].pt[0]
		y1 = kp1[m.queryIdx].pt[1]
		x2 = kp2[m.trainIdx].pt[0]
		y2 = kp2[m.trainIdx].pt[1]



