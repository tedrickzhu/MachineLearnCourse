#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 18-8-28 下午9:50
# @Author  : zhuzhengyi
# @File    : rmWm.py
# @Software: PyCharm
'''
去除水印
'''

import cv2
import numpy

#基于inpaint方法
def baseInpaint(test_dir,mask_dir,save_dir1,save_dir2):
	src = cv2.imread(test_dir)  # 默认的彩色图(IMREAD_COLOR)方式读入原始图像
	mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)  # 灰度图(IMREAD_GRAYSCALE)方式读入水印蒙版图像

	# 参数：目标修复图像; 蒙版图（定位修复区域）; 选取邻域半径; 修复算法(包括INPAINT_TELEA/INPAINT_NS， 前者算法效果较好)
	dst1 = cv2.inpaint(src, mask, 3, cv2.INPAINT_TELEA)
	cv2.imwrite(save_dir1, dst1)

	dst2 = cv2.inpaint(src, mask, 3, cv2.INPAINT_NS)
	cv2.imwrite(save_dir2, dst2)

#将黑底水印图变为白底水印图
def changeBackColor(blackmask):
	for row in range(blackmask.shape[0]):
		for column in range(blackmask.shape[1]):
			for channel in range(blackmask.shape[2]):
				if blackmask[row,column,channel]==0:
					blackmask[row, column, channel]=255
				else:
					blackmask[row, column, channel]=255-blackmask[row,column,channel]
	cv2.imwrite('./whitemask.jpg',blackmask)

#基于像素的反色中和
#参考自ps去水印原理，通过一张白底的反色水印图来中和原图水印
def rmWm2(test_dir,mask_dir,save_dir):
	src = cv2.imread(test_dir)
	mask = cv2.imread(mask_dir)
	print('src size:',src.shape)
	print('mask size:',mask.shape)
	# changeBackColor(mask)
	save = numpy.zeros(src.shape, numpy.uint8)
	# cv2.imwrite('./empty.jpg',save)
	for row in range(src.shape[0]):
		for col in range(src.shape[1]):
			for channel in range(src.shape[2]):
				if mask[row, col, channel] == 0:
					val = 0
				else:
					reverse_val = 255 - src[row, col, channel]
					val = 255 - reverse_val * 256 / mask[row, col, channel]
					if val < 0:
						val = 0
				save[row, col, channel] = val
	cv2.imwrite(save_dir, save)


if __name__=='__main__':
	test_dir = './sourceimg2.jpg'
	black_mask_dir = './mask2.png'
	white_mask_dir = './mask2white.png'
	result1_1 = './result1_1.jpg'
	result1_2 = './result1_2.jpg'
	result2 = './result2_bywhite.jpg'

	# baseInpaint(test_dir,mask_dir,result1_1,result1_2)
	rmWm2(test_dir,white_mask_dir,result2)