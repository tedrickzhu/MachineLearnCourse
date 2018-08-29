# encoding=utf-8

import os
import cv2

#图片路径，将一个文件夹里面的所有图片按顺序处理成视频
#图片名称为固定格式，如，test+序号，序号为固定位数（如4位）的数字，如编号为1的则为0001，编号为33的则为0033
im_dir = '/home/zzy/Pictures/output/'
#输出视频路径
video_dir = '/home/zzy/Videos/testimg2video.avi'
#帧率
fps = 30

#图片尺寸
img_size = (640,480)
files = os.listdir(im_dir)
#图片数
num = len(files)

#fourcc = cv2.cv.CV_FOURCC('M','J','P','G')#opencv2.4
fourcc = cv2.VideoWriter_fourcc(*'XVID') #opencv3.0
videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for i in range(1,num):
    im_name = os.path.join(im_dir, 'test'+str(i).zfill(4)+'.jpg')
    frame = cv2.imread(im_name)
    videoWriter.write(frame)
    print(im_name)

videoWriter.release()
print('finish')
