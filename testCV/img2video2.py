# encoding=utf-8

# import os
# import cv2
#
# #图片路径，将一个文件夹里面的所有图片按顺序处理成视频
# im_dir = '/home/zzy/Pictures/trash/testimgs'
# #输出视频路径
# video_dir = '/home/zzy/Videos/testimg2video.avi'
# #帧率
# fps = 30
#
# #图片尺寸
# img_size = (960,544)
# files = os.listdir(im_dir)
# #图片数
# num = len(files)
#
# #fourcc = cv2.cv.CV_FOURCC('M','J','P','G')#opencv2.4
# fourcc = cv2.VideoWriter_fourcc(*'XVID') #opencv3.0
# videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
# #im_name = os.path.join(im_dir, 'frames'+str(i).zfill(6)+'.jpg')
# for i in range(1,num):
#     #im_name = os.path.join(im_dir, 'frames'+str(i)+'.jpg')
#     im_name = im_dir+'/frames'+str(i)+'.jpg'
#     frame = cv2.imread(im_name)
#     print frame.shape
#     videoWriter.write(frame)
#     print im_name
#
# videoWriter.release()
# print 'finish'

import cv2
import os

dir = '/home/zzy/Pictures/trash/testimgs'
files = os.listdir(dir)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out_path = './test.avi'
print 'this is video path:',video_out_path
out = cv2.VideoWriter(video_out_path, fourcc, 30, (int(960), int(544)))
for i in range(len(files)):
        # if file.endswith('jpg'):
        print file
        img = cv2.imread(dir+'/frames'+str(i)+'.jpg')
        img = cv2.flip(img,1)
        print img.shape
        out.write(img)
out.release()

print('end')

