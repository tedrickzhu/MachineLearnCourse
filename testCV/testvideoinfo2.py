import cv2 as cv

# capture = cv.VideoCapture('/home/zzy/Videos/trueupcutfore30f.mp4')
capture = cv.VideoCapture('/home/zzy/Videos/cutstandard000.avi')
# capture2 = cv.VideoCapture('/home/zzy/Videos/webvideos/withpose/20180203/2_mxnet_withpose_5point.mp4')
nbFrames1 = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
# nbFrames2 = int(capture2.get(cv.CAP_PROP_FRAME_COUNT))
width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv.CAP_PROP_FPS)
codec = capture.get(cv.CAP_PROP_FOURCC)


for f in range(nbFrames1):
    ret1,frameImg1 = capture.read()
    # out1.write(frameImg1)
    # ret2, frameImg2 = capture2.read()
    # out2.write(frameImg)
    cv.imwrite('/home/zzy/work/mxnet_Realtime_Multi-Person_Pose_Estimation/dance_pose_imgs/teacher_imgs/cutstandard000_'+str(f)+'.jpg',frameImg1)
    cv.imshow("The Video1", frameImg1)
    # cv.imshow("The Video2", frameImg2)
    #cv.waitKey(0)


#test = capture.get(cv.)
# In OpenCV 3.2, drop the CV in front of the flag. This should work just fine
#
# videoCapture = cv2.VideoCapture(file_path)
# fps = videoCapture.get(cv2.CAP_PROP_FPS)
# size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
#         int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

#CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream
#CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream

# wait = int(1/fps * 1000/1)
#
# duration = (nbFrames1 * fps) / 1000
#
# time = nbFrames1 / fps
#
# print width,height
# print 'Num. Frames = ', nbFrames1
# print 'Frame Rate = ', fps, 'fps'
# print 'Duration = ', duration, 'sec'
# print "time = " , time , "s"

# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out1 = cv.VideoWriter('/home/zzy/Videos/webvideos/withpose/20180203/1_mxnet_1frames.mp4', int(codec), fps, (width, height))
# out2 = cv.VideoWriter('/home/zzy/Videos/webvideos/withpose/20180203/2_mxnet_1frames.mp4', int(codec), fps, (width, height))


# t_begin = 500
# for i in range(t_begin):
#     capture.grab()
#     # capture2.grab()

#
# for f in xrange(nbFrames2):
#     ret,frameImg = capture2.read()
#     out2.write(frameImg)
#     # cv.imshow("The Video", frameImg)
#     cv.waitKey(500)

# for f in xrange(nbFrames1):
#     ret,frameImg = capture.read()
#     if f>68:
#         # cv.imwrite("/home/zzy/Pictures/test754.jpg",frameImg)
#     # cv.putText(frameImg, "frames:%f" % (f), (20, 20), 0, 0.5, (0, 0, 255))
#         out1.write(frameImg)
#         #out1.write(frameImg)
#     # if f>1396 and f<2175:
#     #     out2.write(frameImg)
#     # if f >2176:
#     #     out3.write(frameImg)
#     #print capture.get(cv.CAP_PROP_POS_FRAMES)
#     cv.imshow("The Video", frameImg)
#     cv.waitKey(1)



# if delta_t > 0:
#         #以教师的帧数排列为基准,教师均从0取，学生data从delta_t取,img从0取
#         for i in range(len(error_frames_infor_list)):
#             error_index = error_frames_infor_list[i][0]
#             error_range = error_frames_infor_list[i][1]
#             fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#             video_out_path = error_result_path +'/'+str(video_name)+'_'+str(i)+'.avi'
#             #print('this is video path:',video_out_path)
#             #print("the frames range: ",error_range)
#             error_videos_out = cv2.VideoWriter(video_out_path, fourcc, 30.0, (img_width*2, img_height))
#             #print 'this is the index of error '+str(error_index)
#             for frame_index in range(error_range[0],error_range[1]):
#                 error_frame = cv2.imread(student_source_imgs_path + '/' + str(video_name) +'_'+ str(
#                     frame_index+stu_start_index) + '.jpg')
#                 teacher_frame = cv2.imread(teacher_imgs_path + '/' + str(dance_id)+ '/' + str(dance_id) +'_'+ str(
#                     frame_index+abs(delta_t)) + '.jpg')
#                 if error_frame is not None and teacher_frame is not None:
#                     draw_person(error_frame, data_teacher[frame_index+abs(delta_t)], (0, 0, 255))
#                     draw_person(error_frame, data_student[frame_index+abs(delta_t)], (255, 255, 255))
#                     draw_person(teacher_frame, data_teacher[frame_index+abs(delta_t)], (0, 0, 255))
#                     merge_imgs[0] = error_frame
#                     merge_imgs[1] = teacher_frame
#                     merge_frames = show_in_one(merge_imgs,(img_height,img_width*2))
#                     #print("error_frame: ",error_frame.shape)
#                     error_videos_out.write(merge_frames)
#                     #cv2.waitKey(0)
#                     #print('write error videos ' + str(frame_index))
#                     if int(frame_index) == int(error_index):
#                         error_frame_out = error_result_path+'/'+str(video_name) +'_'+str(i)+'.jpg'
#                         cv2.imwrite(error_frame_out,merge_frames)
#                         error_frames_result.append(error_frame_out)
#                     #print('write error frames ' + str(error_index))
#                 else:
#                     print("frame is none: ",frame_index)
#             error_videos_result.append(video_out_path)
#     else:
#         #以学生的帧数排列为基准