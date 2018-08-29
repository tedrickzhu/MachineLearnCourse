import cv2 as cv

# capture = cv.VideoCapture('/home/zzy/Videos/trueupcutfore30f.mp4')
capture = cv.VideoCapture('/home/zzy/Videos/dancewithmusic/wrong2.mp4')

#通过opencv 获得视频的相关信息，如图片的宽高，帧率，编码格式，共计多少帧
nbFrames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv.CAP_PROP_FPS)
codec = capture.get(cv.CAP_PROP_FOURCC)
#test = capture.get(cv.)
# In OpenCV 3.2, drop the CV in front of the flag. This should work just fine
#
# videoCapture = cv2.VideoCapture(file_path)
# fps = videoCapture.get(cv2.CAP_PROP_FPS)
# size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
#         int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

#CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream
#CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream

wait = int(1/fps * 1000/1)

duration = (nbFrames * fps) / 1000

time = nbFrames / fps

print width,height
print 'Num. Frames = ', nbFrames
print 'Frame Rate = ', fps, 'fps'
print 'Duration = ', duration, 'sec'
print "time = " , time , "s"

fourcc = cv.VideoWriter_fourcc(*'XVID')
out1 = cv.VideoWriter('/home/zzy/Videos/something1.avi', fourcc, fps, (width, height))
out2 = cv.VideoWriter('/home/zzy/Videos/something2.avi', fourcc, fps, (width, height))
out3 = cv.VideoWriter('/home/zzy/Videos/something3.avi', fourcc, fps, (width, height))
out4 = cv.VideoWriter('/home/zzy/Videos/wrong2cutforeuf.mp4', int(codec), fps, (width, height))

for f in xrange(nbFrames):
    ret,frameImg = capture.read()
    if f>68:
        # cv.imwrite("/home/zzy/Pictures/test754.jpg",frameImg)
    # cv.putText(frameImg, "frames:%f" % (f), (20, 20), 0, 0.5, (0, 0, 255))
        out4.write(frameImg)
        #out1.write(frameImg)
    # if f>1396 and f<2175:
    #     out2.write(frameImg)
    # if f >2176:
    #     out3.write(frameImg)
    #print capture.get(cv.CAP_PROP_POS_FRAMES)
    cv.imshow("The Video", frameImg)
    cv.waitKey(1)