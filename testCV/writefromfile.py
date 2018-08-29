# encoding=utf-8

import cv2 as cv
capture = cv.VideoCapture("/home/zzy/Videos/1person/1person_wrong.avi")

nbFrames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv.CAP_PROP_FPS)
codec = capture.get(cv.CAP_PROP_FOURCC)

wait = int(1/fps * 1000/1) #Compute the time to wait between each frame query

duration = (nbFrames * fps) / 1000 #Compute duration

print 'Num. Frames = ', nbFrames
print 'Frame Rate = ', fps, 'fps'

fourcc = cv.VideoWriter_fourcc(*'XVID')
# fourcc = cv.VideoWriter_fourcc(*'MPEG')
#writer=cv.VideoWriter("/home/zzy/Videos/writefromfileresize.avi", fourcc, int(fps), (640,480)) #Create writer with same parameters
writer=cv.VideoWriter("/home/zzy/Videos/cutstandard222.avi", fourcc, int(fps), (width,height)) #Create writer with same parameters

#capture.set(cv.CAP_PROP_POS_FRAMES,80) #Set the number of frames

for f in xrange( nbFrames ): #Just recorded the 80 first frames of the video

    ret,frame = capture.read()
    height,width = frame.shape[:2]

    res2 = cv.resize(frame, (width / 5, height / 5), interpolation=cv.INTER_AREA)

    #print capture.get(cv.CAP_PROP_POS_FRAMES)
    if f >35:
        writer.write(frame)

writer.release()
capture.release()