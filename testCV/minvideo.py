# encoding=utf-8

import cv2 as cv
import numpy

capture = cv.VideoCapture("/home/zzy/Videos/multipose.mp4")
#capture = cv.VideoCapture(0)

#outvideo = cv.wr

# nbFrames = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
# width = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH))
# height = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))
# fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
# codec = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FOURCC)

nbFrames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

#nbFrames = capture.get(cv.CV_CAP_PROP_FRAME_COUNT)
width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv.CAP_PROP_FPS)
codec = capture.get(cv.CAP_PROP_FOURCC)

print nbFrames,width,height,fps,codec

wait = int(1/fps * 1000/1) #Compute the time to wait between each frame query
duration = (nbFrames * fps) / 1000 #Compute duration

fourcc = cv.VideoWriter_fourcc(*'XVID')
out1 = cv.VideoWriter('/home/zzy/Videos/resizevideo1.avi',fourcc, int(fps), (width,height))
out2 = cv.VideoWriter('/home/zzy/Videos/resizevideo2.avi', fourcc, int(fps), (width/5,height/5))
out3 = cv.VideoWriter('/home/zzy/Videos/resizevideo3.avi', fourcc, int(fps), (850,480))


while(capture.isOpened()):
    ret, frame = capture.read()

    if ret==True:
        cv.imshow("videotest",frame)
        out1.write(frame)

        h0, w0 = frame.shape[:2]
        h2,w2 =int( h0 / 5),int(w0 / 5)
        res2 = cv.resize(frame, (w0 / 5 , h0 / 5), interpolation=cv.INTER_AREA)
        out2.write(res2)

        cv.imshow("small",res2)
        h, w = res2.shape[:2]
        print "this is the size of res2"
        print h, w

        res3 = cv.resize(res2,(w*5,h*5),interpolation=cv.INTER_CUBIC)
        out3.write(res3)

        cv.imshow("big",res3)
        h3, w3 = res3.shape[:2]
        print "this is the size of res3"
        print h3,w3

        cv.waitKey(wait)

    else:
        break

out1.release()
out2.release()
out3.release()
capture.release()
cv.destroyAllWindows()

