# encoding=utf-8

import cv2
import numpy
import matplotlib.pyplot as plt
import reSizeImg as resimg

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('/home/zzy/Videos/resizevideo1.avi',fourcc, 20.0, (640,480))
out2 = cv2.VideoWriter('/home/zzy/Videos/resizevideo2.avi', fourcc, 20.0, (128,96))
out3 = cv2.VideoWriter('/home/zzy/Videos/resizevideo3.avi', fourcc, 20.0, (640,480))

while(1):
    ret, frame = cap.read()
    #print "this is ret"
    #print ret
    #print "this is frame"
    #print frame
    cv2.imshow("videotest",frame)
    out1.write(frame)
    #res1=resimg.resizesmall(frame)
    height, width = frame.shape[:2]
    res2 = cv2.resize(frame, (width / 5, height / 5), interpolation=cv2.INTER_AREA)
    out2.write(res2)
    cv2.imshow('res2', res2)
    h, w = res2.shape[:2]
    print h, w

    res3 = cv2.resize(res2,(w*5,h*5),interpolation=cv2.INTER_CUBIC)
    out3.write(res3)
    h3,w3 = res3.shape[:2]
    print h3,w3
    #res2=resimg.resizebig(frame)
    cv2.imshow("big",res3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

