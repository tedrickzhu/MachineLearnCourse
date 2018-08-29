"""
Example script using PyOpenPose.
"""
#encoding = utf-8

import PyOpenPose as OP
import time
import cv2

import numpy as np
import os

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]


def showHeatmaps(hm):
    for idx, h in enumerate(hm):
        cv2.imshow("HeatMap "+str(idx), h)


def showPAFs(PAFs, startIdx=0, endIdx=16):
    allpafs = []
    for idx in range(startIdx, endIdx):
        X = PAFs[idx*2]
        Y = PAFs[idx*2+1]
        tmp = np.dstack((X, Y, np.zeros_like(X)))
        allpafs.append(tmp)

    pafs = np.mean(allpafs, axis=0)
    cv2.imshow("PAF", pafs)


def run():

    cap = cv2.VideoCapture("/home/zzy/Videos/testdanceposeniuyao.avi")
    #cap = cv2.VideoCapture(0)

    nbFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    download_heatmaps = False
    with_face = with_hands = False
    op = OP.OpenPose((320, 240), (240, 240), (width, height), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,download_heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)
    #op = OP.OpenPose((320, 240), (240, 240), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0, download_heatmaps)

    actual_fps = 0
    paused = False
    delay = {True: 0, False: 1}

    #output the file
    f1 = open('/home/zzy/Videos/personpoints20.txt', 'w+')
    #f2 = open('/home/zzy/Videos/dancer2.txt', 'w+')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter('/home/zzy/Videos/testyaobu20.avi', fourcc, fps, (width, height))
    print("Entering main Loop.")
    while True:
        start_time = time.time()
        h0 = 0
        w0 = 0
        try:
            ret, frame = cap.read()
            h0, w0 = frame.shape[:2]

            h2, w2 = int(h0 / 5), int(w0 / 5)
            res2 = cv2.resize(frame, (w0 / 5, h0 / 5), interpolation=cv2.INTER_AREA)

            rgb = frame
            print("RGB", rgb.shape)

        except Exception as e:
            print("Failed to grab", e)
            break

        t = time.time()
        op.detectPose(rgb)
        #op.detectFace(rgb)
        #op.detectHands(rgb)
        t = time.time() - t
        op_fps = 1.0 / t

        #f.write('UI FPS = %f, OP FPS = %f,time = %f' % (actual_fps, op_fps,t)+"\n")

        res = op.render(rgb)

        cv2.putText(res, 'UI FPS = %f, OP FPS = %f ,op time=%f  inputframe h0=%f,w0=%f' % (actual_fps, op_fps,t,h0,w0), (20, 20), 0, 0.5, (0, 0, 255))
        persons = op.getKeypoints(op.KeypointType.POSE)[0]

        if download_heatmaps:
            hm = op.getHeatmaps()
            print("HM ",hm.shape, hm.dtype)
            showHeatmaps(hm)

            # hm = op.getHeatmaps()
            # parts = hm[:18]
            # background = hm[18]
            # PAFs = hm[19:]  # each PAF has two channels (total 16 PAFs)
            # cv2.imshow("Right Wrist", parts[4])
            # cv2.imshow("background", background)
            # showPAFs(PAFs)

        if persons is not None and len(persons) > 0:
            # print "this is the first person:"
            # print persons[0]
            f1.write(str(persons[0])+"\n this is first person \n")
            # print "this is the second person:"
            # print persons[1]
            # f1.write(str(persons[1]) + "\n this is second person \n")
            posepoints = persons[0]
            #print posepoints
            #print "this is two points"
            #f1.write(str(posepoints[8][0:3])+"------"+str(posepoints[11][0:3])+"\n")
            target_weigu = (int((posepoints[8, 0] + posepoints[11, 0]) / 2), int((posepoints[8, 1] + posepoints[11, 1]) / 2))
            cv2.circle(res, target_weigu, 4, (255, 255, 255), 4)

            target_yaozhui =( int((posepoints[1, 0] * 22 + 78 * target_weigu[0]) / 100),
                              int((posepoints[1, 1] * 22 + 78 * target_weigu[1]) / 100))

            cv2.circle(res, target_yaozhui, 4, (0, 0, 255), 4)

            #print posepoints[7],posepoints[10]
            print("First Person: ", persons[0].shape)

        cv2.imshow("OpenPose result", res)
        out1.write(res)

        key = cv2.waitKey(delay[paused])
        if key & 255 == ord('p'):
            paused = not paused

        if key & 255 == ord('q'):
            break

        actual_fps = 1.0 / (time.time() - start_time)
        uit = time.time() - start_time

        f1.write('UI FPS = %f, time = %f,OP FPS = %f,time = %f,inputframe h0=%f,w0=%f' % (actual_fps,uit, op_fps, t,h0,w0) + "\n")

    # f0.close()
    f1.close()

if __name__ == '__main__':
    run()