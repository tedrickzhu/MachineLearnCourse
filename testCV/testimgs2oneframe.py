import argparse
import glob

import cv2
import numpy as np
import os


def show_in_one(images, show_size=(480, 1280), blank_size=0, window_name="merge"):
    small_h, small_w = images[0].shape[:2]
    column = int(show_size[1] / (small_w + blank_size))
    row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    print images[0]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])
    print shape
    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    if count < max_count:
        print("ingnore count %s" % (max_count - count))
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, merge_img)
    cv2.waitKey(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate mouse interaction with images')
    parser.add_argument("-i", "--input", help="Input directory.")
    args = parser.parse_args()
    path = args.input
    if path is None:
        test_dir = "/home/zzy/Pictures/testMergeImg"
        path = test_dir

    debug_images = []

    vc1 = cv2.VideoCapture('/home/zzy/Videos/utiltestvideo/test1.avi')
    vc2 = cv2.VideoCapture('/home/zzy/Videos/1person/1person_speed1.avi')

    c = 1
    print 'step 1 open video file'
    if vc1.isOpened():
        rval1, frame1 = vc1.read()
        debug_images.append(frame1)
    else:
        rval1 = False

    if vc2.isOpened():
        rval2, frame2 = vc2.read()
        debug_images.append(frame2)
    else:
        rval2 = False

    timeF = 1
    print "output images-------------------------"
    while  rval2:
        rval1, frame1 = vc1.read()
        rval2, frame2 = vc2.read()
        debug_images[0]=frame1
        debug_images[1]=frame2
        if (c % timeF == 0):
            show_in_one(debug_images)
            # cv2.imshow(frame)
            #cv2.imwrite('/home/zzy/Pictures/output/test' + str(c).zfill(6) + '.jpg', frame)
        c = c + 1

    vc1.release()
    vc2.release()
    # for infile in glob.glob(os.path.join(path, '*.*')):
    #     ext = os.path.splitext(infile)[1][1:]  # get the filename extenstion
    #     if ext == "png" or ext == "jpg" or ext == "bmp" or ext == "tiff" or ext == "pbm":
    #         print(infile)
    #         img = cv2.imread(infile)
    #         #print "output source img"
    #         #cv2.imshow("single",img)
    #         #cv2.waitKey(1000)
    #         if img is None:
    #             continue
    #         else:
    #             debug_images.append(img)

    #show_in_one(debug_images)
    #cv2.waitKey(0)
    cv2.destroyWindow()