#encoding=utf-8
#author:Tedrick
#software:Pycharm
#file:data_utils.py
#time:2019/4/12 下午9:38

from os import listdir
import os
from cv2 import imread
from cv2 import imwrite
from cv2 import resize
import scipy.io as sio
import numpy as np

def MiniSizeImage_catdog(dataset,matfile):
    filelist = listdir(dataset)
    countcat = 0
    countdog = 0
    classnums = 2
    samples = 50
    datacell = np.empty((classnums,samples),dtype=object)

    for filename in filelist:
        if countdog==samples and countcat==samples:
            break
        if filename[0]=='c':
            if countcat<samples:
                origin_image_gray = imread(dataset+filename,0)
                minicat = resize(origin_image_gray,(32,32))
                normalizecat = minicat/255.0
                datacell[0,countcat]=normalizecat
                countcat+=1
            else:
                continue
        if filename[0]=='d':
            if countdog<samples:
                origin_image_gray = imread(dataset+filename,0)
                minidog = resize(origin_image_gray,(32,32))
                normalizedog = minidog/255.0
                datacell[1,countdog]=normalizedog
                countdog+=1
            else:
                continue
    print(datacell.shape, type(datacell), type(datacell[0, 49]),type(datacell[0,0]),type(datacell[1,0]),type(datacell[1,49]))
    sio.savemat(matfile, {'catdog': datacell})

def MiniSizeImage_face(face1,face2,matfile):
    face1list = listdir(face1)
    face2list = listdir(face2)
    countcat = 0
    countdog = 0
    classnums = 2
    samples = min(len(face1list),len(face2list))
    datacell = np.empty((classnums,samples),dtype=object)

    for filename in face1list:
        if countcat==samples:
            break
        if countcat < samples:
            origin_image_gray = imread(face1 + filename, 0)
            minicat = resize(origin_image_gray, (32, 32))
            normalizecat = minicat / 255.0
            datacell[0, countcat] = normalizecat
            countcat += 1
        else:
            break
    for filename in face2list:
        if countdog==samples:
            break
        if countdog<samples:
            origin_image_gray = imread(face2+filename,0)
            minidog = resize(origin_image_gray,(32,32))
            normalizedog = minidog/255.0
            datacell[1,countdog]=normalizedog
            countdog+=1
        else:
            break
    #模拟错误标记训练数据
    # origin_image_gray = imread(face1 + filename, 0)
    # minicat = resize(origin_image_gray, (32, 32))
    # normalizecat = minicat / 255.0
    # datacell[0, countcat] = normalizecat
    #
    # origin_image_gray = imread(face2 + filename, 0)
    # minidog = resize(origin_image_gray, (32, 32))
    # normalizedog = minidog / 255.0
    # datacell[1, countdog] = normalizedog

    print(datacell.shape, type(datacell), type(datacell[0, 9]),type(datacell[0,0]),type(datacell[1,9]))
    sio.savemat(matfile, {'face': datacell})


def MiniSizeImage_nface(face,n,matfile):

    classnums = n
    samples = 10
    datacell = np.empty((classnums,samples),dtype=object)
    for classi in range(classnums):
        dataseti = face+'/s'+str(classi+1)+'/'
        faceilist = listdir(dataseti)
        counti = 0
        for filename in faceilist:
            if counti < samples:
                print(dataseti + filename)
                origin_image_gray = imread(dataseti + filename, 0)
                # print(dataseti + filename, origin_image_gray.shape)
                minicat = resize(origin_image_gray, (32, 32))
                normalizecat = minicat / 255.0
                datacell[classi, counti] = normalizecat
                counti += 1
            else:
                break

    print(datacell.shape, type(datacell), type(datacell[0, 9]),type(datacell[0,0]),type(datacell[1,9]))
    sio.savemat(matfile, {'face': datacell})

if __name__ == '__main__':
    # MiniSizeImage_catdog(dataset = './train/',matfile='catdog2x50.mat')
    # face1 = './att_faces/s1/'
    # face2 = './att_faces/s2/'
    # MiniSizeImage_face(face1,face2,matfile='faces2x10.mat')
    MiniSizeImage_nface(face= './att_faces/',n=10,matfile='faces10x10.mat')

###下面是讲解python怎么读取.mat文件以及怎么处理得到的结果###
# load_fn = 'xxx.mat'
# load_data = sio.loadmat(load_fn)
# load_matrix = load_data['matrix'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
# load_matrix_row = load_matrix[0] #取了当时matlab中matrix的第一行，python中数组行排列

###下面是讲解python怎么保存.mat文件供matlab程序使用###
# save_fn = 'xxx.mat'
# save_array = np.array([1,2,3,4])
# sio.savemat(save_fn, {'array': save_array}) #和上面的一样，存在了array变量的第一行

# save_array_x = np.array([1,2,3,4])
# save_array_y = np.array([5,6,7,8])
# sio.savemat(save_fn, {'array_x': save_array_x, 'array_y': save_array_y}) #同理，只是存入了两个不同的变量供

# npose = 5
# nsmile = 2
# poseSmile_cell = np.empty((npose,nsmile),dtype=object)
# for i in range(5):
#     for k in range(2):
#         poseSmile_cell[i,k] = np.zeros((4,4))
# # sio.savemat(save_fn, {'testcell':poseSmile_cell}) #同理，只是存入了两个不同的变量供
# print poseSmile_cell.shape
# print poseSmile_cell[0,0]

