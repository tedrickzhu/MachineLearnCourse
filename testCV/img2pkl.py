# encoding=utf-8

import cPickle
import os
import json
import pylab

import numpy
from PIL import Image

k = 0;
for filename1 in os.listdir(r'data'):
    print filename1
    filename1 = 'data/' + filename1
    # 或者通过这种方式来获取根目录
    # fileList = os.listdir(path)
    # allFile = []
    #     for fileName in fileList:
    #         filePath = os.path.join(path, fileName)
    #         if os.path.isdir(filePath):
    #             dirList(filePath)
    #         allFile.append(filePath)
    for filename2 in os.listdir(filename1):
        print filename2
        filename2 = filename1 + '/' +filename2
        for filename3 in os.listdir(filename2):
            print filename3
            if(filename3!='Thumbs.db'):
              k = k + 1
# k是用来调节加载所有图片次数的
k = k * 10
print k



i = 0;
j=0;
label =0;
olivettifaces=numpy.empty((k,28*28))
olivettifaces_label=numpy.empty(k)
# j是用来调节加载所有图片次数的 。
while(j<10):
    for filename1 in os.listdir(r'data'):
        print filename1
        filename1 = 'data/' + filename1
        for filename2 in os.listdir(filename1):
            print filename2
            label = label + 1
            if(label == 7):
                label = 1
            filename2 = filename1 + '/' +filename2
            for filename3 in os.listdir(filename2):
                print filename3
                if(filename3!='Thumbs.db'):
                    filename3 = filename2 + '/' +filename3
                    #矩阵->数组->向量。
                    imgage = Image.open(filename3)
                    imgage = imgage.resize((28,28))
                    img_ndarray = numpy.asarray(imgage, dtype='float64')/256
                    olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)

                    olivettifaces_label[i]=label-1

                    i = i + 1
    j = j+1

print i




olivettifaces_label=olivettifaces_label.astype(numpy.int)

write_file=open('olivettifaces.pkl','wb')
cPickle.dump([[olivettifaces[0:28000],olivettifaces_label[0:28000]],
              [olivettifaces[10001:13000],olivettifaces_label[10001:13000]],
              [olivettifaces[13001:16499],olivettifaces_label[13001:16499]]],write_file,-1)
write_file.close()


