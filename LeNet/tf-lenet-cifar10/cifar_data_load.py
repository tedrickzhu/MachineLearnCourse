#encoding = utf-8
__author__='zzy'
__date__ ='2018-08-08'

cifar10path = '/home/zzy/TrainData/cifar-10-batches-py/'

import numpy as np
import pickle

def unpickle(filepath):
	with open(filepath,'rb')as fileio:
		filedict = pickle.load(fileio,encoding='bytes')
	return filedict

def convert2image(raw):
	#raw_float = float(raw)
	#print(type(raw_float))
	image = np.reshape(raw, [3, 32, 32])
	image = image.transpose(1, 2, 0)

	# r = raw[0:1024]
	# r = np.reshape(r, [32, 32, 1])
	# g = raw[1024:2048]
	# g = np.reshape(g, [32, 32, 1])
	# b = raw[2048:3072]
	# b = np.reshape(b, [32, 32, 1])
	#
	# photo = np.concatenate([r, g, b], -1)

	return image

#load all datafiles to one ndarray data once
#according to different label ,return different data about cifar10 dataset
def getTrainDataByLabel(label):
	batch_label = []
	labels = []
	data = []
	filenames = []
	#load five batch file once
	for i in range(1,1+5):

		batch_label.append(unpickle(cifar10path+"data_batch_%d" % i)[b'batch_label'])
		labels += unpickle(cifar10path+"data_batch_%d" % i)[b'labels']
		data.append(unpickle(cifar10path+"data_batch_%d" % i)[b'data'])
		filenames += unpickle(cifar10path+"data_batch_%d" % i)[b'filenames']
	#conbine data list to ndarray
	data = np.concatenate(data,0)
	label = str.encode(label)

	if label == b'data':
		array = np.ndarray([len(data),32,32,3],dtype=np.int32)
		for i in range(len(data)):
			array[i]=convert2image(data[i])
		return array
	elif label == b'labels':
		return labels
		pass
	elif label == b'batch_label':
		return batch_label
		pass
	elif label == b'filenames':
		return filenames
		pass
	else:
		raise NameError


def getTestDataByLabel(label):
	batch_label = []
	filenames = []

	file = unpickle(cifar10path+"test_batch")
	batch_label.append(file[b'batch_label'])
	labels = file[b'labels']
	data = file[b'data']
	filenames += file[b'filenames']

	label = str.encode(label)
	if label == b'data':
	    array = np.ndarray([len(data), 32, 32, 3], dtype=np.int32)
	    for i in range(len(data)):
	        array[i] = convert2image(data[i])
	    return array
	    pass
	elif label == b'labels':
	    return labels
	    pass
	elif label == b'batch_label':
	    return batch_label
	    pass
	elif label == b'filenames':
	    return filenames
	    pass
	else:
	    raise NameError