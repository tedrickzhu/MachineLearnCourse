#encoding = utf-8
__author__='zzy'
__date__ ='2018-08-08'

from tensorflow.contrib.keras.api.keras.layers import Input,Conv2D,MaxPool2D,Dense,Flatten
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard
from tensorflow.contrib.keras.api.keras.utils import to_categorical

from cifar_data_load import getTrainDataByLabel,getTestDataByLabel

#build LeNet
def lenet(input):
	#conv1
	conv1 = Conv2D(6,5,(1,1),'valid',use_bias=True)(input)
	#maxpooling1
	maxpool1 = MaxPool2D((2,2),(2,2),'valid')(conv1)
	#conv2
	conv2 = Conv2D(6, 5, (1, 1), 'valid', use_bias=True)(maxpool1)
	#maxpool2
	maxpool2 = MaxPool2D((2, 2), (2, 2), 'valid')(conv2)
	#conv3
	conv3 = Conv2D(16, 5, (1, 1), 'valid', use_bias=True)(maxpool2)
	#flatten
	flatten = Flatten()(conv3)
	#fully connection dense1
	dense1 = Dense(120,)(flatten)
	dense2 = Dense(84,)(dense1)
	dense3 = Dense(10,activation='softmax')(dense2)

	return dense3

def train():
	#input
	myinput = Input([32,32,3])
	#构建网络
	output = lenet(myinput)
	#建立模型
	model = Model(myinput,output)

	#定义优化器，使用Adam优化器，learning rate = 0.0003
	adam = Adam(lr=0.003)
	#compile model
	model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

	#get input data
	X = getTrainDataByLabel('data')
	# 获取图像的label，这里使用to_categorical函数返回one-hot之后的label
	Y = to_categorical(getTrainDataByLabel('labels'))

	#start training model batchsize=200,50 epoches
	model.fit(X,Y,200,10,1,callbacks=[TensorBoard('./log',write_images=1,histogram_freq=1)],
	          validation_split=0.2,shuffle=True)

	#save model
	model.save('lenet-no-activation-model.h5')

if __name__ == '__main__':
	train()


