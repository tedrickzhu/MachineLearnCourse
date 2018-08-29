#encoding=utf-8

import numpy as np
import math
import json
import csv

#定义数据文件读取函数
def LoadFile(filename):
    data=sio.loadmat(filename)
    data=data[filename[0:-4]]
    return data

def read_dataset(path):
    dataset = []
    keys = None
    labels = []
    with open(path, 'U') as csvfile:
        reader = csv.DictReader(csvfile)
        keys = reader.fieldnames

    # print(keys)
    # print(gene_featers)
    for key_i in range(3, len(keys)):
        key = keys[key_i]
        column = []
        if key[-1] == 'L':
            labels.append(1)
        if key[-1] == 'M':
            labels.append(0)
        with open(path, 'U') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                column.append(row[key])

            dataset.append(column)
    # print(len(dataset))
    # print(dataset[0])
    return dataset,labels

#定义Sigmoid函数
def get(x):
    act_vec=[]
    for i in x:
        act_vec.append(1/(1+np.exp(-i)))
    act_vec=np.array(act_vec)
    return act_vec

def TrainNetwork(sample,label,epoch,input_learnrate,hid_learnrate):
    sample_num = len(sample)
    sample_len = len(sample[0])
    sample = np.float64(sample)
    out_num = 2
    hid_num = 10
    w1 =np.float64( 2 * np.random.random((sample_len, hid_num)) - 1)
    w2 =np.float64( 2 * np.random.random((hid_num, out_num)) - 1)
    hid_offset = np.zeros(hid_num)
    out_offset = np.zeros(out_num)

    loss = None
    for epoch_i in range(epoch):
        print('epoch:   ',str(epoch_i))
        for i in range(0,len(sample)):
            t_label=np.zeros(out_num)
            t_label[label[i]]=1
            # t_label = label[i]
            #前向的过程
            hid_value=np.dot(sample[i],w1)+hid_offset #隐层的输入
            hid_act=get(hid_value)                 #隐层对应的输出
            out_value=np.dot(hid_act,w2)+out_offset
            out_act=get(out_value)    #输出层最后的输出
            print(out_act)
            #后向过程
            err=t_label-out_act
            print('loss error:  ', str(err))
            errsqur = err[0]**2+err[1]**2
            if loss is not None and (loss-errsqur<-0.05):
                print('loss error:  ', str(err))
                break
            else:
                loss = errsqur
            print('loss error squre  :  ',str(err**2))
            out_delta=err*out_act*(1-out_act) #输出层的方向梯度方向
            hid_delta = hid_act*(1 - hid_act) * np.dot(w2, out_delta)
            #权值更新
            for j in range(0,out_num):
                w2[:,j]+=hid_learnrate*out_delta[j]*hid_act
            for k in range(0,hid_num):
                w1[:,k]+=input_learnrate*hid_delta[k]*sample[i]
            # 各层的偏移量的更新
            out_offset += hid_learnrate * out_delta
            hid_offset += input_learnrate * hid_delta

    return w1,w2,hid_offset,out_offset

#测试过程
def test(test_data,test_labels,bpnnparam):
    # train_sample=LoadFile('mnist_train.mat')
    # train_sample=train_sample/256.0
    # train_label=LoadFile('mnist_train_labels.mat')
    # test_sample=LoadFile('mnist_test.mat')
    # test_sample=test_sample/256.0
    # test_label=LoadFile('mnist_test_labels.mat')
    test_data = np.float64(test_data)
    compare_labels = []
    right = np.zeros(10)
    # numbers = np.zeros(10)
    # for i in test_labels:
    #     numbers[i]+=1
    # print(numbers)
    for i in range(0,len(test_labels)):
        hid_value=np.dot(test_data[i],bpnnparam['w1'])+bpnnparam['hid_offset']
        hid_act=get(hid_value)
        out_value=np.dot(hid_act,bpnnparam['w2'])+bpnnparam['out_offset']
        out_act=get(out_value)
        compare_labels.append([test_labels[i],out_act])
        if np.argmax(out_act) == test_labels[i]:
          right[test_labels[i]] += 1

    # print(compare_labels)
    # print(right.sum()/ len(test_labels))
    return compare_labels

def main():

    train_path = './train.csv'
    test_path = './test.csv'

    train_data,train_labels = read_dataset(train_path)
    epoch = 5
    input_learnrate = 0.1
    hid_learnrate = 0.1
    w1,w2,hid_offset,out_offset=TrainNetwork(train_data,train_labels,epoch,input_learnrate,hid_learnrate)
    result = {'w1':w1,'w2':w2,'hid_offset':hid_offset,'out_offset':out_offset}
    print (type(result))
    # print('训练所得参数',result)l
    # with open('./BPNNparameters.json','w') as file:
    #     json.dump(result,file)
    bpnnparam = result

    # with open('./BPNNparameters.json','r') as file:
    #     bpnnparam =json.load(file)

    test_data,test_labels = read_dataset(test_path)
    accurocy = test(test_data,test_labels,bpnnparam)
    print('正确率',str(accurocy))


if __name__ == '__main__':
    main()