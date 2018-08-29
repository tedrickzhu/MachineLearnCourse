#encoding=utf-8

from math import log
import operator
import csv
import json

def calcShannonEnt(dataSet):  # 计算数据的熵(entropy)
    numEntries=len(dataSet)  # 数据条数
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1] # 每行数据的最后一个字（类别）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1  # 统计有多少个类以及每个类的数量
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries # 计算单个类的熵值
        shannonEnt-=prob*log(prob,2) # 累加每个类的熵值
    return shannonEnt

def createDataSet1():    # 创造示例数据
    dataSet = [['long', 'low', 'boy'],
               ['short', 'low', 'boy'],
               ['short', 'low', 'boy'],
               ['long', 'high', 'girl'],
               ['short', 'high', 'girl'],
               ['short', 'low', 'girl'],
               ['long', 'low', 'girl'],
               ['long', 'low', 'girl']]
    labels = ['hair', 'voice']  # 两个特征
    return dataSet,labels

def read_dataset(path):
    dataset = []
    keys = None
    gene_featers = []
    with open(path, 'U') as csvfile:
        reader = csv.DictReader(csvfile)
        keys = reader.fieldnames
        for row in reader:
            gene_featers.append(row[keys[2]])
    # print(keys)
    # print(gene_featers)
    for key_i in range(3, len(keys)):
        key = keys[key_i]
        column = []
        with open(path, 'U') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                column.append(row[key])
            column.append(key[-1])
            dataset.append(column)
    # print(len(dataset))
    # print(dataset[0])
    return dataset,gene_featers

def splitDataSet(dataSet,axis,value): # 按某个特征分类后的数据,分为两类，一类大于当前特征值，一类小于当前特征值，然后计算这两个类别的信息熵及信息增益。
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec =featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):  # 选择最优的分类特征
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet)  # 按特征分类后的熵
        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
        if (infoGain>bestInfoGain):   # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature

def chooseBestFeatureToSplit_v2(dataSet):  # 选择最优的分类特征
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = None
    bestFeature = -1
    bestFeature_value = 0
    bestFeature_bigsubdata = None
    bestFeature_smallsubdata = None
    for i in range(numFeatures):
        #拿到当前特征的所有的样本值
        featSampleList = [example[i] for example in dataSet]
        uniqueVals = set(featSampleList)
        newEntropy = 0
        bestvalue = None
        #当前特征对应的最佳分类时的数据集的信息熵
        bestvalue_entroy = None
        bestvalue_subdata = None

        #逐个处理当前特征的所有样本值，计算取该值为分割点时的信息增益
        for value in uniqueVals:
            smalldataset,biggerdataset = splitDataSet_v2(dataSet,i,value)
            p_small = len(smalldataset)/float(len(dataSet))
            p_big = len(biggerdataset)/float(len(dataSet))
            smalldataentroy = calcShannonEnt(smalldataset)
            bigdataentroy = calcShannonEnt(biggerdataset)
            #用切分后的两个子集的平均信息熵来表示此值对应的切分效果
            value_entroy = p_small*smalldataentroy+p_big*bigdataentroy
            if bestvalue_entroy is None or bestvalue_entroy>value_entroy:
                bestvalue_entroy = value_entroy
                bestvalue = value
                bestvalue_subdata = [smalldataset,biggerdataset]

        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
        if bestInfoGain is None or (infoGain>bestInfoGain):   # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain=infoGain
            bestFeature = i
            bestFeature_value = bestvalue
            bestFeature_smallsubdata = bestvalue_subdata[0]
            bestFeature_bigsubdata = bestvalue_subdata[1]
    print('this is best features:',bestFeature,bestFeature_value)
    return bestFeature,bestFeature_value,bestFeature_smallsubdata,bestFeature_bigsubdata

def splitDataSet_v2(dataSet,axis,value): # 按某个特征分类后的数据,分为两类，一类大于当前特征值，一类小于当前特征值，然后计算这两个类别的信息熵及信息增益。
    smallDataSet=[]
    biggerDataSet = []
    for sampleVec in dataSet:
        reducedSampleVec = sampleVec[:axis]
        reducedSampleVec.extend(sampleVec[axis + 1:])
        if sampleVec[axis]<value:
            smallDataSet.append(reducedSampleVec)
        if sampleVec[axis]>value:
            biggerDataSet.append(reducedSampleVec)
    return smallDataSet,biggerDataSet



def majorityCnt(classList):    #按分类后类别数量排序，比如：最后分类为2男1女，则判定为男；
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]  # 类别：L或者M

    if classList.count(classList[0])==len(dataSet):#只有一类
        return classList[0]
    if len(dataSet[0])==1:#所有特征均已使用
        return majorityCnt(classList)
    bestFeat,bestFeat_value,smalldataset, bigdataset=chooseBestFeatureToSplit_v2(dataSet) #选择最优特征
    bestFeatLabel=labels[bestFeat]

    # myTree = {bestFeatLabel: [bestFeat_value,{'>':{},'<':{}}]}  # 分类结果以字典形式保存
    myTree={} #分类结果以字典形式保存
    # smalldataset, bigdataset = splitDataSet_v2(dataSet, bestFeat, bestFeat_value)
    #切分完数据后更新labels
    del(labels[bestFeat])
    bigsubtree = {}
    smallsubtree={}
    if len(bigdataset)>0:
        bigsubtree = createTree(bigdataset,labels)
    if len(smalldataset):
        smallsubtree = createTree(smalldataset,labels)
    root_value = [bestFeat_value,{'big':bigsubtree,'small':smallsubtree}]
    myTree[bestFeatLabel]=root_value
    # myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

def classficatesample(sample,trees,labels):
    rootfeat = list(trees.keys())[0]
    print('this is root feature:',str(rootfeat))
    rootfeat_value = trees[rootfeat][0]
    smallsubtree = trees[rootfeat][1]['small']
    bigsubtree = trees[rootfeat][1]['big']
    featIndex = labels.index(rootfeat)
    sample_value = sample[featIndex]
    sample_result = None
    if sample_value < rootfeat_value:
        print('进入小子树')
        # print(type(smallsubtree).__name__ )
        if type(smallsubtree).__name__ == 'unicode':
            sample_result = smallsubtree
            print(smallsubtree)
        if type(smallsubtree).__name__ == 'dict':
            if not smallsubtree:
                if type(bigsubtree).__name__ == 'unicode':
                    if bigsubtree == 'L':
                        sample_result = 'M'
                    if bigsubtree == 'M':
                        sample_result = 'L'
            else:
                sample_result = classficatesample(sample, smallsubtree, labels)
    else:
        print('进入大子树')
        # print(type(bigsubtree).__name__ )
        if type(bigsubtree).__name__ == 'unicode':
            sample_result = bigsubtree
            print(bigsubtree)
        if type(bigsubtree).__name__ == 'dict':
            if not bigsubtree:
                if type(smallsubtree).__name__ == 'unicode':
                    if smallsubtree == 'L':
                        sample_result = 'M'
                    if smallsubtree == 'M':
                        sample_result = 'L'
            else:
                sample_result = classficatesample(sample, bigsubtree, labels)

    return sample_result

def classficate(testdata,trees,labels):
    print(type(trees.keys()))
    print(trees.keys())
    results = []
    for i in range(len(testdata)):
        sample = testdata[i]
        sample_res = classficatesample(sample,trees,labels)
        results.append(sample_res)
    return results

def cal_accuracy(testdata,results):
    cal_res = []
    true_res = []
    rate = 0
    if len(testdata) == len(results):
        right_nums = 0
        for i in range(len(testdata)):
            right = testdata[i][-1]
            true_res.append(right)
            if right == results[i]:
                cal_res.append(1)
                right_nums += 1
            else:
                cal_res.append(0)
        rate = right_nums/float(len(testdata))
        return true_res,cal_res,rate
    else:
        return -1,-1,0

if __name__=='__main__':
    # dataSet, labels=createDataSet1()  # 创造示列数据
    trainpath = './train.csv'
    testpath = './test.csv'
    dataSet, labels = read_dataset(trainpath)
    # mytree = createTree(dataSet, labels)
    # print(mytree)
    # with open("./mytree.json", "w") as f:
    #     json.dump(mytree, f)
    # print("载入文件完成...")
    testdata, testlabels = read_dataset(testpath)
    with open('./mytree.json','r') as treejson:
        mytree = json.load(treejson)
    classlables = classficate(dataSet,mytree,labels)
    print(classlables)
    truth,cal_res,rate = cal_accuracy(dataSet,classlables)
    print(truth)
    print(cal_res)
    print(rate)

