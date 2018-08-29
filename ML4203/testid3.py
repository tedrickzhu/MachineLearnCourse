#encoding=utf-8

from  math import log
import csv

def shannonent(dic):
    mm = {}
    dicLen= len(dic)
    shannonent = 0.0
    for elm in dic:
        mm[elm]=mm.get(elm,0)+1
    for key in mm:
        prob = float(mm[key])/dicLen
        shannonent -= prob*log(prob,2)
    return shannonent


def splitdataSet(udata,xais,value):
    reDataSet=[]
    for line in udata:
        if line[xais]==value:
            newdata=line[:xais]
            newdata.extend(line[xais+1:])
            reDataSet.append(newdata)
    return reDataSet


def choiceBestW(dataSet):
    labels = [ elm[-1] for elm in dataSet]
    origshannon = shannonent(labels)
    kuan = len(dataSet[1])-1
    chang  = len(dataSet)
    bestInforGain = 0.0
    bestFeature = -1
    for i in range(kuan):
        featList = [elm[i] for elm in dataSet]
        uniVals = set(featList)
        newEntropy = 0.0
        for word in uniVals:
            newer = splitdataSet(dataSet,i,word)
            prob = len(newer)/len(dataSet)
            newEntropy +=prob*shannonent(newer)
        infoGain = origshannon - newEntorpy
        if (infoGain > bestInforGain):
            bestInforGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classlist):
    classCount = {}
    for vote in classlist:
        classCount[vote]=classCount.get(vote,0)+1
    sortedClass = sorted(classCount.iteritems,\
                         key=operator.itemgetter(1),reverse=Ture)
    return sortedClass[0][0]


def creatmytree(oridata,lables):
    diclist = [elm[-1] for elm in oridata]
    if diclist.count(diclist[0])==len(diclist):
        return diclist[0]
    if len(oridata[0])==1:
        return majorityCnt(diclist)
    bestFeat = choiceBestW(oridata)
    bestFeatLabel = lables[bestFeat]
    mytree ={bestFeatLabel:{}}
    del(lables[bestFeat])
    bestline = [elm[bestFeat] for elm in oridata]
    uniqvals = set(bestline)
    for val in uniqvals:
        nlabels = lables[:]
        mytree[bestFeatLabel][val] = creatmytree(splitdataSet(oridata,bestFeat,val),nlabels)
    return mytree

def classficate(testdata,trees,labels):
    firstStr = trees.keys()[0]
    secondDict = trees[firstStr]
    featIndex = featLabels.index(firstStr)
    for val in secondDict.keys():
        if testdata[featIndex] == val:
            if type(sencondDict[val]).__name__ =='dict':
                classLabel = classficate(testdata,secondDict[keys],labels)
            else:
                classLabel = secondDict[key]
    return classLabel

def read_dataset(path):
    dataset = []
    keys = None
    gene_featers = []
    with open(path, 'U') as csvfile:
        reader = csv.DictReader(csvfile)
        keys = reader.fieldnames
        for row in reader:
            gene_featers.append(row[keys[2]])
    for key_i in range(3, len(keys)):
        key = keys[key_i]
        column = []
        with open(path, 'U') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                column.append(row[key])
            column.append(key[-1])
            dataset.append(column)
    return dataset,gene_featers

if __name__=='__main__':
    # dataSet, labels=createDataSet1()  # 创造示列数据
    trainpath = './train.csv'
    testpath = './test.csv'
    dataSet, labels = read_dataset(trainpath)
    testdata, testlabels = read_dataset(testpath)
    mytree = creatmytree(dataSet, labels)
    print(mytree)
    classlables = classficate(testdata,mytree,labels)
    print(classlables)
    with open('./result2.txt','w') as resultfile:
        resultfile.write(mytree)

