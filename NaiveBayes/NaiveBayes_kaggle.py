#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 18-8-26 下午4:32
# @Author  : zhuzhengyi
# @File    : NaiveBayes_kaggle.py
# @Software: PyCharm

'''
conference:https://blog.csdn.net/fisherming/article/details/79509025
Kaggle Competition: San Francisco Crime Classification
https://www.kaggle.com/c/sf-crime/data
'''

from sklearn.naive_bayes import BernoulliNB
import time

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
train = pd.read_csv('/train.csv', parse_dates = ['Dates'])
test = pd.read_csv('/test.csv', parse_dates = ['Dates'])


#对犯罪类别:Category; 用LabelEncoder进行编号  
leCrime = preprocessing.LabelEncoder()
crime = leCrime.fit_transform(train.Category)   #39种犯罪类型  
#用get_dummies因子化星期几、街区、小时等特征  
days=pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour)
#组合特征  
trainData = pd.concat([hour, days, district], axis = 1) #将特征进行横向组合  
trainData['crime'] = crime  #追加'crime'列 
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = test.Dates.dt.hour
hour = pd.get_dummies(hour)
testData = pd.concat([hour, days, district], axis=1)


features=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
          'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION','NORTHERN', 'PARK', 'RICHMOND',
          'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
X_train, X_test, y_train, y_test = train_test_split(trainData[features], trainData['crime'], train_size=0.6)
NB = BernoulliNB()
nbStart = time.time()
NB.fit(X_train, y_train)
nbCostTime = time.time() - nbStart
#print(X_test.shape)  
propa = NB.predict_proba(X_test)    #X_test为263415*17； 那么该行就是将263415分到39种犯罪类型中，每个样本被分到每一种的概率  
print("朴素贝叶斯建模%.2f秒"%(nbCostTime))
predicted = np.array(propa)
logLoss=log_loss(y_test, predicted)
print("朴素贝叶斯的log损失为:%.6f"%logLoss)



