#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 18-8-26 下午4:26
# @Author  : zhuzhengyi
# @File    : NaiveBayes_sklearn.py
# @Software: PyCharm

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import datasets

iris = datasets.load_iris()
gnb = GaussianNB()
scores = cross_val_score(gnb,iris.data,iris.target,cv=10)
print("accuracy:%.3f"%scores.mean())

