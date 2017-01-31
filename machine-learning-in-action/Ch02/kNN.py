#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 导入科学计算包Numpy和运算符模块
from numpy import *
import operator

# createDataSet()函数，自定义创建数据集和标签
def createDataSet():
	# 定义样本训练集
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # 定义数据标签
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# inX：用于分类的输入向量。即将对其进行分类。
# dataSet：训练样本集
# labels:标签向量 p

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 以下三行距离计算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        #以下两行选择距离最小的k的点
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # 排序
        sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)        # 得到文件行数
    returnMat = zeros((numberOfLines, 3))   # 创建返回的Numpy矩阵
    classLabelVector = []
    index = 0
    # (以下三行)解析文件数据到列表
    for line in arrayOlines:
        line = line.strip()                 # 截取掉所有的回车字符
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

