# -*- coding: utf-8 -*-
'''
SVM的应用
线性可分情况下 简单例子
'''
import numpy as np
from sklearn import svm
X = [[2,0],[1,1],[3,6],[5,6],[5.1,5.9],[7,8]]
X = np.array(X)
y = [0,0,0,1,1,1] #label，SVM中为-1，1；这里用0，1表示
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X,y)
print X
print '分类器信息：',classifier

print classifier.support_vectors_

#得到支持向量在X中的index
print classifier.support_

#得到支持向量的数量
print classifier.n_support_

#分类
#print classifier.predict([9,2])

from random import *
for i in range(10):
    a = [uniform(0,10),uniform(0,10)]
    print '分类结果：',classifier.predict(a)

