# -*- coding: utf-8 -*-
from sklearn import svm
import numpy as np
import pylab as pl

np.random.seed(0)
a = np.random.randn(20,2) - [2,2] #产生一个20行2列的随机数，且满足标准正态分布(0,1)，均值为0，标准差为1. 然后将均值改为-2,标准差不变
b = np.random.randn(20,2) + [2,2] #与上面类似
X = np.r_[a,b] #按行来组合两个矩阵
Y = [0] * 20 + [1] * 20 #标签
#print a
#print X,X.shape
#print Y

classifier = svm.SVC(kernel = 'linear')
classifier.fit(X,Y)
print '支持向量：',classifier.support_vectors_

#得到支持向量在X中的index
print '支持向量在样本中的位置：',classifier.support_

#得到支持向量的数量
print '支持向量的个数：',classifier.n_support_


w = classifier.coef_[0] #coefficient:系数
k = -w[0] / w[1] #斜率   w0x + w1y + b = 0,则 y = -(w0/w1)x - (b/w1)
b = classifier.intercept_[0]
x = np.linspace(-5,5)
y = k * x - b / w[1]  #最大边缘超平面(在这里是线)

#计算左右两个边界上的线的方程
point = classifier.support_vectors_[0] #左边的一个点
l = point[1] - k * point[0] #截距
y_left = k * x + l

point = classifier.support_vectors_[-1]
l = point[1] - k * point[0] #截距
y_right = k * x + l


pl.plot(x,y,'k-') #第三个参数:实线
pl.plot(x,y_left,'k--') #第三个参数：虚线
pl.plot(x,y_right,'k--') #第三个参数：虚线


pl.scatter(classifier.support_vectors_[:,0],classifier.support_vectors_[:,1],s=80,facecolors = 'yellow')
pl.scatter(X[:,0],X[:,1],c = Y, cmap = pl.cm.Paired)
#pl.axis('tight')

from random import *
for i in range(10):
    a = [uniform(-5,5),uniform(-5,5)]
    #print a
    print '分类结果：',classifier.predict(a)
    pl.scatter(a[0],a[1],s = 180,facecolors = 'black')

pl.axis('tight')
pl.show()