# DAY43 k—均值聚类
1、无监督学习： 没有已知标签的训练集，只给一堆数据集，通过学习去发现数据内在的性质及规律。   
2、聚类算法： 用于将簇群或者数据点分隔成一系列的组，使得相同簇中的数据点比其他组更相似。基本上，目的是分隔具有相似性状的组，并分配到簇中。   
3、k-均值聚类： 该算法中将所有项分为k个簇，使得相同簇中的所有项彼此尽量相似，而不同簇中的所有项尽量不同。每个族中有一个形心，可将其其理解为最能代表簇的点。  
动画演示：  
![图](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/43_k-means.gif)   
4、K—均值聚类算法步骤：  
    （1）随机取k个样本作为初始均值向量（或者采用别的方式获取初始均值向量）    
    （2）根据每个样本与均值向量的距离来判断各个样本所属的蔟。  
    （3）根据分好的蔟再次计算新的均值向量，根据新的均值向量再对每个样本进行划分。  
    （4）循环步骤2，3，直到分类结果相同或者在我们规定的误差范围内时中止。   
5、k—均值聚类目标是使总体群内方差最小。
代码如下：
```python
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# day34 k-均值聚类
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # 导入K-均值聚类函数
# coding=utf-8


# 读取网页中的数据表
table = []
for i in range(1, 7):
    table.append(pd.read_html('https://nba.hupu.com/stats/players/pts/%d' % i)[0])  # 获取网页数据

# 所有数据纵向合并为数据框
players = pd.concat(table)
players.drop(0, inplace=True)  # 删除行标签为0的记录，因为，换完页，行标签为0时，没有数据

X = players.iloc[1:, 9].values  # 自变量为罚球命中率
Y = players.iloc[1:, 5].values  # 因变量为命中率

# 将带百分号的字符型转化为float型
x = []
for i in X:
     x.append(float(i.strip('%')))  # 去掉百分号
x = np.array(x)/100
# print(x)

y = []
for j in Y:
     y.append(float(j.strip('%')))
y = np.array(y)/100
# print(y)

# 合并成矩阵
n = np.array([x.ravel(), y.ravel()]).T
# print(n)

# 绘制原始数据散点图
plt.style.use('ggplot')  # 设置绘图风格
plt.scatter(n[:, 0], n[:, 1])  # 画散点图
plt.xlabel('free throw hit rate')
plt.ylabel('hit rate')
plt.show()

# 选择最佳的K值
X = n[:]
K = range(1, int(np.sqrt(n.shape[0])))  # 确定K值的范围
GSSE = []
for k in K:  # 统计不同簇数下的平方误差
    SSE = []
    kmeans = KMeans(n_clusters=k, random_state=10)  # 构造聚类器
    kmeans.fit(X)  # 聚类
    labels = kmeans.labels_  # 获取聚类标签

    centers = kmeans.cluster_centers_  # 获取每个簇的形心
    for label in set(labels):  # set创建不重复集合
# 不同簇内的数据减去该簇内的形心
        SSE.append(np.sum((np.array(n[labels == label, ])-np.array(centers[label, :]))**2))
# 总的误差
    GSSE.append(np.sum(SSE))

# 绘制K的个数与GSSE的关系
plt.plot(K, GSSE, 'b*-')
plt.xlabel('K')
plt.ylabel('Error')
plt.title('optimal solution')
plt.show()

#调用sklearn的库函数
num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=1)
kmeans.fit(X)

# 聚类中心
centers = kmeans.cluster_centers_

# 绘制簇散点图
plt.scatter(x=X[:, 0], y=X[:, 1], c=kmeans.labels_)
# 绘制形心散点图
plt.scatter(centers[:, 0], centers[:, 1], c='k', marker='*')
plt.xlabel('free throw hit rate')
plt.ylabel('hit rate')
plt.show()
```
