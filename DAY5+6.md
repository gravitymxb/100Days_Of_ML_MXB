# 100天机器学习
## day5+6 逻辑回归

```python
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# 100-Days-Of-ML-Code day5、6  逻辑回归实现

# 一、数据预处理
# 1、导入库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 导入数据集
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values
# print(X)
# print(Y)
# 将数据集分成训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.25, random_state = 0)
# 特征缩放
# 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = X_train.astype(np.float64)  # 指定数据类型，避免警告
X_test = X_test.astype(np.float64)
X_train = sc.fit_transform(X_train)    # fit_transform()先拟合数据，再标准化
X_test = sc.transform(X_test)    # 直接用transform()数据标准化，与前者用用同一个fit出来的模型

# 二、逻辑回归模型
# 该项工作的库将会是一个线性模型库，之所以被称为线性是因为逻辑回归是一个线性分类器,
# 这意味着我们在二维空间中，我们两类用户（购买和不购买）将被一条直线分割。
# 然后导入逻辑回归类。下一步我们将创建该类的对象，它将作为我们训练集的分类器。
# 见day4记录

# 将逻辑回归应用于训练集
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 三、预测  预测测试集结果
y_pred = classifier.predict(X_test)

# 四、评估预测
# 我们预测了测试集,现在我们将评估逻辑回归模型是否正确的学习和理解。
# 因此这个混淆矩阵将包含我们模型的正确和错误的预测。

#生成混淆矩阵
from sklearn.metrics import  confusion_matrix   # 混淆矩阵见DAY4记录
cm = confusion_matrix(y_test, y_pred)
# 可视化
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train
X1,X2=np. meshgrid(np. arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                   np. arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np. unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c = ListedColormap(('red', 'green'))(i), label=j)

plt. title(' LOGISTIC(Training set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt. legend()
plt. show()

X_set,y_set=X_test,y_test
X1,X2=np. meshgrid(np. arange(start=X_set[:,0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                   np. arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np. unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c = ListedColormap(('red', 'green'))(i), label=j)

plt. title(' LOGISTIC(Test set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt. legend()
plt. show()
```
![LOGISTIC(Training set).png](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/LOGISTIC(Training%20set).png)  

![LOGISTIC(Test set).png](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/LOGISTIC(Test%20set).png)

