# 100天机器学习
## day3 多元线性回归
```python
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# 100-Days-Of-ML-Code day3 多元线性回归

# 第一步 数据预处理
# 导入库
import pandas as pd
import numpy as np

# 导入数据集
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values
# print(X)
# print(X[:10])  # 输出X的0行到9行
# print(Y)

# 将类别数据数字化
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
# print("labelencoder: \n",X[:10])
# print(X[:10])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
print("onehot:")
print(X[:10])

# 躲避虚拟变量陷阱
# ================================================================================
# 数据集中有非数值数据类型时，可以转换为虚拟变量表示，通常取0or1
# “虚拟变量陷阱”的实质是：完全多重共线性。
# 解决方法：在引入虚拟变量时要求如果有m个定性变量，在模型中引入m-1个虚拟变量。
# ================================================================================
X = X[:, 1:]    # 去掉第0列
print("躲避虚拟变量陷阱后的X= \n",X[:10])

# 拆分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.2, random_state = 0)

# 第二步：在训练集上训练多元线性回归模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# 第三步： 在测试集上预测结果
y_pred = regressor.predict(X_test)
# print("y_pred= \n",y_pred)
```
