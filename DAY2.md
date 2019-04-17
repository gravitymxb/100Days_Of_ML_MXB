# 100天机器学习
## day2 简单线形回归模型
```python
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# 100-Days-Of-ML-Code day2 简单线性回归模型

# 第一步数据预处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   # Matplotlib是一个Python 2D绘图库，它可以在各种平台上生成图形
# matplotlib.pyplot是一个命令型函数集合，pyplot中的每一个函数都会对画布图像作出相应的改变，
# 如创建画布、在画布中创建一个绘图区、在绘图区上画几条线、给图像添加文字说明等
# plt.plot([1, 2, 3, 4])
# plt.ylabel('some numebers')
# plt.show()                     试用matplotlib画图

dataset = pd.read_csv('studentscores.csv')
# 手动找到文件并路径，记书写路径方法：D:\python_pycharm\data\studentscores.csv:
# 上边为绝对路径，应改为相对路径（将csv文件与代码的py文件放到同一个文件夹下）使代码更简洁，不易出错

# print(dataset)
X = dataset.iloc[:, :1].values   #输出全部行，第0列的list   而且value:将表的形式转化为矩阵或者向量；iloc:提取时前闭后开
Y = dataset.iloc[:, 1].values    #输出全部行，第1列的list
# print(X)
# Y[0] = 22222
# print(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 1/4, random_state = 0)
# 将数据集分为1:3的测试集和训练集

# 第二步：训练集使用简单线性回归模型来训练
# 线性回归：线性回归是利用数理统计中回归分析，来确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# 第三步：预测结果
Y_pred = regressor.predict(X_test)
# print(Y_pred)

# 第四步：可视化
# 训练集结果可视化
plt.scatter(X_train, Y_train, color = 'red')
# print(X_train)
# print(Y_train)
#scatter 绘制散点图， X、Y需为相同长度的数组序列
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()
#测试集结果可视化
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test),color = 'blue')
plt.show()
```
## 运行结果  
### train_set_result: 
![](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/train_set_result.png)
### test_set_result:
![](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/test_set_result.png)

