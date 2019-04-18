# 100天机器学习
## day1 数据预处理
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#100-Days-Of-ML-Code day1 数据预处理

#第一步 导入库
import numpy as np
import pandas as pd

#第二步 导入数据集
dataset = pd.read_csv('Data.csv')      #读取csv文件
X = dataset.iloc[:, :-1].values        #.iloc[行，列]
Y = dataset.iloc[:, 3].values          # : 全部行 or 列；[a]第a行 or 列

#第三步 处理丢失数据
from sklearn.preprocessing import Imputer   
# sklearn包括分类、回归、降维、聚类四大机器学习算法
imputer = Imputer(missing_values = "NaN",strategy = "mean",axis = 0)    
# missing_values：缺失值，可以为整数或NaN(缺失值numpy.nan用字符串‘NaN’表示)，默认为NaN
# 训练模型，拟合出作为替换值的均值
# 创建Imputer器,sklearn处理器Imputer,只接受DataFrame类型，且DataFrame中必须全部为数值属性，需要单独取出
imputer = imputer.fit(X[ : ,1:3])
X[ : ,1:3] = imputer.transform(X[ : ,1:3])

# 第四步 解析分类数据
# 分类数据是具有标签值而不是数值的变量，取值范围通常固定（类似枚举值），不能用于模型的数学计算，需要解析成数字。
#sklearn.preprocessing.LabelEncoder():标准化标签，将标签值统一转换成range（标签值个数-1）范围内
from sklearn.preprocessing import LabelEncoder
# 对X中的标签数据编码
labelencoder_X = LabelEncoder()   
X[:, 0] = labelencoder_X.fit_transform(X[:,0])
# 创建虚拟变量
# 使用onehotencoder 对经过标签编码的第0行数据进行独热编码
onehotencoder = OneHotEncoder(categorical_features = [0])   
X = onehotencoder.fit_transform(X).toarray()
# 对 Y 中的标签数据编码
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# 第五步 拆分数据集为训练集合和测试集合
from  sklearn.model_selection import  train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# 第六步 特征量化
from  sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```
