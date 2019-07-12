# day23决策树
1、它是一种监督学习算法，主要用于分类问题，适用于可分类的、连续的输入和输出变量。本质上它是从一层层if/else问题中进行学习并得出结论的。  
2、决策树在逻辑上以树的形式存在，包含根节点、内部节点和叶节点。  
* 根节点：包含数据集中所有数据的集合  
* 内部节点：每个内部节点为一个判断条件，并且包含数据集中满足从根节点到该节点所有条件的数据集合。根据内部结点的判断条件测试结果，内部结点对应数据的集合分别分到两个或者多个子节点中。  
* 叶节点： 叶节点为最终的类别，被包含在该叶节点的数据属于该类别。     

3、决策树学习过程  
* 特征选择：从训练数据的特征中选择一个特征作为当前节点的分裂标准（特征选择的标准不同产生了不同的特征决策树算法）。
* 决策树生成：根据所选特征评估标准，从上至下递归地生成子节点，直到数据集不可分则停止决策树停止声场。
* 剪枝：决策树容易过拟合，需要剪枝来缩小树的结构和规模（包括预剪枝和后剪枝）。   

4、[决策树算法之ID3](https://blog.csdn.net/acdreamers/article/details/44661149)  
* 信息熵：离散随机事件出现的概率，一个系统越有序其信息熵就越低，反之就越高。   
* 信息增益是针对某个特征而言的，看该系统有无此特征时的信息量各是多少，两者之间的差值就是该特征给系统带来的信息量，即信息增益。  
* 该算法的核心思想就是以信息增益来度量属性的选择，选择分裂后信息增益最大的属性进行分裂。该算法采用自顶向下的贪婪搜索遍历可能的决策空间。    

5、代码如下

```python
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# day25 决策树分类

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
print("0:\n", X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=0)
print("1:\n", X_train)
# 特征缩放: 将数据减去平均值并除以方差
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("2:\n", X_train)

# 对测试集进行决策树分类拟合
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train, y_train)

# 预测测试集结果
y_pred = classifier.predict(X_test)

# 制作混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# 将训练结果进行可视化
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# 将测试结果可视化
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
DecisionTreeClassifier()

```
![Train set](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/25.Decision%20Tree%20Classification%20(Training%20set).png)
![Test set](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/25.Decision%20Tree%20Classification%20(Test%20set).png)
