# DAY47 深入研究numpy
## 2.7 花哨的索引
* 概念： 它意味着传递一个索引数组来同时访问多个数组元素。
```[x[3], x[7], x[2]]```  =  ``` ind = [3, 7, 4]; x[ind]```
* 使用花式索引时，结果的形状反映索引数组的形状，而不是索引数组的形状
例如：
```python
x=[51 92 14 71 60 20 82 86 74 74]
ind = np.array([[3, 7],
                [4, 5]])
print(X[ind])     
#结果为 ：
[[71 86]
 [60 20]]
```  
* 花式索引  
```python
X = np.arange(12).reshape((3, 4))
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
print(X[row, col])
# 输出为：  [ 2  5 11]   

print(X[row[:, np.newaxis], col]) 
# 输出为：
 [[ 2  1  3]
 [ 6  5  7]
 [10  9 11]]
```
* 选择随机点的例子
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn

rand = np.random.RandomState(42)
mean = [0, 0]
cov = [[1, 2],
       [2, 5]]

# np.random.multivariate_normal:生成一个多元正态分布矩阵
# 参数：mean是多维分布的均值维度；cov是协方差矩阵，size为生成矩阵的维度
X = rand.multivariate_normal(mean, cov, 100) 

seaborn.set() 
plt.scatter(X[:, 0], X[:, 1])
# 从100个数中选择20个，replace为False即表示进行不放回的操作
indices = np.random.choice(X.shape[0], 20, replace=False) 
selection = X[indices]

plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1],
            facecolor='red', s=200)
plt.show()
```
![47.1]()
![47.2]()


## 2.8 数组的排序

## 2.9 结构化数据
