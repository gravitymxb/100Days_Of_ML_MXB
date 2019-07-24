# DAY 46 numpy
1、聚合   
* np.sum(L) 对L求和 np.min(L)求最小， np.max 求最大 三者速度比sum、min、max快   
* 总统身高的例子  
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('height.csv')  # 导数据集
heights = np.array(data['height(cm)'])  # 将数据转成数组形式

print("Mean height:\n", heights.mean())  # 输出身高平均值
print("Standard deviation:\n", heights.std())  # 输出身高标准差
print("Minimum height:\n", heights.min())    
print("Maximum height:\n", heights.max())   
print("25th percentile:\n", np.percentile(heights, 25))  # 输出占比25%的数值
print("Median:\n ", np.median(heights))  # 中位数
print("75th percentile:\n", np.percentile(heights, 75))

plt.hist(heights)  # 直方图
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number');
plt.show()
```
图示：
![图示]()

2、广播(Broadcast)是numpy对不同形状(shape)的数组进行数值计算的方式，对数组的算术运算通常在相应的元素上进行。  
NumPy中广播的规则：  
规则1：如果两个数组的维数不同，则尺寸较少的数组的形状与其前导(左侧)的形状相同。   
规则2：如果两个数组的形状在任何维度中不匹配，则在该维度中形状等于1的数组将被拉伸以与另一个形状匹配。   
规则3：如果在任何维度中，大小不一致，且两者都不等于1，则会引发错误。   

例：
```python
M = np.ones((2, 3))
a = np.arange(3)
M.shape = (2, 3)  
a.shape = (3,)  
```
由规则1变为                     由规则2    
```python
M.shape -> (2, 3)       =>     M.shape -> (2, 3)  
a.shape -> (1, 3)       =>     a.shape -> (2, 3)   
```  
3、比价、掩码、布尔运算   
比较：  

operator  | equivalent ufunc  | operator | equivalent ufunc 
-------  | ------  | ----- | --- 
==  | np.equal | !=  | np.not_equal 
<  | np.less | <= | np.less_equal
>  |  np.greater  |  >=  |  np.greater_equal  

布尔运算以及掩码操作：   
```python
x = np.array[1, 2, 3, 4, 5] 
print(x<3)   # 作为运算符 输出为[True True False False False]
print(x[x<3])   # 作为掩码使用  输出为[1 2]
```
