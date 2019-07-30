# DAY 50
## 向量化字符串操作
* 字符串操作
```python
import numpy as np
x = np.array([2, 3, 5, 7, 11, 13])
x*=2
print(x)
输出
[2, 3, 5, 7, 11, 13]

import pandas as pd
data = ['peter', 'Paul', 'MARY', 'gUIDO']
data = [s.capitalize() for s in data] 
print(data)
输出：
['Peter', 'Paul', 'Mary', 'Guido']
names = pd.Series(data)
print（names）
输出：

0    peter
1     Paul
2     None
3     MARY
4    gUIDO
dtype: object
```
* python字符串方法表
```python
import pandas as pd

monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
print(monte.str.lower(), '\n')    # 输出为字符串
print(monte.str.len(), '\n')       # 输出字符串长度值（空格算一个字符）
print(monte.str.startswith('T'), '\n')  # 返回布尔值
print(monte.str.split(), '\n')   # 返回列表
输出：
0    graham chapman
1       john cleese
2     terry gilliam
3         eric idle
4       terry jones
5     michael palin
dtype: object 

0    14
1    11
2    13
3     9
4    11
5    13
dtype: int64 

0    False
1    False
2     True
3    False
4     True
5    False
dtype: bool 

0    [Graham, Chapman]
1       [John, Cleese]
2     [Terry, Gilliam]
3         [Eric, Idle]
4       [Terry, Jones]
5     [Michael, Palin]
dtype: object 

```
## 处理时间序列
* datetime 和 datautil用法
```python
from datetime import datetime    # 处理日期和时间
from dateutil import parser      # 解析来自字符串格式的日期


a = datetime(year=2015, month=7, day=4) # 自己构建一个日期
date = parser.parse("4th of July, 2015")
print(a)
print(date)
输出为：
2015-07-04 00:00:00
2015-07-04 00:00:00

```
# 处理时间类型—datetime64
```python
date = np.array('2015-07-04', dtype=np.datetime64)  # 格式化日期
print(date)
print(date + np.arange(12))  # 矢量运算
输出：2015-07-04
['2015-07-04' '2015-07-05' '2015-07-06' '2015-07-07' '2015-07-08'
 '2015-07-09' '2015-07-10' '2015-07-11' '2015-07-12' '2015-07-13'
 '2015-07-14' '2015-07-15']
```
* 按时间索引
```python

index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                          '2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)  # 构建具有时间索引数据的Series对象
print(data)
print('\n')
print(data['2014-07-04':'2015-07-04'])  # 开始索引
print('\n')
print(data['2015'])
输出：
2014-07-04    0
2014-08-04    1
2015-07-04    2
2015-08-04    3
dtype: int64

2014-07-04    0
2014-08-04    1
2015-07-04    2
dtype: int64

2015-07-04    2
2015-08-04    3
dtype: int64
```
## 高性能Pandas：eval() and query()
* 复合表达式
```python
rng = np.random.RandomState(42)
x = rng.rand(1000000)
y = rng.rand(1000000)
%timeit x + y
# 比循环算法快得多
```
* eval()用于有效的操作
```python
nrows, ncols = 100000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols))
                      for i in range(4))

print(pd.eval('df1 + df2 + df3 + df4'))
```
# DataFrame.query()方法
```python
rng = np.random.RandomState(42)
# 对任何列进行赋值
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])

result1 = df[(df.A < 0.5) & (df.B < 0.5)]
result2 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
a = np.allclose(result1, result2)  # 在默认误差下，每一个元素是否接近。返回布尔值
print(a)

result2 = df.query('A < 0.5 and B < 0.5')
b = np.allclose(result1, result2)
print(b)
输出：
True
True
```
