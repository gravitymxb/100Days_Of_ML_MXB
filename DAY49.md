# day 49 Pandas
## 合并与连接
```python
# 重新定义 Display
import pandas as pd
import numpy as np

class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)
```
合并数据集
```python
import pandas as pd

df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})

print('df1:\n', df1)
print('df2:\n', df2)
# 合并数据集，相同的标签合并在一起，字母按照顺序排
df3 = pd.merge(df1, df2)
print('df3:\n', df3)
输出：
df1:
   employee        group
0      Bob   Accounting
1     Jake  Engineering
2     Lisa  Engineering
3      Sue           HR
df2:
   employee  hire_date
0     Lisa       2004
1      Bob       2008
2     Jake       2012
3      Sue       2014
df3:
   employee        group  hire_date
0      Bob   Accounting       2008
1     Jake  Engineering       2012
2     Lisa  Engineering       2004
3      Sue           HR       2014
```
## 累计与分组
```python
2.1 应用Series
import seaborn as sns
import numpy as np
import pandas as pd

planets = sns.load_dataset('planets')  # 数据集
# print(planets.shape)
# print(planets.head())

# 简单聚合
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))  # 默认创建整形索引
print('ser:\n', ser)
print('ser.sum:\n', ser.sum())
print('ser.mean:\n', ser.mean())
输出：
ser:
0    0.374540
1    0.950714
2    0.731994
3    0.598658
4    0.156019
dtype: float64
ser.sum:
 2.811925491708157
ser.mean:
 0.5623850983416314
 
```
* 聚合、筛选、转换、应用
```python
import pandas as pd
import numpy as np

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                   columns =['key', 'data1', 'data2'])

print('df:\n', df)

# 使用aggregation聚合有更大的灵活性，求列数据中的最小、最大、均值值
df_aggregation = df.groupby('key').aggregate(['min', np.median, max])
print(df_aggregation)
输出：
df:
   key  data1  data2
0   A      0      5
1   B      1      0
2   C      2      3
3   A      3      3
4   B      4      7
5   C      5      9
    data1            data2           
      min median max   min median max
key                                  
A       0    1.5   3     3    4.0   5
B       1    2.5   4     0    3.5   7
C       2    3.5   5     3    6.0   9

```
## 数据透视表
* 激励数据透视表
```python
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
a = titanic.groupby('sex')[['survived']].mean()  # 按性别划分的存活率
print('a:\n', a)
b = titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()  # 加入阶级的影响
print('b:\n', b)
age = pd.cut(titanic['age'], [0, 18, 80])  # 继续加入年龄的因素
c = titanic.pivot_table('survived', ['sex', age], 'class')
print('c:\n', c)
fare = pd.qcut(titanic['fare'], 2)  # 自动计算分数位
d = titanic.pivot_table('survived', ['sex', age], [fare, 'class'])  # 加入票价信息
print('d:\n', d)
输出：
a:
         survived
sex             
female  0.742038
male    0.188908
b:
 class      First    Second     Third
sex                                 
female  0.968085  0.921053  0.500000
male    0.368852  0.157407  0.135447
c:
 class               First    Second     Third
sex    age                                   
female (0, 18]   0.909091  1.000000  0.511628
       (18, 80]  0.972973  0.900000  0.423729
male   (0, 18]   0.800000  0.600000  0.215686
       (18, 80]  0.375000  0.071429  0.133663
d:
 fare            (-0.001, 14.454]            ... (14.454, 512.329]          
class                      First    Second  ...            Second     Third
sex    age                                  ...                            
female (0, 18]               NaN  1.000000  ...          1.000000  0.318182
       (18, 80]              NaN  0.880000  ...          0.914286  0.391304
male   (0, 18]               NaN  0.000000  ...          0.818182  0.178571
       (18, 80]              0.0  0.098039  ...          0.030303  0.192308

[4 rows x 6 columns]
```
