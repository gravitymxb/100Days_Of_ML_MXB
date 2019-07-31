# DAY51   Matplotlib数据可视化
## 直方图
* 设置绘图风格

```python
x = np.linspace(0, 10, 100)   # 表示0-10之间均匀取得100个数
fig = plt.figure()
plt.plot(x, np.sin(x))   # 画出来的图为正弦函数
plt.plot(x, np.cos(x))    # 为余弦函数图像
plt.style.use('classic')  # 设置绘图风格为经典风格
fig.savefig('my_figure.png')  # 生成并将图片保存到该目录
plt.show()   # 显示图

```
* MATLAB风格
```python
import matplotlib.pyplot as plt          # 若引入matplotlib则 as 为 mpl
import numpy as np

# matlab风格的工具
plt.figure()           
x = np.linspace(0, 10, 100)             # 在0-10之间均匀取得100个数
# 设置一个子图
plt.subplot(2, 1, 1)    # (行， 列，子图编号)
plt.plot(x, np.sin(x))

# 设置第二个子图
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.show()
输出为：
![tu1]()
```
## 简易散点图
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.style.use('seaborn-whitegrid')  # 设置画图风格为seaborn-whitegrid
plt.plot(x, y, 'o', color='black')  # 点色为黑
plt.show()
![tu2]()


rng = np.random.RandomState(0)  # 固定随机数种子
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:  # 各种类型的表示
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8)
plt.show()
![tusan]()

```
* 用scatter画散点图
```python
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.scatter(x, y, marker='o')  # 散点函数
plt.show()
![tu4]()
```
## 可视化异常处理
```python 
plt.errorbar()    函数用于表现有一定置信区间的带误差数据。
plt.errorbar(x,   
	y,   
	yerr=None,  
	xerr=None,  # xerr,yerr: 数据的误差范围  
	fmt='',   # 数据点的标记样式以及相互之间连接线样式
	ecolor=None, 
	elinewidth=None,   # 误差棒的线条粗细
	capsize=None,   # 误差棒边界横杠的大小
	capthick=None  # 误差棒边界横杠的厚度
)

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)

plt.style.use('seaborn-whitegrid')
plt.errorbar(x, y, yerr=dy, fmt='.k')
plt.show()
```

