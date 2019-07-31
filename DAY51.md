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

```
输出为：  
![tu1](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/51.1.png)
## 简易散点图
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.style.use('seaborn-whitegrid')  # 设置画图风格为seaborn-whitegrid
plt.plot(x, y, 'o', color='black')  # 点色为黑
plt.show()



rng = np.random.RandomState(0)  # 固定随机数种子
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:  # 各种类型的表示
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8)
plt.show()


```
![tu2](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/51.2.png)
![tu3](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/51.3.png)
* 用scatter画散点图
```python
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.scatter(x, y, marker='o')  # 散点函数
plt.show()

```
![tu4](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/51.4.png)
## 可视化异常处理
```python 
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt='.k') # 此函数即为输出带误差数据的函数
plt.show()

```
![tu5](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/51.5.png)
