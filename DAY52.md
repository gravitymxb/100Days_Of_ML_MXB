# day52 matplotlib
## 直方图
1、普通直方图
```python
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-white')          # 设置绘图style
np.random.RandomState(1)
data = np.random.randn(1000)

plt.hist(data, bins=30,  # 指定条状图的个数
         normed=True,    # 每个条状图的占比例比,默认为1
         alpha=0.5,      # 透明度
         histtype='stepfilled',  # 线条的类型
         color='steelblue',    # 设定条状图整体颜色
         edgecolor='red')      # 设定edge的颜色
plt.show()
```
![tu1]()
2、二维直方图以及装箱操作

```python
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-white')

mean = [0, 0]           # mean表示多维分布的均值
cov = [[1, 1], [1, 2]]  # cov表示协方差矩阵
x, y = np.random.multivariate_normal(mean, cov, 10000).T 
plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()  # 显示颜色变化柱
cb.set_label('counts in bin')
plt.show()
```
![tu2]()
## 配置颜色条
```python
import matplotlib.pyplot as plt
plt.style.use('classic')   # 设置绘画风格为经典
import numpy as np

x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])

plt.imshow(I, cmap='gray')  # 映射为灰度
plt.colorbar()              # 配置颜色条函数
plt.show()
```
![tu3]()
## 多子图 
```python
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

ax1 = plt.axes()  # 标准图
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])   # 设置宽度和高度的比例为0.65，设置轴的0.2比例
plt.show()
```
![tu4]()
```python
# 非对称子图，设置为2行3列，子图之间宽度间隔为0.4，高度间隔为0.3
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])  # 第一列和第二列合在一起
plt.subplot(grid[1, 2])

plt.show()
```
![tu5]()
