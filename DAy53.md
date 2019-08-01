# DAY53  matplotlib
1、画三维图
```python
from mpl_toolkits import mplot3d   # 导入三维绘图工具包
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')     # 指定画3D图
plt.show()
```
![tu1](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/53.1.png)

2、三维图中点与线的可视化
```python
from mpl_toolkits import mplot3d  
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')  

# 设定3d中的线
zline = np.linspace(0, 15, 1000)     # 将0-15均匀分成1000份
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')        

# 设定3d中的点
zdata = 15 * np.random.random(100)      # 随机生成100个点
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)  
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')  # 默认情况下 散点图中点的透明度是不一的，使其有深度感
plt.show()
```
![tu2](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/53.2.png)
3、三维等高线图
```python
from mpl_toolkits import mplot3d 
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):

    return np.sin(np.sqrt(x ** 2 + y ** 2))    # 计算平方根

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)                 # 用x，y轴上的点画出网格示意
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')           
ax.contour3D(X, Y, Z, 50, cmap='binary')  # 画等高线3维图
ax.set_xlabel('x')         
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
```
![tu3](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/53.3.png)
4、线框图与曲面图
```python

from mpl_toolkits import mplot3d  
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):

    return np.sin(np.sqrt(x ** 2 + y ** 2))  # 平方根计算

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)        # 用x，y轴上的点画出网格示意
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')  
ax.set_title('wireframe')

plt.show()
```
![tu4](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/53.4.png)

```python
from mpl_toolkits import mplot3d 
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):

    return np.sin(np.sqrt(x ** 2 + y ** 2))  # 平方根计算

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)    # 用x，y轴上的点画出网格示意
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')   # 画3d图
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,  # rstride: 设定行的跨度为1，cstride：设定列的跨度为1
                cmap='viridis', edgecolor='none')
ax.set_title('surface')
plt.show()

```
![tu5](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/53.5.png)
