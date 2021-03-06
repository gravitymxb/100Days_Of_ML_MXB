# day45 NumPy
* numpy（Numerical Python）提供了python对多维数组对象的支持：ndarray，具有矢量运算能力，快速、节省空间。numpy支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。            
* 在C语言中，每个变量的数据类型是显式声明的。而在Python中，类型是动态推断的。所以我们可以将任何类型的数据幅值给任何变量   
* 与Python列表不同，NumPy受限于所有包含**相同类型**的数组。如果类型不匹配，NumPy将在可能的情况下向上转换(例子中 整数向上转换为浮点)。也可以直接用 dtype明确设定其数据类型。
* 记录几种比较陌生的函数用法
```python
    a = x[::-1]  # 列表倒序输出 
    b = x[1::2]  # 从第一个数起，隔一取一  
    c = np.random.random((3, 3))  # 生成3行，3列的数，每个数都从0-1中随机得到  
    d = np.random.normal(0, 1, (3, 3))  # 生成高斯分布的概率密度随机数,格式为(均值，标准差，形状)    
    e = np.random.randint(0, 10, (3, 4))  # 随机生成[0, 10）之间的整数  (0. 10)控制随机地范围，（3,4）控制行列数及维度。    
    f = np.concatenate([x, y], axis=0)  # 拼接x、y两矩阵，若axis为0，按行拼接，若为1，按列拼接   
    print(x3.ndim)  # 输出的是数组x3的维度     
    print(x3.nbytes)  #输出x3的总大小    
    x1, x2, x3 = np.split(x, [3, 5])  # 将x数组划分为三部分，其中的3表示第1，2段划分位置，5表示第2，3段划分位置,且前开后闭
    
```
具体见[此处](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.03-Computation-on-arrays-ufuncs.ipynb)
