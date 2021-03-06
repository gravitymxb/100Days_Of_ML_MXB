# day41
## ![CNN算法](https://blog.csdn.net/love__live1/article/details/79481052)  
其基本结构如下：  
1、输入层：用于数据的输入  
2、卷积层：使用卷积核进行特征提取和特征映射   
3、激励层：由于卷积也是一种线性运算，因此需要增加非线性映射   
4、池化层：进行下采样，对特征图稀疏处理，减少数据运算量。  
5、全连接层：通常在CNN的尾部进行重新拟合，减少特征信息的损失   
三个特点：  
局部连接：每个神经元不再和上一层的所有神经元相连，而只和一小部分神经元相连。这样就减少了很多参数    
权值共享：一组连接可以共享同一个权重，而不是每个连接有一个不同的权重，这样又减少了很多参数。    
下采样：可以使用Pooling来减少每层的样本数，进一步减少参数数量，同时还可以提升模型的鲁棒性。  
卷积:   
1、从原始图像的左上角开始，选择和卷积核大小相同的区域；  
2、选出来的区域和卷积核逐个元素做乘积，然后求和，得到的值作为新的图像的一个像素点的值；  
3、在原始图片上水平和垂直移动选择的区域，重复步骤2的操作。移动的步长可以是1或者大于1（如果是大于1的步长，得到的新图像尺寸会缩小）。   
4、有时为了不让新生成的图片缩小，可以给原始图片添加填充（padding）  
池化：即将空间上相邻的点进行聚合处理。不仅可以降低特征的维度，同时还可以改善过拟合的结果  
    平均池化：池化区域内所有值的平均值作为池化结果。  
    最大池化：池化区域内所有值中的最大值作为池化结果。   

[详解](https://zhuanlan.zhihu.com/p/61510829)
代码如下：
```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle

pickle_in = open("D:/python_pycharm/datasets/X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("D:/python_pycharm/datasets/y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0
model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(64))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)
```

