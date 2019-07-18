# day39 深度学习基础Python，TensorFlow和Keras
## [TensorFlow](https://blog.51cto.com/zero01/2065598)与Keras
* TensorFlow是Google开源的基于数据流图的机器学习框架，支持python和c++程序开发语言，其命名基于工作原理，tensor 意为张量（即多维数组），flow 意为流动。即多维数组从数据流图一端流动到另一端。   
![tensorflow](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/TensorFlow%E6%95%B0%E6%8D%AE%E6%B5%81%E5%9B%BE.gif)
* Keras是基于TensorFlow和Theano的深度学习库，是由纯python编写而成的高层神经网络API，也仅支持python开发。它是为了支持快速实践而对tensorflow或者Theano的再次封装。   
## 代码如下
```python
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# day 39  TensorFlow
# 导入keras
import tensorflow.keras as keras

# 导入tensorflow
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train[0])


import matplotlib.pyplot as plt

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
print(y_train[0])

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

print(x_train[0])

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)
predictions = model.predict(x_test)
print(predictions)

import numpy as np
print(np.argmax(predictions[0]))

plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()


# 保存模型
model.save('epic_num_reader.model')

# 加载保存的模型
new_model = tf.keras.models.load_model('epic_num_reader.model')

# 测试保存的模型
predictions = new_model.predict(x_test)
print(np.argmax(predictions[0]))
```
![图一](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/39.1.png)
![图二](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/39.2.png)
![图三](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/39.3.png)
