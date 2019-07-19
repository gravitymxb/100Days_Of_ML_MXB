# DAY40 
1、新引入了两个模块os与cv2
**os模块**：os模块提供了多数操作系统的功能接口函数。当os模块被导入后，它会自适应于不同的操作系统平台，根据不同的平台进行相应的操作。用于对文件、目录等进行操作   
**cv2模块**：即opencv，可以用于处理图像信息——改变大小、腐蚀图像、膨胀图像、等等。   
2、部分函数    
cv2.imread(文件名，标记)读入图像，
cv2.IMREAD_COLOR（）：读入彩色图像
cv2.IMREAD_GRAYSCALE（）：以灰度模式读入图像
cv2.imshow()：显示图像
cv2.imwrite(文件名，img)：保存图像

代码如下   
```python
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Day 40
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from  tqdm import tqdm

DATADTR = "D:/python_pycharm/datasets/PetImages"  # 路径
CATEGORIES = ["Dog","Cat"]
for category in CATEGORIES:
    path = os.path.join(DATADTR, category)  # 创建路径
    for img in os.listdir(path):            # 迭代遍历每个图片
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # 转化成array
        plt.imshow(img_array, cmap='gray')  # 转化成图像展示
        plt.show()   # 展示图片

        break  # 跳出循环，展示一张图片
    break
print(img_array)
print(img_array.shape)
IMG_SIZE = 50
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

# 改变size为100，看图片是否更清晰
IMG_SIZE = 100
new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADTR,category)
        class_num = CATEGORIES.index((category))  # 得到分类，dog=0，cat=1

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))    # 大小转换
                training_data.append([new_array, class_num])  # 加入训练数据中
            except Exception as e:     # 为了保证输出是整洁的
                pass
create_training_data()
print(len(training_data))

import random
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,1)

import pickle

pickle_out = open("D:/python_pycharm/datasets/X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("D:/python_pycharm/datasets/y.pickle", "wb")
pickle_out,dump(y, pickle_out)
pickle_out.close()

pickle_in = open("../datasets/X.pickle", "rb")
X = pickle_load(pickle_in)

pickle_in = open("../datasets/y.pickle", "rb")
y = pickle_load(pickle_in)
```
![40.1](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/40.1.png)
![40.2](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/40.2.png)
![40.3](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/40.3.png)

