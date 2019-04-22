# 2019.04.22 学习日志
* 维度（特征维度）：一般无特别说明，指的都是特征的数量。  [维度其他角度理解](https://blog.csdn.net/yoggiecda/article/details/88574418)   
* 降维算法中的“降维”：指的是降低特征矩阵中特征的数量，目的是让算法运算快，效果好，另外可使 数据可视化（三维及以下可视）  
* 运用StandardScaler出现警告：DataConversionWarning: Data with input dtype int64 were all converted to float64 by StandardScaler.    
  解决方法：指定数据类型：  
  `X_train = X_train.astype(np.float64)`     
  ` X_test = X_test.astype(np.float64)`
* **fit**会重置模型： scikit-learn模型的一个重要性质，调用fit总会重置模型之前学习到的所有内容，因此，如果在一个数据集上构建模型，然后再在另一个数据集上再次调用fit，那么模型会‘忘记’从第一个数据集中学到的所有内容。可以对一个模型多次调用fit，其结果与在‘新’模型上调用fit是完全相同的。
# 2019.04.21学习日志
* StandardScaler类 作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。  
  其中fit_transform 和 transform 的区别：  
     * fit_transform()先拟合数据，再标准化    
     * transform()数据标准化    
     * 具体见[通俗地讲清楚fit_transform()和transform()的区别](https://blog.csdn.net/appleyuchi/article/details/73503282) 以及 [有关StandardScaler的transform和fit_transform方法](https://www.jianshu.com/p/2a635d9e894d)   
* 混淆矩阵（Confusion Matrix）  
    * 又称为可能性表格或是错误矩阵。  
    * 它是一种特定的矩阵用来呈现算法性能的可视化效果，通常是监督学习（非监督学习，通常用匹配矩阵：matching matrix）。其每一列代表预测值，每一行代表的是实际的类别。这个名字来源于它可以非常容易的表明多个类别是否有混淆（也就是一个class被预测成另一个class）。
    * 具体见[混淆矩阵分析]（https://blog.csdn.net/vesper305/article/details/44927047）  


# 2019.04.19学习日志
逻辑回归相关 见[DAY4.md](https://github.com/gravitymxb/100Days_Of_ML_MXB/blob/master/DAY4.md)
# 2019.04.18学习日志  
1、欠拟合与过拟合的概念：  
   两者都是模型学习能力与数据复杂度之间失配的结果  
   “欠拟合”常常在模型学习能力较弱，而数据复杂度较高的情况出现，此时模型由于学习能力不足，无法学习 到数据集中的“一般规律”，因而导致泛化能力弱。  
    “过拟合”常常在模型学习能力过强的情况中出现，此时的模型学习能力太强，以至于将训练集单个样本自身的特点都能捕捉到，并将其认为是“一般规律”，同样这种情况也会导致模型泛化能力下降。  
2、特征缩放：常用的方法有调节比例、标准化等  
    多数的分类器利用两点间的距离（欧氏距离）计算两点的差异，若其中一个特征具有非常广的范围，那两点间的差异就会被该特征左右，因此，所有的特征都该被标准化，这样才能大略的使各特征依比例影响距离。    
3、print的用法
    `print(value,....,sep = ' ',end = '\n', file=sys.stdout,flush=False)`    
    python中的print自动换行，因为默认**end = '\n'** ，想要不换行，我们把end='\n'设置为**end = ''** 就行  
    sep = ' ' :单引号中的内容是在输出两个参数是，两者之间出现的字符，可以自己更改  
4、虚拟变量陷阱
    即：由于引入虚拟变量个数与定性因素个数相同出现的模型无法估计的问题
    解决方法：在引入虚拟变量时要求如果有m个定性变量，在模型中引入m-1个虚拟变量。不可使模型解释变量之间出现完全共线的情况。
# 2019.04.17学习日志
   回顾了day1和day2的代码，温故而知新。虽然不好描述，但确实加深了对前两天代码整体的理解。   
   纠正了之前的一些错误，如读取csv文件我一直用的绝对路径（用相对路径——将csv文件与py文件放在一起）   
   代码注释放在代码下边（应放在上边，翻看群记录发现学长说过）  
   另外，根据群聊学习了虚拟变量陷阱、独热编码等    
   [LabelEncoder 和 OneHotEncoder 辨析](https://blog.csdn.net/weixin_38656890/article/details/80849334)    
   [数据挖掘OneHotEncoder独热编码和LabelEncoder标签编码](https://blog.csdn.net/ccblogger/article/details/80010974)  
   
# 2019.04.16学习日志  
1、学习了`matplotlib.pyplot`的运用：  
      day2的代码中主要是用其做的散点图  
      如：plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')  
      另外还有折线图、柱状图等  
   基础参数设置：  
       x轴与y轴标签：  
        ```  plt.xlabel('...') # str类型    
          plt.ylabel('...')  ```  
       图的名称：``` plt.title('...', fontsize= 20)  ```  
       轴刻度：``` plt.axis(xmin, xmax, ymin, ymax)  如：plt.axis([0,2,4,6])  ```  
2、安装插件autopep8
3、学会了如何上传图片，如何在md文件中显示等

# 2019.04.14学习日志
## 主要内容：   学习day2代码
1、用`pd.read_csv`读取文件时，文件路径的填写方式：  
`'D:\python_pycharm\data\studentscores.csv'`或直接在pycharm上找到文件，再copy path  
2、读取list
例：`X = dataset.iloc[:, :1].values`  #  ` : `即为读取全部行，`:1 ` 为从第0列到第一列

# 2019.04.13学习日志
## 主要内容：
##     一、day2码
##     二、学习README.md文件的写法
##     三、安装Anaconda
### 详细记录安装Anaconda  
1、若直接在官网上下载，下载速度很慢。故在清华大学开源软件镜像站下载  
下载地址：[清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/)  
2、安装*Anaconda3-2019.03-Windows-x86_64*版本，出现conda环境未激活情况，查询各种资料未果后，改为下载安装*Anaconda3-4.3.0-Windows-x86_64*版本，安装成功，未出现错误。  
3、安装成功后修改其包管理镜像为国内源，即在**cmd**中执行    
  ```
  conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  conda config --set show_channe1_urls yes
  ```  
4、初次安装的包一般比较老，为防止出错在命令行模式执行 ```conda update --all```命令对所有包进行升级。

