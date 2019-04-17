# 2019.04.17学习日志
   回顾了day1和day2的代码，温故而知新。虽然不好描述，但确实加深了对前两天代码整体的理解。
   纠正了之前的一些错误，如读取csv文件我一直用的绝对路径（用相对路径——将csv文件与py文件放在一起）
   代码注释放在代码下边（应放在上边，翻看群记录发现学长说过）
   另外，根据群聊学习了虚拟变量陷阱、独热编码等
   
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

