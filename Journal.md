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
