# c++气象项目数据中心实战
## 1. 前言：项目介绍
气象行业的数据（海洋、陆地、大气、雷达卫星）有数百种；
数据总量超过万亿条，每天新增的数据超过一亿；
气象行业的业务系统有几十个（预警预报和公共服务）。
气象数据中心是气象行业的基础系统，为业务系统提供数据支撑环境。
数据种类很多，数据量很庞大，业务系统很复杂；
图1
图2
### 1.1. 数据采集子系统
ftp客户端，采用ftp协议，采集数据文件
http客户端，采用http协议，从WEB服务接口采集数据
直连数据源的数据库，从表中抽取数据。
### 1.2. 数据处理和加工子系统
把各种格式的原始数据解码转换成xml格式的数据文件。
对原始数据进行二次加工，生成高可用的数据集。
### 1.3. 数据入库子系统
把数百种数据储存到数据中心的表中。
### 1.4. 数据同步子系统
MySQL的高可用方案只能解决部分问题，不够灵活、效率不高
把核心数据库（Slave）表中的数据按条件同步到业务数据库中
把核心数据库（Slave）表中的数据增量同步到业务数据库中。
### 1.5. 数据管理子系统
清理（删除）历史数据
把历史数据备份、归档
### 1.6. 数据交换子系统
把数据中心的数据从表中导出来，生成数据文件。
采用ftp协议，把数据文件推送至对方的ftp服务器。
基于tcp协议的快速文件传输系统
### 1.7. 数据服务总线
用c++开发WEB服务，为业务系统提供数据访问接口
效率极高（数据库连接池、线程池）
### 1.8. 网络代理服务
用于运维而开发的一个工具
I/O复用技术(select/poll/epoll)

### 1.9. 重点和难点
服务程序的稳定性
数据处理和数据服务的效率
功能模块的通用性

开发环境
CentOS7 ssh客户端SecureCRT vi
MySQL 5.7 客户端Navicat Premium
gcc g++
字符集utf-8

boost底层框架，不适合业务开发
不同的行业、公司有不同的开发框架，有不同的风格
开发框架是各家的秘笈，不外传，百度不到


## 2. 如何开发永不停机的服务程序

### 2.1. 开篇语
前端开发所见即所得，重点是实现功能
后台开发处理实现功能，还需要考虑服务程序的效率和稳定性

程序的异常：程序死掉或闪退是普遍现象，最好的解决办法是重启
后台服务程序无人值守，没有界面

用守护进程监控服务程序的运行状态
如果服务程序故障，调用进程将重启服务程序
保证系统7*24小时不间断运行


生成测试数据
实现生产测试数据的功能，提升代码能力
逐步掌握开发框架的使用
熟悉cdv, xml和json这三种常用的数据格式

服务程序的调度
学习linux信号的基础知识和使用方法
学习linux多进程的基础知识和使用方法
开发服务程序调度模块

守护进程的实现
学习linux共享内存的基础知识和使用方法
学习linux信号量的基础知识和使用方法
开发守护进程模块，与调度模块相结合，保证服务程序永不停机

两个常用的小工具
开发压缩文件模块
开发清理历史数据文件模块

调度和守护是后台程序员普遍采用的方案
与编程语言无关的方法
与业务无关的通用功能模块

### 2.2. 生成测试数据
全国气象站点参数
数据存放在文本文件中，全国有839个气象站



全国气象站分钟观测数据
站点每分钟进行一次观测，每观测一次产生一行数据

业务要求
根据全国气象站点参数，模拟生产观测数据
程序每分钟运行一次，每次生成839行数据，存放在一个文件中

搭建程序框架（运行的参数、说明文档、运行日志）
把全国气象战点参数文件加载到站点参数的容器中
遍历站点参数容器，生产每个站点的观测数据，存放在站点观测数据容器中
把站点观测数据容器中的记录写入文件

* 搭建程序的框架
* 加载站点参数
* 模拟观测数据
* 生成csv文件

### 2.3. 守护进程

服务程序由调度程序启动
如果服务程序死机（挂起），守护进程将终止它
服务程序被终止后，调度程序（procctrl）将重新启动它

服务程序在共享内存中维护自己的心跳信息
开发守护程序，终止已经死机的服务程序。


### 2.4. 服务程序的运行策略

/etc/rc.d/rc.local


## 3. 基于ftp协议的文件传输系统

ftp的基础知识
* ftp协议的基本概念
* 在CentOc7中安装和配置ftp服务
* 掌握ftp的常用命令

ftp客户端的封装
* 寻找开源的ftplib库，封装成C++的类
* 掌握Cftp类的使用方法

文件下载功能的实现
* 开发通用的文件下载模块，从ftp服务器下载文件

文件上传功能的实现
* 开发通用的文件上传模块，把文件上传到ftp服务器

文件的下载是数据采集子系统的一个模块
文件的上传是数据分发和数据交换子系统的模块


ftp协议很简单，创建ftp服务器很容易
适用于内部网络环境

### 3.1. ftp常用命令



### 3.2. ftp客户端的封装
* 寻找开源的ftplib库，封装成C++的类
* 掌握Cftp类的使用方法

### ftp 下载文件


