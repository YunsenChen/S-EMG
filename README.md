# S-EMG
【淘宝】https://m.tb.cn/h.gXBFWwQTwdw40tK?tk=MZLTWuMwxl5 MF6563 「干电极肌肉电传感器EMG单导 模拟信号采集模块智能可穿戴开源套件」
点击链接直接打开 或者 淘宝搜索直接打开

我当时做的时候是用了3个干电极


干电极可以替换成湿电极，湿电极波形更加稳定，但是佩戴相对麻烦

干电极买回来后，需要用arduino采集信号，并传输给电脑处理

设置好相应端口号后，运行main函数，按照屏幕上的指示进行信号采集，然后等待一会，做模型训练

训练完后就可以实时识别手势信号了

bilibli:【《基于表面肌电信号的手势识别系统》，介绍了一下自己的本科毕业设计，并找到了当时的源码】 【精准空降到 00:00】 https://www.bilibili.com/video/BV1CQ4y177Bc/?share_source=copy_web&vd_source=3e6359be3152aaff260e69cbed59150a&t=0

# todo：
1.可以用识别出来的信号做一些有趣的demo，比如游戏操控之类

2.main函数写的很啰嗦 并且没有封装 后续会上传一个封装的版本

3.很多特征其实没啥用 可以降低特征数量
