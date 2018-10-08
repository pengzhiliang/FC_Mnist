# FC_Mnist(课程作业)
时间：10.5-10.8 彭智亮
### 目的
用全连接神经网络对MNIST数据集进行分类
### 实现功能
1. 自定义层数及各层激活函数(ReLU & Leaky_relu & sigmoid & tanh) 见act_fun.py
2. Batch Normalization 见Network_BN.py
3. 学习速率随迭代次数衰减
### 最佳效果
![](http://p1i1k4m2v.bkt.gdipper.com/18-10-8/42603982.jpg)
### 训练技巧
1. 权重Msra 初始化 表现更佳
2. 训练数据标准化比归一化效果好
3. 层数尽量小，收敛更快
### 所遇问题
1. 对浅层神经网络，不使用BN收敛更快，效果更好 why?
2. 最后一层激活函数不能使用ReLu或者Leaky ReLu
### 实验平台
Linux 64bits + python2.7
### 使用
1. 不使用BN(能够复现我目前的最佳结果) 时间仓促，未整合
```bash
python Network.py
```
2. 使用BN
```bash
python Network_BN.py
```
