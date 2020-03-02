#Algorothm
<hr>


* 监督学习 

在拥有特征与正确结果相对应的数据情况下进行学习

* 无监督学习

聚类

### 线性回归 

* 单变量线性回归

基于数据 训练方程参数

构建cost function 计算真实值与预测值误差 得到使方程的代价最小的参数

计算cost 方法 
batch批量梯度下降 计算下降最快的方向 求导 每次进行参数更新 减去学习率*cost的导数 
学习率为下降的步长

* 多变量线性回归

特征为多维  

 处理数据尺度不同 - 特征缩放 能够提高梯度下降速度 椭圆变正圆

正规方程法（最小二乘）
>注：对于那些不可逆的矩阵（通常是因为特征之间不独立，如同时包含英尺为单位的尺寸和米为单位的尺寸两个特征，也有可能是特征数量大于训练集的数量），正规方程方法是不能用的

![截屏2020-03-01下午6.08.03](https://tva1.sinaimg.cn/large/00831rSTly1gcevmq8btoj31820dqn00.jpg)

即：均方误差最小化
单元线性 

![截屏2020-03-02上午12.07.23](https://tva1.sinaimg.cn/large/00831rSTly1gcevmb5f6qj30g8078my1.jpg)

多元线性转为矩阵

![截屏2020-03-02上午12.07.47](https://tva1.sinaimg.cn/large/00831rSTly1gcevmkmquqj30fu030aac.jpg)
**w为（w:b）**
当矩阵为可逆时得到结果

![截屏2020-03-02上午12.14.02](https://tva1.sinaimg.cn/large/00831rSTly1gcevsfav04j309u02qdfx.jpg)



### 逻辑回归

分类问题 与预测值为离散值
sigmoid function逻辑回归
```1 / (1 + np.exp(-z))```

cost function构建

![截屏2020-03-02下午7.37.16](https://tva1.sinaimg.cn/large/00831rSTly1gcftenhml2j30ey02gwen.jpg)
其中，![54249cb51f0086fa6a805291bf2639f1](https://tva1.sinaimg.cn/large/00831rSTly1gcftg9fygyj30ev01t3yi.jpg)

代入后得到最终代价函数

![截屏2020-03-02下午7.39.55](https://tva1.sinaimg.cn/large/00831rSTly1gcfthqgarqj30sy02m0t3.jpg)
然后进行梯度下降算法求θ



一些梯度下降算法之外的选择： 除了梯度下降算法以外，还有一些常被用来令代价函数最小的算法，这些算法更加复杂和优越，而且通常不需要人工选择学习率，通常比梯度下降算法要更加快速。有：**共轭梯度**（**Conjugate Gradient**），**局部优化法**(**Broyden fletcher goldfarb shann,BFGS**)和**有限内存局部优化法**(**LBFGS**) ，**fminunc**



**共轭梯度**

**局部优化法**

**有限内存局部优化法**





### Onehot编码



### Normalization

数据标准化处理主要包括数据**同趋化**处理和**无量纲**化处理两个方面。

1.把特征的各个维度标准化到特定的区间

2.把有量纲表达式变为无量纲表达式

* 优点
加快基于梯度下降法或随机梯度下降法模型的收敛速度
提升模型的精度


### 如何解决数据imbalance情况

### 深度学习模型层数对实验效果影响