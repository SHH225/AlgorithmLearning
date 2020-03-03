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

<img src="https://tva1.sinaimg.cn/large/00831rSTly1gcevmb5f6qj30g8078my1.jpg" alt="截屏2020-03-02上午12.07.23" style="zoom:50%;" />

多元线性转为矩阵

<img src="https://tva1.sinaimg.cn/large/00831rSTly1gcevmkmquqj30fu030aac.jpg" alt="截屏2020-03-02上午12.07.47" style="zoom:50%;" />
**w为（w:b）**
当矩阵为可逆时得到结果

<img src="https://tva1.sinaimg.cn/large/00831rSTly1gcevsfav04j309u02qdfx.jpg" alt="截屏2020-03-02上午12.14.02" style="zoom:50%;" />



### 逻辑回归

分类问题 与预测值为离散值
sigmoid function逻辑回归
```1 / (1 + np.exp(-z))```

cost function构建

<img src="https://tva1.sinaimg.cn/large/00831rSTly1gcftenhml2j30ey02gwen.jpg" alt="截屏2020-03-02下午7.37.16" style="zoom:50%;" />
其中，<img src="https://tva1.sinaimg.cn/large/00831rSTly1gcftg9fygyj30ev01t3yi.jpg" alt="54249cb51f0086fa6a805291bf2639f1" style="zoom:50%;" />

代入后得到最终代价函数

<img src="https://tva1.sinaimg.cn/large/00831rSTly1gcfthqgarqj30sy02m0t3.jpg" alt="截屏2020-03-02下午7.39.55" style="zoom:50%;" />
然后进行梯度下降算法求θ

逻辑回归的代价函数看起来和线性回归一样  但是 h(theta) 不同 即假设函数。线性回归多为多项式

<img src="https://tva1.sinaimg.cn/large/00831rSTly1gcgt7vb51lj307i022weh.jpg" alt="截屏2020-03-03下午4.15.28" style="zoom:50%;" />



g为sigmoid X为特征向量 theta参数

一些梯度下降算法之外的选择： 除了梯度下降算法以外，还有一些常被用来令代价函数最小的算法，这些算法更加复杂和优越，而且通常不需要人工选择学习率，通常比梯度下降算法要更加快速。有：**共轭梯度**（**Conjugate Gradient**），**局部优化法**(**Broyden fletcher goldfarb shann,BFGS**)和**有限内存局部优化法**(**LBFGS**) ，**fminunc**



**共轭梯度**

**局部优化法**

**有限内存局部优化法**


处理过拟合问题

PCA主成分分析（降维）

正则化：保留所有的特征，但是减少参数的大小（magnitude）

### 正则化

保留所有的特征，但是减少参数的大小
高次项导致了过拟合的产生，降低高次项系数
**cost function**

引入正则化参数lamda
* 正则化线性回归

**梯度下降方法**
<img src="https://tva1.sinaimg.cn/large/00831rSTly1gcgqzagzrhj30iq03caab.jpg" alt="截屏2020-03-03下午2.58.04" style="zoom:50%;" />

求导整理后得



<img src="https://tva1.sinaimg.cn/large/00831rSTly1gcgrt7hxlxj30k202ujrm.jpg" alt="截屏2020-03-03下午3.27.40" style="zoom:50%;" />

**正规方程**

<img src="https://tva1.sinaimg.cn/large/00831rSTly1gcgrul7l50j308c0230so.jpg" alt="71d723ddb5863c943fcd4e6951114ee3" style="zoom:50%;" />


* 正则化逻辑回归

  cost function

<img src="https://tva1.sinaimg.cn/large/00831rSTly1gcgsfdfui8j30xc03wmxn.jpg" alt="截屏2020-03-03下午3.48.50" style="zoom:50%;" />



* L1正则化：Lasso回归
  产生**稀疏权值矩阵**，即产生一个稀疏模型，可以用于特征选择
  权值向量w中各个元素的绝对值之和

  <img src="https://tva1.sinaimg.cn/large/00831rSTly1gcgv4zgwcuj30ay03yaa3.jpg" alt="截屏2020-03-03下午5.22.29" style="zoom:50%;" />

令L=正则项，此时我们的任务变成在L约束下求出J0取最小值的解。



* L2正则化：Ridge回归
可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合
权值向量ww中各个元素的平方和然后再求平方根



### 随机森林（分类、回归）

### Onehot编码

**分类变量**作为**二进制向量**的表示

要求每个类别之间相互独立，如果之间存在某种连续型的关系

独热编码解决了分类器不好处理属性数据的问题，在一定程度上也起到了扩充特征的作用。它的值只有0和1，不同的类型存储在垂直的空间。
缺点：当类别的数量很多时，特征空间会变得非常大，成为一个高维稀疏矩阵。在这种情况下，一般可以用PCA来减少维度。而且one hot encoding+PCA这种组合在实际中也非常有用

四. 什么情况下(不)用独热编码？

用：独热编码用来解决类别型数据的离散值问题;

不用：将离散型特征进行one-hot编码的作用，是为了让距离计算更合理，但如果特征是离散的，并且不用one-hot编码就可以很合理的计算出距离，那么就没必要进行one-hot编码。 有些基于树的算法在处理变量时，并不是基于向量空间度量，数值只是个类别符号，即没有偏序关系，所以不用进行独热编码。  Tree Model不太需要one-hot编码： 对于决策树来说，one-hot的本质是增加树的深度。



### Normalization（归一化）

数据标准化处理主要包括数据**同趋化**处理和**无量纲**化处理两个方面。

1.把特征的各个维度标准化到特定的区间

2.把有量纲表达式变为无量纲表达式

* 优点
加快基于梯度下降法或随机梯度下降法模型的收敛速度
提升模型的精度

###### 常见标准化方法
* z-score
   x* = (x - μ ) / σ
 z-score标准化方法适用于属性A的最大值和最小值未知的情况，或有超出取值范围的离群数据的情况。该种归一化方式要求原始数据的分布可以近似为高斯分布，否则归一化的效果会变得很糟糕。

#

### LDA

### L1 L2


### 如何解决数据imbalance情况

下采样 

过采样

数据增强

权重设置

### 深度学习模型层数对实验效果影响

### RESNET （深度残差网络）
深度残差网络的设计就是为了克服这种由于网络深度加深而产生的学习效率变低，准确率无法有效提升的问题（也称为**网络退化**）

甚至在一些场景下，网络层数的增加反而会降低正确率。这种本质问题是由于出现了信息丢失而产生的过拟合问题（overfitting，所建的机器学习模型或者是深度学习模型在训练样本中表现的过于优越，导致在验证数据集及测试数据集中表现不佳，即为了得到一致假设而使假设变得过度复杂）。解决思路是尝试着使他们引入这些刺激的差异性和解决泛化能力为主。
### VGG19

### Inception

### MobileNet 

### LSTM GRU
