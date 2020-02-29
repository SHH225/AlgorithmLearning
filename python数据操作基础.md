# Python 基础 
### Numpy


》特性
- ndarray ,一个具有矢量算术运算和复杂广播能力的快速且节省空间的多维数组。
- 整组数据快速运算
- 读写磁盘数据、操作内存映射

》功能
- 数据整理、清理，子集构造、唯一化、集合运算。
- 数组算法 - 排序、唯一化、集合运算
- 异构数据合并、连接，数据对齐和关系型数据运算
- 数据的分组运算

运算比原生python快
原因：在一个连续内存中存储数据，使用内存少，独立于其他python内置对象，可以整个数组计算。


numpy数组对象 =》ndarray

一般引用
```python
import numpy as np
```

np.zeros() - generate target shape matrix with value 0
np.ones()  same as above
np.empty() - sometimes it returns with undefined trash value    not always 0

np.arange(15)  -> ([0,1,2,.....14])

#### **dtype** 
- mostly float64

np.array([1,2,3],dtype=np.float64)

int8,int16,int32,int64
float16,float32,float64,float128
complex64  bool object string_  unicode

**类型转换** 
float_arr=arr.astype(np.float64)

string_类型长度固定 过长会被截掉

#### 数组运算
