---
title: GBDT 梯度提升决策树
draft: false
tags:
  - 机器学习
---
 

# 1 回顾

## 1.1 集成学习

在学习GBDT之前，再了解下**集成学习的概念**。

几乎所有的数据科学任务可以抽象为基于已知数据x预测未知取值的变量y的取值，即函数y=f(x)，这是一个理想的状态，真实世界中可不会预测那么准确，预测值和真实值总是存在一定的差异：y=f(x)+residual，其中residual被称为残差。一般评价一个模型的预测能力，考察两个方面：偏差，即与真实值分布的偏差大小；方差，体现模型预测能力的稳定性，或者说鲁棒性。

集成学习（Ensemble Model）不是一种具体的模型，而是一种模型框架，采用的是“三个臭皮匠顶一个诸葛亮”的思想。集成模型的一般做法是，将若干个模型（俗称“弱学习器”，weak learner，基础模型，base model）按照一定的策略组合起来，共同完成一个任务——特定的组合策略，可以帮助我们降低预测的偏差或者方差。

常见的集成策略有bagging，stacking，boosting

- bagging
    
    bagging之前有介绍
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084340.png)
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084340.png)


- stacking
    
    stacking模型比bagging模型更进两步，a：允许使用不同类型的模型作为base model，b：使用一个机器学习模型把所有base model的输出汇总起来，形成最终的输出。b中所述模型称为“元模型”。在训练的时候，base model们直接基于训练数据独立训练，而元模型会以它们的输出为输入数据、以训练数据的输出为输出数据进行训练。stacking模型认为，各个基础模型的能力不一，投片的时候不能给以相同的权重，而需要用一个“元模型”对各个基础模型的预测值进行加权。
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084357.png)

    
- boosting
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084410.png)

    

## 1.2 监督学习

**回顾下监督学习**，假定有N个训练样本:{(X(1),y(1)),(X(2),y(2)),⋯,(X(n),y(n))}，找到一个函数F(x)对应一种映射使得其损失函数最小

$$
F^* = argminL(y,F(x))
$$

如何最小？需要使用梯度下降

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084420.png)


找到一组W使得L最小，进而求得F*

找到下降方向

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084426.png)


不断更新

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084434.png)


当最终结果保持不变时，认为我们找到了f(w)也就是L(y,F(x))最小的一组w

最终求得的w可以表示为连加的结构

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084442.png)


## 1.3 **函数空间的梯度下降**

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084450.png)


![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084457.png)


# 2 GBDT（梯度提升决策树）

- 利用多棵回归树处理回归问题的解决方案，通过调整损失函数也可以处理分类问题
- 通过全部样本迭代生成多棵回归树，对于回归问题的损失函数是什么？

## 2.1 回归树

- 每次分裂后计算分裂出的节点的平均值
- 将平均值带入MSE损失函数进行评估
- 分裂标准：1.分裂特征 2.分裂的值

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084505.png)


### 2.1.1 每棵回归树拟合负梯度

GBDT应用于回归问题

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084512.png)


### 2.1.2整体流程

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084521.png)


### 2.1.3 例子

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084528.png)


### 2.1.4 总结

GBDT与传统的Boosting区别较大，它的每一次计算都是为了减少上一次的残差，而为了消除残差，我们可以在残差减小的梯度方向上建立模型，所以说，在GradientBoost中，每个新的模型的建立是为了使得之前的模型的残差往梯度下降 的方向，与传统的Boosting中关注正确错误的样本加权有这很大的区别

在GrandientBoosting算法中，关键就是利用损失函数的负梯度方向在当前模型的值作为残差的近似值，进而拟合一棵CART回归树

GBDT会累加所有树的结果，而这种累加是无法通过分类完成的，因此GBDT的树都是CART回归树，而不是分类树（尽管GBDT调整后也可以用于分类但不代表GBDT的树为分类树）

## 2.2 GBDT做二分类任务

### 2.2.1 GBDT算法概述

GBDT是Boosting算法的一种，按照boosting的思想，在GBDT算法的每一步，用一棵决策树去拟合当前学习器的残差，获得一个新的弱学习器。将这每一步的决策树组合起来，就得到了一个强学习器。

具体来说，假设训练样本{xi，yi}，i=1…n，每m-1步获得的集成学习器为Fm-1(x)，那么GBDT通过下面的递推式，获得一个新的弱学习器h(x)：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084535.png)


其中h(x)是在函数空间H上最小化损失函数，一般来说这是比较难做到的。但是，如果我们只考虑到精确地拟合训练数据的话，可以将损失函数

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084541.png)


看做向量

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084548.png)


上的函数，这样在第m-1轮迭代之后，向量位于

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084555.png)


如果我们想进一步减小损失函数，则根据梯度下降法，向量移动的方向应为损失函数的负梯度方向，即：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084603.png)


这样如果使用训练集：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084612.png)


去训练一棵树的话，相当于朝着损失函数减小的方向又走了一步（当然在实际应用中需要shrinkage，也就是考虑学习率）。由此可见，GBDT在本质上还是梯度下降法，每一步通过学习一棵拟合负梯度（也就是所谓“残差”）的树，来使损失函数逐渐减小

### 2.2.2 二分类问题

将GBDT应用于回归问题，回归问题的损失函数一般为平方差损失函数，这时的残差，恰好等于预测值与实际值之间的差值。每次拿一棵决策树去拟合这个差值，使得残差越来越小，这个过程还是比较直观的（intuitive）。而将GBDT应用于分类问题就不是很明显。

类似于逻辑回归、FM模型用于分类问题，其实是在用一个线性模型或者交叉项的非线性模型，去拟合所谓的对数几率

$$
\ln \frac{p}{1-p}
$$

而GBDT也是一样，只是用一系列的梯度提升树去拟合这个对数几率，实际上最终得到的是一系列CART回归树。其分类模型可表达为

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084619.png)


其中hm(x)就是学习到的决策树。

清楚这一点后，我们便可以参考逻辑回归的损失函数：交叉熵

$$
loss(x_i,y_i)=-y_i\log \hat{y_i}-(1-y_i)\log(1-\hat{y_i})
$$

假设第K步迭代之后当前学习器为



将hat(y)的表达式带入后，可将损失函数表示为：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084631.png)


可以求得损失函数相对于当前学习器的负梯度为：



可以看到，这里和回归问题很类似，下一棵决策树的训练样本为：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084644.png)


其所需要的拟合的残差为真实标签与预测概率之差。于是便有了下面GBDT应用于二分类的算法：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084651.png)


## 2.3 多分类问题

类似地，对于多分类问题，则需要考虑以下softmax模型：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084701.png)


其中F1，Fk是k个不同的tree ensemble。每一轮的训练实际上是训练了k棵树去拟合softmax的每一个分支模型的负梯度。softmax模型的单样本损失函数为：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084707.png)


这里的yi(i=1,2,3,…,k)是样本label在k个类别上做one-hot编码之后的取值，只有一维为1，其余都是0。由以上表达式不难推导：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084714.png)


可见，这k棵树同样是拟合了样本的真实标签和预测概率之差，与二分类的过程非常类似

举例：

第一步：训练的时候，是针对样本X每个可能的类都训练一个分类回归树。如目前的训练集共有3类，即K=3，样本x属于第二类，那么针对样本x的分类结果，我们可以用一个三维向量

第一步：训练的时候，是针对样本 X 每个可能的类都训练一个分类回归树。如目前的训练集共有三类，即K = 3，样本x属于第二类，那么针对样本x的分类结果，我们可以用一个三维向量【0,1,0]来表示，0表示不属于该类，1表示属于该类，由于样本已经属于第二类了，所以第二类对应的向量维度为1，其他位置为0。

 针对样本有三类的情况，我们实质上是在每轮的训练的时候是同时训练三颗树。第一颗树针对样本x的第一类，输入是(x,0)，第二颗树针对样本x的第二类，输入是(x,1)，第三颗树针对样本x的第三类，输入是(x,0)。

在对样本x训练后产生三颗树，对x 类别的预测值分别是f1(x),f2(x),f3(x)，那么在此类训练中，样本x属于第一类，第二类，第三类的概率分别是：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084722.png)


然后可以求出针对第一类，第二类，第三类的残差分别是：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084728.png)


然后开始第二轮训练，针对第一类输入为(x,y11)，针对第二类输入为(x,y22)，针对第三类输入为(x,y33)，继续训练出三颗树。一直迭代M轮，每轮构建三棵树

当训练完毕以后，新来一个样本x1，我们需要预测该样本的类别的时候，便产生三个值F1(x),F2(x),F3(x)，则样本属于某个类别c的概率为：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084733.png)


## 2.4 叶子节点分值计算

对于生成的决策树，计算各个叶子节点的最佳残差拟合值为：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084740.png)


我们一般使用近似值代替：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084749.png)


![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084806.png)


GBDT二分类每轮迭代，换一种数学符号表达方式：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084820.png)


GBDT多分类每轮迭代：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301084828.png)


参考文章：

[https://zhuanlan.zhihu.com/p/144855223](https://zhuanlan.zhihu.com/p/144855223)

[https://blog.csdn.net/ShowMeAI/article/details/123402422](https://blog.csdn.net/ShowMeAI/article/details/123402422)