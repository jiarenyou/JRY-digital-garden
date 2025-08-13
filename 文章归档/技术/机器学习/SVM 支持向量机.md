

#数据科学  #机器学习
# 一 支持向量机简介
[[SVM 支持向量机]]

支持向量机（Support Vector Machine， SVM）是一个<font color="red">二元分类算法</font>，是对感知器算法模型的一种拓展，现在的SVM算法支持线性分类和非线性分类应用，并且能够直接将SVM应用于回归应用中，通过OvR或者OvO的方式也可以将SVM应用在多分类领域中。在不考虑集成学习算法，不考虑特定的数据集的时候，在分类算法中SVM是非常优秀的

SVM的基本思想是：<font color="red">找到集合边缘上若干数据（称为支持向量（Support Vector）），用这些点找出一个平面（称为决策面），使得支持向量到该平面的距离最大</font>

## 1.1 适用范围

1. <font color="red">二分类问题</font>
2. <font color="red">高纬空间问题</font>：SVM在高纬空间中变现出色，当特征维度较高时，SVM可以更好地处理数据，避免维度灾难
3. <font color="red">小样本问题</font>：当训练样本数量较少时，SVM可以避免过拟合问题，具有较好的泛化能力
4. <font color="red">非线性问题</font>：通过核函数将输入空间映射到高维特征空间，从而可以处理非线性问题
5. <font color="red">异常检测</font>：SVM可以用于异常检测，通过找到与其它样本差异较大的样本点，可以有效地识别异常数据

## 1.2 SVM模型分类— 线性可分、线性、非线性

- 当训练样本<font color="red">线性可分</font>时，通过<font color="#ffff00">硬间隔最大化</font>，学习一个<font color="#ffff00">线性可分支持向量机</font>
- 当训练样本<font color="#ffff00">近似线性可分</font>时，通过<font color="#ffff00">软间隔最大化</font>，学习一个<font color="#ffff00">线性支持向量机</font>
- 当训练样本<font color="#ffff00">线性不可分</font>时，通过<font color="#ffff00">核函数和软间隔最大化</font>，学习一个<font color="#ffff00">非线性支持向量机</font>

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080122.png)

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080139.png)

# 二 感知机

理解SVM之前先了解一下感知机，感知机算法是最古老的分类算法之一，原理比较简单，不过模型的泛化能力弱，不过感知器模型是 SVM、神经网络、深度学习等算法的基础。感知机的思想很简单：在任意空间中，感知机模型寻找一个超平面，能够把所有的二分类别分割开。感知机模型的前提是：数据是线性可分的。

对于m个样本，每个样本有n维特征，以及二元类别输出y，感知机的目标是找到一个超平面
![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080156.png)

让一个类别的样本满足：θx>0；让另一个类别满足：θx<0

所以模型为：
![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080211.png)



正确分类：y*θx>0，错误分类：y*θx<0；所以我们可以定义我们的损失函数为：期望
使分类错误的所有样本到超平面的距离之和最小。

## 2.1 感知机与逻辑回归的损失函数比较

感知机的损失函数是**计算错误分类下的样本到超平面的距离之和，期望使其最小**

**逻辑回归的几何意义也是在向量空间中寻找合适的超平面，但是损失函数不一样，逻辑回归是超平面一侧为正，另一侧为负，根据sigmod函数将分数映射到0-1之间，通过最大似然估计来赋予概率意义**

## 2.2 几何距离与函数距离

感知机的损失函数是计算错误分类样本到超平面的距离，这距离如何算呢？

高中知识，点到直线的距离：点A（xi, yi）到直线ax+by+c=0的距离为
![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080225.png)


推广到高维空间中，任意一个点x0其对应标签为y0，到某平面![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080239.png)

的距离为：![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080252.png)

其中||w||为l2范数。

对那些正确分类的点，y0必然与wx0+b同号，所以可以将距离公式表示为![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080304.png)

称这个距离为某点到平面的几何距离，分子部分y0（wtx0+b）为某点到平面的函数距离

# 三 损失函数

感知机损失函数的定义：期望使分类错误的样本到超平面的距离之和最小

对于那些分类错误的点，由上面的距离公式可知，y0与wtx0+b符号是相反的，要想使损失函数大于0，那些错误分类的点距离在前面加个符号，即：![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080319.png)

因为此时分子和分母中都包含θ，当分子扩大N倍，分母也会随之扩大，所以可以固定分子或者分母为1，然后求另一个分子或者分母的倒数的最小化，作为损失函数，简化后为
![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080345.png)


直接使用梯度下降法可以对损失函数求解，不过由于这里的m是错误分类的样本，是不固定的，所以我们不能使用批量梯度下降法（BGD）求解，只能使用随机梯度下降（SGD）或者小批量梯度下降（MBGD）；一般在感知机模型中使用SGD来求解。![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080358.png)

# 四 SVM算法思想

## 4.1与感知机对比

- 相同的地方
    
    SVM也是通过寻找超平面，解决二分类问题的分类算法
    
    超平面一侧为负例，一侧为正例
    
    与感知吉祥通，通过sign给出预测标签，正例为+1， 负例为-1，模式判别式一样
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080417.png)

    

- 不同的地方
    
    损失函数与感知机和逻辑回归的都不同
    
    **感知机是通过判断错的点寻找超平面，逻辑回归是通过最大似然估计寻找超平面，SVM是通过支持向量寻找超平面，这也是SVM这个名字的由来**
    
    感知机和逻辑回归是直接最小化损失函数得到的θ，或者叫W和b，SVM有两种求解方式，一种是直接最小化损失函数求θ，另一种寻找支持向量，找到支持向量超平面就自然找到了
    

## 4.2 高级的地方

感知机可以将可分的样本给分开来，但能给分开的超平面有很多，如何找到一个最好的超平面，这就是SVM要解决的问题

在感知机模型中，我们可以找到对个可以分类的超平面把数据分开，并且优化所有的点都离超平面尽可能的元，但是实际上离超平面足够远的点基本上都是被正确分类的，反而那些离超平面很近的点，是比较容易分错的![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080428.png)


假设未来拿到的数据含有一部分噪声，那么不同的超平面对于噪声的容忍度是不同的，显然最右边的图是最robust（健壮性）的

换一种角度看，其实是找到最宽的超平面

SVM是找到距离超平面近的点使其远离超平面。就是在支持向量机中，距离超平面最近的且满足一定条件的几个训练样本点被称为支持向量

# 五 SVM支持向量机

**中心思想是：能正确分类的条件下，距离最近的点越远越好**

由第一节知道SVM分为三类：线性可分支持向量机（硬间隔最大化解决）；线性支持向量机（软间隔最大化）；非线性支持向量机（升维、核函数）

- 线性可分（Linearly Separable）
    
    在数据集中，如果数据可找到一个超平面，将其分开，那么这个数据叫做线性可分数据
    
- 线性不可分（Learn Inseparable）
    
    在数据集中，没法找到一个超平面将两组数据分开， 那么数据就叫做线性不可分数据
    
- 间隔（Margin）
    
    数据点到分割超平面的距离成为间隔
    
- 支持向量（Support Vector）
    
    离分割超平面最近的那些点叫做支持向量
    

## 5.1 线性可分支持向量机

需要找到一个超平面，实现硬间隔最大化，要求：

1. 能够完美分类正负例
2. 距离最近的点越远越好

如何确定超平面：

1. y = sign(wT x+b)
2. wT x+b=0表示分割超平面
3. **只要确认了w和b就确认了分割超平面**

<aside>
💡 所以我们的目标是找到一组最好的w和b固定一个超平面，是这个超平面能完美区分正负例的基础上，距离最近的点间隔最大

</aside>

### **目标**

转换为有约束的函数最优化问题：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080445.png)


其中γ’代表支持向量的函数距离

### **简化目标**

一组w，b只能固定一个超平面

一个超平面对应无数多个w，b，只要找到其中任意一个w符合条件的w就可以了

选择最好求的，令γ’=1 则原最优化问题为：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080456.png)


![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080509.png)


等价于，求整体最大，分子固定为1，则分母越小，整体越大，`**问题是为何为1/2**`

为何是1/2，因为求导后可以将2给约掉

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080519.png)


### 函数最优化问题

给定一个函数f(x)找到一个x使得f(x)最小，一般使用梯度下降、L-BFGS 、SMO算法解决函数最优化问题

最优化问题分为**有约束条件的最优化问题**和**无约束条件的最优化问题**

- **无约束条件的最优化问题**

$$
f(x)=min_{x\in R^{n}}f(x)
$$

      上公式对x的取值范围没有做任何限制，所以成文无约束条件的最优化问题

- **有约束条件的最优化问题**
    
    例子：
    
    $$
    f(x)=4x^2+5x+10
    $$
    
    最优化问题就是求得x使f(x)最小，结果为x=-5/8
    
    假定给定约束条件x≥0，则x=0的时候去的最小值
    
    **那么如何通过一种方法求解这种带约束条件的函数最优化问题？**
    
- **原始问题**
    
    为了解决这种带约束条件的函数最优化问题，我们定义带约束条件的最优化问题泛化表示方法：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080530.png)

    
    可以将约束条件表述为k个不等式约束条件，和L个等式约束条件
    
    我们命名其为原始最优化问题
    
- **拉格朗日函数**
    
    <aside>
    💡 **拉格朗日函数**
    
    拉格朗日函数是一种在数学和物理中常用的工具，它用于求解约束条件下的最优化问题。拉格朗日函数是由法国数学家约瑟夫·路易·拉格朗日在18世纪提出的。
    
    拉格朗日函数的引出是为了解决约束条件下的优化问题，即在一定的约束条件下，找到使得目标函数取得最大或最小值的变量取值。这类问题常见于经济学、物理学、工程学等领域。
    
    拉格朗日函数的基本思想是**将约束条件引入目标函数中，通过引入拉格朗日乘子来将约束条件转化为目标函数的一部分。这样，原问题就可以转化为一个无约束的优化问题，通过求解该无约束问题的驻点，即可得到原问题的最优解**。
    
    总之，拉格朗日函数的出现解决了约束条件下的最优化问题，并为解决这类问题提供了一种有效的数学工具。
    
    </aside>
    
    定义某原始最优化问题的拉格朗日函数为：
    

    
    其中ci是第i个不等式约束函数（需要整理），hj是第j个等式约束函数
    
    αi和βi是拉格朗日乘子
    
- **拉个朗日函数的特性**
    
    令
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080737.png)

    
    若x不满足之前的约束条件：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080751.png)

    
    若x满足约束条件：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080801.png)

    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080810.png)

    
- **对偶问题**
    
    通过对偶，可以先将x用α、β来表示，进而求出α和β
    
    对于一个优化问题，我们希望：
    
    （1）当原问题非凸时能够找到一种办法将其转化为凸优化问题
    
    （2）给出原问题的解的下界，并用此下界逼近[原函数](https://www.zhihu.com/search?q=%E5%8E%9F%E5%87%BD%E6%95%B0&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A2444538449%7D)的最小值
    
    为此，我们定义一个对偶函数：
    
    定义
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080819.png)

    
    此时极大化θD
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080828.png)

    
    称为拉格朗日的极大极小问题，也称为原始问题的对偶问题
    
    设
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080837.png)

    
    为对偶最优化问题的最优解
    
    当f(x)和ci函数为凸函数，hj函数为仿射函数时，有：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080851.png)

    
- **如何求解**
    
    KKT条件：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080903.png)

    

### 求解最优化问题

对于原始问题：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080913.png)


我们构造拉格朗日函数：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080923.png)


可将原始有约束的最优化问题转换为对拉格朗日函数进行无约束的最优化问题（也叫二次规划问题）

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080931.png)


由于我们的原始问题满足f(x)为凸函数，那么可以将原始问题的极小极大优化转换为对偶函数的极大极小优化进行求解：

对于原始问题：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080939.png)


对偶函数为：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221080947.png)


下面就开始对求解对偶函数的第一步

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081000.png)


- 第一步求极小
    
    对拉格朗日函数分别求w和b的偏导：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081011.png)

    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081021.png)

    
    可以看出：我们已经求得了w和α的关系，下一步将w反代回原来的拉格朗日函数中就可以进行第二步求关于α的极大值
    
- 反代
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081033.png)![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081050.png)

    
- 整理对偶函数
    
    对偶函数的优化问题
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081059.png)

    
    去掉负号转换为求极小问题：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081111.png)

    
    解决此问题，SVM的学习问题就完成了，通常使用**SMO算法进行求解**，可以求得一组α*是的函数最优化
    
- 求得最终的超平面
    
    假设我们使用SMO算法，求w*很容易得：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081121.png)

    
    b*如何求？ 对于任意支持向量，有：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081128.png)

    
    如何找到支持向量？根据KKT条件有：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081139.png)

    
    那么所有α>0时后边的一项需要=0也就是
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081147.png)

    
    求b的过程：找到所有支持向量带进去求出所有b，然后求平均
    
    这样就得到分割超平面
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081155.png)

    

### 硬分割SVM总结

流程

1. 原始目标：求得一组w和b使得分割margin最大
2. 转换目标：通过拉格朗日函数构造目标函数，问题由求得n个w和1个b转换为求得m个α
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081203.png)

    
3. 利用smo算法求得m个α*
4. 利用求得的m个α*求得w*和b*
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081211.png)

    

## 5.2 线性支持向量机

### 硬间隔面临的问题

有些时候，线性不可分是由噪声点决定的

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081219.png)


### 软间隔SVM

对于之前的讲述的线性可分svm可通过构造超平面令硬间隔最大化，从而求得最好的分隔超平面

条件：

1. 正负例完美分开（体现在约束条件≥1上）
2. 找到能使间隔最大的点（有约束条件的函数优化问题）

如果数据集线性不可分，意味着找不到一个合格的超平面

体现在优化问题上，任何的w和b都无法满足优化条件

### 引入松弛变量

对于之前问题，硬间隔不可分，体现在满足不了约束条件上，所以引入松弛变量ξi≥0（每
个数据点自己有一个ξi）

我们将约束条件放松为：

$$
y_{i}(w*x_{i}+b)>= 1-\xi_{i}
$$

这样就至少肯定有好多的w和b满足条件了，但是这相当于没有约束条件了，只要ξi无穷大，那么所有w和b都满足条件

**ξ代表异常点嵌入间隔面的深度，我们要在能选出符合约束条件的最好的w和b的同时，让嵌入间隔面的总深度越少越好**

### 目标函数优化

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081228.png)


1. 根据f(x)和约束条件构造拉格朗日函数
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081236.png)

    
    其中要求μi和αi≥0
    
2. 优化原始问题：**构造拉格朗日函数，将约束条件加入到其中，转换为凸优化问题**
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081243.png)

    
3. 对偶问题
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081250.png)

    
    先求L函数对w，b，ξ的极小值，再求其对α和μ的极大值
    

### 对偶问题求解，解决极小问题

对3个参数分别求偏导得到的一定的信息，反带回拉格朗日函数

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081259.png)


回带拉格朗日函数

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081309.png)![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081323.png)

### 整理约束条件解决极大问题

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081339.png)


与之前的目标函数一模一样，只不过约束条件不同了

由于目标函数中没有出现C，可将约束条件的第2,3,4项合并，消去C，得到最终的待优化函数为：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081349.png)


与之前相比，知识多了个约束条件而已，仍然可以使用SMO来求解

### 分析软间隔问题的支持向量

结论：

- αi=0 —>该点为分类正确的点
- 0<αi<C—>该点为软边界上的点
- αi=C—>该点嵌入了软边界内
    - 此时如果ξ<1 ，该点被正确分类
    - 此时如果ξ=1 ，该点刚好落在超平面上
    - 此时如果ξ>1 , 该点被错误分类

### 总结软间隔最大化算法

1. 设定惩罚系数C，构造最优化问题

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081358.png)


1. 用SMO算法求出α*
2. 计算

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081406.png)


3. 找到全部的支持向量，计算出

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081417.png)


4. 计算所有的b*s的平均值得到最终的

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081430.png)


### 判别函数的另一种表达方式

对于线性SVM来说，判别函数为

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081441.png)


由于

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081448.png)


所以也有下面这种判别函数的形式：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081455.png)


我们得到一个很好的结论，每一次在计算判别函数结果时需要求得待判断点和所有训练集样本点的内积

## 5.3 非线性支持向量机

非线性不可分问题是如何处理的？SVM是采用升维，或者使用核函数将非线性问题转化为线性可分

### SVM升维

对于线性SVM来说，最优化问题为：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081506.png)


如果使用φ(x)对训练集升维，最优化问题就变成了

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081513.png)


![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081523.png)


### 升维带来的问题-维度爆炸

看似这种升维方式完美解决了线性不可分问题，但是带来了一个新问题

假设就使用多项式回归的方式进行升维：对于二维x1，x2升维后的结果是：x1，x2，x1*x2，x1^2，x2^2。如果是四维，五维…，可想而知，时间和空间的消耗太大

### 引入核函数

我们发现在SVM学习过程中

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081532.png)


只需要求得φ(xi)·φ(xj)的结果，并不需要知道具体的φ(x)是什么

于是先驱们决定，跳过φ(x)直接定义φ(xi)·φ(xj)的结果，这样既可以达到升维的效果，又可以避免维度爆炸的问题

所以，定义

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081541.png)


此时，对偶问题的目标函数变为了：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081549.png)


判别函数变为

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081556.png)


### 常用核函数

5. 线性核函数

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081602.png)


6. 多项式核函数
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081609.png)

    
7. 高斯核函数
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081616.png)

    
8. sigmod核函数
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081623.png)

    

## 5.4 SVM算法流程总结

9. 选择某个核函数及其对应的超参数
10. 选择惩罚系数C
11. 构造最优化问题
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081632.png)

    
12. 利用SMO算法求解出一组α*
13. 根据α*计算w*
14. 根据α*找到全部支持向量，计算每个支持向量对应的bs*
15. 对bs*求均值得到最后的b*

学得的超平面为：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081640.png)


最终的判别函数为：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081647.png)


# 六 SMO算法

## 6.1 SMO思路

回顾下我们要解决的问题，在将SVM原始问题转换为对偶问题之后，我们先求得w和b的值，带回到原式中并化简，得到了如下的最优化问题：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081655.png)


可以看到，共有N个决策变量需要处理，每两个决策变量还会以乘积的形式出现在目标函数中，那么这个问题如何求解，就要用到SMO算法

其中(xi, yi)表示训练样本数据，xi 为样本特征，yi∈{−1,1}为样本标签，C 为惩罚系数由自己设定。上述问题是要求解 N 个参数(α1,α2,α3,...,αN)，其他参数均为已知，有多种算法可以对上述问题求解，但是算法复杂度均很大。但 1998 年，由 Platt 提出的序列最小最优化算法(SMO)可以高效的求解上述 SVM 问题，它把原始求解 N 个参数二次规划问题分解成很多个子二次规划问题分别求解，每个子问题只需要求解2个参数，方法类似于坐标上升，节省时间成本和降低了内存需求。每次启发式选择两个变量进行优化，不断循环，直到达到函数最优值。

概括说，SMO算法主要分为以下两步：

16. 选择接下来要更新的一对αi和αj，采用启发式的方法进行选择， 以使目标函数最大程度地接近其全局最优值
17. 将目标函数对αi和αj进行优化，保持其它所有的αk（k<> i, j）不变

### 6.2 视为一个二元函数

暂时略过，有些难

# 七 SVM概率化输出

## 7.1 SVM合页损失

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081705.png)


- hinge loss function
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081712.png)

    
    下标“+”表示以下取正值的函数，我们用z表示中括号中的部分：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081718.png)

    
    也就是说，如果数据分类正确，没有损失，如果分错，损失为z。
    
    合页损失函数如下图所示：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081727.png)

    
    SVM的损失函数就是合页损失函数加上正则项， 即：
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081734.png)

    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081743.png)

    

# 八 SVM多分类

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250221081751.png)


### one-versus-the-rest

对于K个类别的问题，在训练样本上，采用SVM训练出K个分类器，每个分类器将训练样本分成Ki类和非Ki类，然后采用SVM训练出模型。如上图所示，**每个分类器仅仅能回答是否属于Ki的答案。此方法会造成一个样本数据属于多个类别的情况**，上左图阴影部分

也能够採用：y(x)=maxk yk(x)，即採用最大的函数间隔的那个类别。但不同的分类器有可
能尺度不同样，函数距离自然不能作为推断标准。
同一时候，训练样本的不平衡也可能造成分类器有误差。

### one-versus-one

在 K 分类的情况下，训练出 K(K−1)2 个分类器，即每两个类别训练出一个分类器，然后依
据 K(K−1)2 个分类器的结果，採用投票方法给出预測结果。
**此种方法依旧造成部分数据不属于任何类的问题，上右图阴影部分所看到的。**

# 九 SVM算法小结

SVM算法是一个优秀的算法，在集成学习和神经网络之类的算法没有表现出优越性能之前，SVM算法基本占据了分类模型的统治地位。目前在大数据时代的大样本背景下，SVM由于其在大样本时超级大的计算量，热度有所下降，但仍然是一个常用的机器学习算法。

### 优点

18. 解决高维特征的分类问题和回归问题很有效，在特征维度大于样本数时依然有很好的效果
19. 仅仅使用一部分支持向量来做超平面的决策，无需依赖全部数据
20. 有大量的核函数可以使用，从而可以很灵活的来解决各种非线性的分类回归问题
21. 样本量不是海量数据的时候，分类准确率高，泛化能力强

### 缺点

22. 如果特征维度远远大于样本数，则SVM表现一般
23. SVM在样本量非常大，核函数映射维度非常高时，计算量过大，不太适合使用
24. 非线性问题的核函数的选择没有通用标准，难以选择一个合适的核函数
25. SVM对缺失数据敏感

### SVM对比逻辑回归

26. LR采用log损失，SVM采用合页损失
27. LR对异常值敏感，SVM对异常值不敏感
28. 在训练集较小时，SVM较使用，而LR需要较多的样本
29. LR 模型找到的那个超平面，是尽量让所有点都远离它，而 SVM 寻找的那个超平面，是
只让最靠近中间分割线的那些点尽量远离，即只用到那些支持向量的样本。
30. 对非线性问题的处理方式不同，LR 主要靠特征构造，必须组合交叉特征，特征离散化。
SVM 也可以这样，还可以通过 kernel。
31. SVM 更多的属于非参数模型，而 logistic regression 是参数模型，本质不同。其区别
就可以参考参数模型和非参模型的区别

### 如何选择

那怎么根据特征数量和样本量来选择 SVM 和 LR 模型呢？Andrew NG 的课程中给出了以
下建议：
- 如果 Feature 的数量很大，跟样本数量差不多，这时候选用 LR 或者是 Linear Kernel 的 SVM
- 如果 Feature 的数量比较小，样本数量一般，不算大也不算小，选用 SVM+Gaussian Kernel
- 如果 Feature 的数量比较小，而样本数量很多，需要手工添加一些 feature 变成第一种情况。
(LR 和不带核函数的 SVM 比较类似。)