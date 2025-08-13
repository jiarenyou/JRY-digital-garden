#机器学习
# 一 简介

XGBoost（eXtreme Gradient Boosting）极致梯度提升，是一种基于GBDT的算法

XGBoost的基本思想和GBDT相同，但是做了一些优化，比如二阶导数使得损失函数更加精准，正则项避免树过拟合，Block存储可以并行计算。

XGBoost具有高效、灵活和轻便的特点，在数据挖掘、推荐系统等领域有广泛应用

## 1.1 应用领域
1. 分类问题
	如信用评分，垃圾邮件检测，图像分类等
2. 回归问题
	房价预测、销售预测等
1. 排序问题
	搜索引擎结果排序，推荐系统等
2. 特征选择
	通过模型的重要性评估，帮助选择最相关的特征


## 1.2 XGBoost相对于GBDT的优点

对比GBDT，XGBoost做了三个方面的优化

1. **算法本身的优化（重点）**
    1. 在算法的弱学习器模型选择上，对比GBDT只支持决策树，而**XGBoost还支持很多其他学习器**
    2. 在损失函数上，在误差的基础上**增加了复杂度的衡量（即正则项）**
    3. 在算法优化方式上，GBDT的损失函数是对误差做一阶泰勒展开，而XGBoost是**做二阶泰勒展开，更加准确。**
2. 算法运行效率的优化
    
    **对每个弱学习器**，比如决策树建立的过程**做并行选择**，找到合适的子树分裂特征和特征值。在并行选择之前，先对所有的**特征值进行排序分组**，方便前面说的并行选择。对分组的特征，选择合适的分组大小，使用CPU缓存进行读取加速。将各个分组保存到多个硬盘以提高IO速度
    
3. 算法健壮性的优化
    
    **对缺失值的特征，通过枚举所有缺失值在当前节点是进入左子树还是右子树来决定缺失值的处理方式。算法本身加入了L1和L2正则项，可以防止过拟合，泛化能力增强**
    
## 1.3 调参

1. **学习率（eta）**：控制每棵树对最终预测的贡献，通常在0.01到0.3之间。
2. **树的深度（max_depth）**：控制树的复杂度，防止过拟合，通常在3到10之间。
3. **子样本比例（subsample）**：每棵树使用的样本比例，通常在0.5到1之间。
4. **列采样比例（colsample_bytree）**：每棵树使用的特征比例，通常在0.5到1之间。
5. **正则化参数（lambda和alpha）**：控制模型的复杂度，防止过拟合。
6. **迭代次数（n_estimators）**：树的数量，通常需要通过交叉验证来确定最佳值。


# 二  回顾有监督学习概念

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301082831.png)


![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301082904.png)


> **L1和L2的区别：**
L1：曼哈顿距离（绝对值）；可以使不少w权值变为0，产生稀疏解，使模型更具鲁棒性，因为忽视不重要的特征，保留重要的
L2：欧氏距离（平方开根号）；不会使w权值完全变为0，但会使其变的很小，也就是会使模型更加平滑
> 

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301082915.png)


![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301082928.png)


L(θ)和Ω(θ)两者不可兼得，要找的是一个平衡点

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301082949.png)


模型复杂度越大，偏差越小，产生过拟合，模型复杂度越低，方差越低，就会欠拟合

XGBoost相比于GBDT，把模型复杂度考虑进去，即Ω(x)

# 三 回归树和集成学习

## 3.1 回归树（Regression Tree）

### 回归树，即CART算法

- 在决策树里面学习一些决策的规则，也就是树的结构
- 计算叶子节点的分值

在GBDT中，

- 因为是回归树，所以通过MSE来计算树的结构，作为分裂的指标。
- 叶子节点分值的计算因为回归、二分类、多分类的不同，有不同的公式来计算叶子节点分值。
- 通过负梯度来拟合回归树

在XGBoost中，

- 拟合什么，—拟合梯度
- 叶子节点分值如何计算

这都是在后边的过程中需要学习的

### 回归树集成计算

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301082959.png)


### 树集成的方法

1. 使用广泛，比如GBM，随机森林
2. 使用方便，输入数据不需要缩放，不用很担心特征归一化的问题
3. 寻找特征组合，不需要之前的进行升维度
4. 大规模使用，应用到工业，数据可以并行，比如GBDT解决多分类问题时，不同的类别形成一个独立的管道独自训练，使得速度更快

### 输入的模型和参数

模型

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083008.png)


假设我们有K棵树，每棵回归树要做的就是把特征x映射为一个分值y（为甚么说是分值呢，因为对于回归问题可以直接用分值，对于分类问题可以对分值进行非线性变换，适用性更广泛）

参数

- 树的结构，叶子节点分值
- 或者简单地用函数表达参数

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083015.png)


- 和之前的回归问题不同，不再是求w权重，而是树结构f(x)

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083024.png)


- Splitting Positions: 条件（树结构）
- The Height in each segment：叶子节点分值
- 我们到底如何去学呢？ 损失函数和正则项

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083035.png)


我们如何衡量Ω的复杂度，来防止其过拟合

在回归模型中，通过衡量w（w越少越好，w越小越好）来评判，而在集成学习中，我们可以通过评判树结构，叶子节点分值来评判泛化能力

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083045.png)


回归树不止能做回归问题，还可以做分类、排序的问题，处理这些问题取决于我们如何定义目标函数。MSE解决回归问题，交叉熵解决分类问题

> 拓展：
什么是损失函数？什么是激活函数？区别是什么
损失函数：
损失函数是用于衡量模型输出与真实标签之间的差异，帮助优化模型参数，使模型能够更好地拟合实际数据。常用的损失函数：MSE，交叉熵，合页损失（Hinge Loss）等。作用是为了量化模型预测的准确程度，更好地调整模型参数。
激活函数：
用于神经网络中每一层对输入进行非线性变换，以便引入非线性特性。常见的激活函数包括sigmod、Tanh、ReLU、Leaky ReLU等，作用是为了引入非线性，使得神经网络学习和拟合复杂的非线性关系。

**损失函数类似于一个衡量器，激活函数类似于一个转换器。**


### XGBoost的思想

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083053.png)


上图中可以看到XGBoost的思想

- 信息增益用training loss表示
- 剪枝过程用正则项代替
- 最大深度用目标函数增长过程中限制函数空间，控制了最优解范围是多大
- 平滑叶子节点分值，也就是是叶子节点分值变小，XGBoost直接把叶子节点分值当做L2正则项权重放到目标函数中去

也就是把凭直觉做的一些参数，直接设计到目标函数中去

# 四 XGBoost如何去学习

## 4.1 梯度提升

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083104.png)


如何去学习，我们构建好了目标函数，不能用梯度下降的常规方法进行优化，因为这不是参数空间的优化，而是函数空间的优化，所以使用Boosting的思想优化，前一时刻预测的值加上新的f(x)，不断地优化

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083113.png)


如何优化ft(xi)，我们知道

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083120.png)


进而将hat(yi)带入，得到

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083127.png)


我们考虑下损失 L用平方差来表示：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083134.png)


对其展开得到：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083141.png)


除了f(x)是未知的，其余都是已知的，经过化简后得到

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083149.png)


## 4.2 使用泰勒展开

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083155.png)


相比于GBDT，XGBoost用的是二阶泰勒展开，而GBDT用的是一阶泰勒公式。二阶泰勒展开在做迭代的时候会有更快的速度，迭代次数更少，就如同XGBoost用的是牛顿法，而GBDT用的是梯度下降法

*二阶泰勒展开的优点*

1. *更准确的损失函数近似：XGBoost使用二阶泰勒展开来近似损失函数，这使得模型能够更准确地拟合数据，特别是在高维度和非线性数据的情况下。*
2. *更快的训练速度：由于二阶泰勒展开提供了更精确的损失函数近似，XGBoost可以更快地收敛到最优解，从而加快了训练速度。*
3. *更好的泛化能力：通过使用二阶泰勒展开来优化损失函数，XGBoost可以更好地泛化到未见过的数据，从而提高了模型的预测性能。*
4. *更好的处理缺失值：XGBoost能够有效地处理缺失值，这得益于二阶泰勒展开提供的更准确的损失函数近似，使得模型对缺失值更加鲁棒。*

*总的来说，XGBoost使用二阶泰勒展开能够提供更准确、更快速、更具有泛化能力的模型，从而在实际应用中取得更好的效果。*

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083204.png)


## 4.3 重新定义函数

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083212.png)


这里重新定义每次学到的小树ft(x)，ft(x)=wq(x)，q(x)是一个索引函数，意思是输入一个样本，可以查询到所在叶子节点位置，找到了叶子结点位置，就可找到对应的w分值

## 4.4 定义树的复杂度

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083219.png)


γ为超参数，T为叶子节点数量

公式的这两项分别代表了叶子节点数量和叶子结点L2正则项，这两项数越小越好

## 4.5 重新整合

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083230.png)


我们经过泰勒展开、f(x)的重新定义，以及树的复杂度定义，我们将他们整合到一起，进行化简，最终得到上面的式子

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083238.png)


## 4.6 损失函数如有优化求解

单棵树的分值

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083247.png)


![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083253.png)


- 通过枚举出每一种树结构即f(x)，是不可能的，因为树的分割会长成各种各样
- 通过推导出的评分obj公式，对穷举出每棵树计算分值，然后找到分值最低的那棵树，使用wj*计算叶子结点的分值，但这是很理想的树结构，也是不可行的。
    
    ![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083302.png)

    
- 所以使用贪婪学习算法，在训练过程中，一点一点生长，通过计算每次分裂的Gain

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083310.png)


![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083336.png)


第一次分裂和第二次分裂比较，通过计算Gain来评判第二次分裂是否有必要，目的是使得分的准确率更高，也就是使得这个树对应的obj分值更小，所以Gain=obj1-obj2>0，因为收益为0或者小于0就没有必要分了。

γ往往是大于0的超参数，为什么？多分裂一次，树复杂度上升，γ作为惩罚项

## 4.7 剪枝

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083344.png)


从收益的公式中可以看出，包含λ和γ都有，既考虑的树的结构是否简单，叶子结点的分值是否够小

树结构上用到了λ和γ，叶子结点分值用到了λ

## 4.8 回顾

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301083351.png)


# 五 总结

1. 传统的GBDT以CART树作为基学习器，XGBoost还支持线性分类器，这个时候XGBoost相当于L1和L2正则化的logstic回归或者线性回归
2. 传统的GBDT在优化的时候只用到一阶导数信息，XGBoost则对代价函数进行了二阶泰勒展开，得到一阶和二阶导数
3. XGBoost在代价函数中加入了正则项，用于控制模型的复杂度。从权衡方差偏差来看，它降低了模型的方差，使得学习出来的模型更加加单，防止过拟合，这也是XGBoost优于GBDT的一个特性
4. shrinkage（缩减），相当于学习率（XGBoost中的eta），XGBoost在进行完一次迭代时，会将叶子结点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。（GBDT也有学习速率）
5. 列抽样，XGBoost借鉴了随机森林的做法，支持列抽样，不仅防止过拟合，还能减少计算
6. 对缺失值的处理。对于特征的值有缺失的样本，XGBoost还可以自动学习出它的分裂方向，通常情况下，我们人为在处理确实值得时候大多会选用中位数、均值或者二者的融合来对数值型特征进行填补，使用出次数最多的类别来填补缺失的类别特征。在逻辑实现上，为了保证完备性，会分别处理将missing该特征值的样本分配到左叶子结点和右叶子结点的两种情形，计算增益后选择增益大的方向进行分裂即可。可以为缺失值或者指定的值指定分支的默认方向，这能大大提升算法的效率。如果在训练中没有缺失值而在预测中出现缺失，那么会自动将缺失值的划分方向放到右子树。
7. XGBoost工具支持并行。Boosting不是一种串行的结构吗?怎么并行的？注意XGBoost的并行不是tree粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。XGBoost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），XGBoost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。



[[Adaboost]]
