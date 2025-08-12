

**想要了解AUC，需要先了解混淆矩阵，然后是ROC，再了解AUC**

## 混淆矩阵

先介绍下混淆矩阵

|         | actual | 真实值 |     |
| ------- | ------ | --- | --- |
| predict |        | 1   | 0   |
| 预测值     | 1      | TP  | FP  |
|         | 0      | FN  | TN  |

recall = TP/(TP+FN)

precision = TP / (TP+FP)

acc = （TP+TN）/ (TP+FN+TF+FP)

TPR（真阳性率）=TP/TP+FN

FPR（假阳性率）=FP/FP+TN

F1

$$F1 = \frac{2*precision*recall}{precesion + recall} $$

## ROC

ROC曲线（受试者工作曲线）就是横轴是FPR，纵轴是TPR的样本点组成的曲线，当取无穷多个点后，ROC Curve越平滑。

### ROC计算

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301082553.png)


假设有20个样本，其中正样本为10例，负样本为10例

1. 首先，将20个样本按照prediction score从大到小排序，即上图表
2. 选择threshold使得i个样本被预测为正样本

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301082600.png)


![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301082607.png)


ROC计算表格：

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301082617.png)


随着点数的越来越多，曲线会越来越平滑

### ROC曲线理解

![image.png](https://build-web.oss-cn-qingdao.aliyuncs.com/my_pic_file/20250301082625.png)


roc的理想点是（0,1），也就是TPR越大，FPR越小则模型的性能越好，红色的虚线是随机猜的ROC曲线，红色虚线下方的模型是无效的。

## AUC

AUC（Area Under ROC Curve）：指ROC曲线下方的面积

**auc的意义**

1. auc只反应模型对正负样本排序能力强弱，对score的大小和精度没有要求
2. auc越高模型的排序能力越强，理论上，当模型把所有正样本排在负样本之前时，auc为1.0，是理论最大值。