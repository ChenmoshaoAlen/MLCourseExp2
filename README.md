# Experiment2
[机器学习实验二](https://msh8so3dvj.feishu.cn/docx/LLo5dloSooJWQxxg2BzcPgFUnje?from=from_copylink)

# SVM实验
### 第一个函数：`hinge_loss`
此函数计算的是铰链损失（Hinge Loss）加上正则项的加权和，用于支持向量机（SVM）中。具体公式如下：

$$
L(\theta) = t \times \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i(x_i \cdot \theta)) + \frac{1}{2} \sum_{j=1}^{m} \theta_j^2
$$

这里：
- $L(\theta)$ 是总损失，
- $x_i$ 是特征向量，
- $y_i$ 是标签值（通常为+1或-1），
- $\theta$ 是模型参数，
- $t$ 是正则化参数前的乘子，控制损失项和正则项的相对重要性，
- $n$ 是样本数量，
- $m$ 是特征数量。
正则项是参数的平方和，用于防止模型过拟合。

### 第二个函数：`hinge_gradient`
此函数计算铰链损失对参数 $\theta$ 的梯度，用于梯度下降法中。具体公式如下：

$$
\nabla L(\theta) = \theta - C \times \frac{1}{n} \sum_{i=1}^{n} x_i y_i \, \text{其中} \, x_i \, \text{的} \, y_i(x_i \cdot \theta) \geq 1 \, \text{被置为} \, 0
$$

此外，对于偏置项（通常是 $\theta$ 的最后一个元素），不进行正则化，因此在计算梯度后需要从该梯度中减去 $\theta$ 的最后一个元素。

这里，$C$ 是一个正则化参数，控制着模型的复杂度和拟合程度。较大的 $C$ 值导致模型更加努力地拟合所有数据点（可能导致过拟合），而较小的 $C$ 值会让模型更加平滑。