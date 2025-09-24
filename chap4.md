# Single-layer Networks:Regression
***
## targets:
1. 理解线性回归模型
2. 解释**基函数(Basis functions)**的**非线性拟合能力**
3. 理解如何从最大似然估计(Maximum Likelihood)推导**最小二乘法**
4. 区分**批量学习(Batch Learning)**和**序列学习(Sequential Learning)**的优缺点
5. 解释**正则化(Regularization)**的作用，以及它如何**防止过拟合**
6. 理解决策论(Decision Theory)在**回归问题**的作用，即如何从概率分布得到一个最佳预测值
7. 解释**偏执-方差权衡(Bias-Variance Trade-off)**，解释它与**模型复杂度、过拟合**的关系
***
## 1 线性回归模型
回归的目标是根据输入变量x预测一个或多个连续的目标变量t  
最简单的回归模型是线性回归，即假设*输出是输入的线性组合*  

### 1.1 基函数 (Basis Functions)

- **基本线性模型**:
  $$
  y(x, w) = w_0 + w_1x_1 + \dots + w_Dx_D \quad (1.1)
  $$

- **使用基函数的线性模型**: 为了拟合非线性关系，引入一组非线性基函数 $\phi_j(x)$。==模型对参数 $w$ 仍是线性的。==
  $$
  y(x, w) = w_0 + \sum_{j=1}^{M-1} w_j\phi_j(x) \quad (1.2)
  $$

- **向量化表示**: 引入 $\phi_0(x) = 1$ 后，模型可简化为向量内积形式。
  $$
  y(x, w) = \sum_{j=0}^{M-1} w_j\phi_j(x) = w^T\phi(x) \quad (1.3)
  $$

- **常用基函数示例**:
  
  - 高斯基函数:
    $$
    \phi_j(x) = \exp\left\{-\frac{(x-\mu_j)^2}{2s^2}\right\} \quad (1.4)
    $$
  - Sigmoid 基函数 (使用 Logistic Sigmoid 函数 $\sigma(a)$):
    $$
    \phi_j(x) = \sigma\left(\frac{x-\mu_j}{s}\right) \quad (1.5)
    $$
    $$
    \sigma(a) = \frac{1}{1+\exp(-a)} \quad (1.6)
    $$
- 在深度学习中，不再需要手动设计基函数，而是让网格自己分析出合适的特征进行变换。

### 1.2 似然函数与最大似然(Likelihood Functions & Maximum Likelihood)

- **高斯噪声假设**: 假设目标值 $t$ 是模型预测 $y(x,w)$ 加上一个均值为零的高斯噪声 $\epsilon$。
  $$
  t = y(x, w) + \epsilon \quad (1.7)
  $$
  于是，$t$ 的条件概率分布为：
  $$
  p(t|x, w, \sigma^2) = \mathcal{N}(t|y(x, w), \sigma^2) \quad (1.8)
  $$
  式1.8是理解回归问题在概率视角的核心。左边表示在给定输入$x$、模型参数$w$和噪声水平$\sigma^2$的情况下，目标值$t$的概率；右边表示目标值$t$在正态分布的概率值，众所周知一个正态函数由均值和方差定义。这里均值由$y(x,w)$，也就是模型的确定性预测值(对于一个给定的输入$x$和参数$w$，模型会计算得出的具体结果)给出。
  
  式1.8展现了核心思想：==预测=准确值+随机噪声($t=y(x,w)+\epsilon$)==
  
  高斯噪声假设就是假设这个随机误差$\epsilon$服从一个高斯分布，且是无偏的，平均来看它的影响是0。
  
- **对数似然函数**: 对于整个数据集，对数似然函数可以表示为：
  $$
  \ln p(\mathbf{t}|X, w, \sigma^2) = -\frac{N}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=1}^{N}\{t_n - w^T\phi(x_n)\}^2 \quad (1.9)
  $$
  由于平方项恒正，且$ -\frac{N}{2}\ln(2\pi\sigma^2)$为恒值，因此==最大化上式等价于最小化下面的**平方和误差函数**==：
  $$
  E_D(w) = \frac{1}{2}\sum_{n=1}^{N}\{t_n - w^T\phi(x_n)\}^2 \quad (1.10)
  $$

- **最大似然解 (正规方程)**: 误差函数的最小化可以通过求解下面的线性方程组得到，该解被称为**正规方程 (Normal Equations)**。
  $$
  w_{ML} = (\Phi^T\Phi)^{-1}\Phi^T\mathbf{t} \quad (1.11)
  $$
  式1.11中$w_{ML}$下标$ML$表示$Maximum Likelihood$，$t$为一列向量，包含了所有训练数据点的真实目标值，即$t=(t_1,t_2,...,t_N)^T$。
  
   $\Phi$ 是 $N \times M$ 的**设计矩阵 (Design Matrix)**：
  $$
  \Phi = \begin{pmatrix}
  \phi_0(x_1) & \phi_1(x_1) & \cdots & \phi_{M-1}(x_1) \\
  \phi_0(x_2) & \phi_1(x_2) & \cdots & \phi_{M-1}(x_2) \\
  \vdots & \vdots & \ddots & \vdots \\
  \phi_0(x_N) & \phi_1(x_N) & \cdots & \phi_{M-1}(x_N)
  \end{pmatrix} \quad (1.12)
  $$
  正规方程通过一次计算就可以得到最佳的权重$w$。但是当==数据的特征数量(M)==或==样本数量(N)==非常大时，计算会极其缓慢且容易报错。因此在现代机器学习任务中，倾向于使用梯度下降等方法。
  
- **噪声方差的最大似然解**:
  $$
  \sigma_{ML}^2 = \frac{1}{N}\sum_{n=1}^{N}\{t_n - w_{ML}^T\phi(x_n)\}^2 \quad (1.13)
  $$
  在找到了最优权重$w_ML$之后，我们还想知道该模型预测的不确定性大小，或者说数据在模型预测点周围的分散程度是怎么样的，因此需要考虑$\sigma_{ML}^2$。
***
### 1.3 序列学习 (Sequential Learning)

- **随机梯度下降 (Stochastic Gradient Descent, SGD)**: 每次使用单个数据点更新参数，适用于大规模数据集。
  $$
  w^{(\tau+1)} = w^{(\tau)} - \eta\nabla E_n \quad (1.14)
  $$
  解读这个公式1.14：
  
  - $w^{(\tau)}$是模型在第$\tau$步时的权重，也就是当前的”猜测“。
  - $w^{(\tau+1)}$是下一步的权重，也就是更好一些的"猜测"。
  - $\nabla E_n$是误差函数$E$对权重$w$的梯度，下标$n$仅仅表示是根据第n个数据计算得出。==这个梯度会指向误差增长最快的方向。==
  - 
  
  对于平方和误差，这被称为**LMS算法**:
  $$
  w^{(\tau+1)} = w^{(\tau)} + \eta(t_n - w^{(\tau)T}\phi_n)\phi_n \quad (1.15)
  $$

### 1.4 正则化最小二乘 (Regularized Least Squares)

- **带正则化的误差函数**: 在原误差函数上增加一个惩罚项，以防止过拟合。
  $$
  \frac{1}{2}\sum_{n=1}^{N}\{t_n - w^T\phi(x_n)\}^2 + \frac{\lambda}{2}w^Tw \quad (4.26)
  $$

- **正则化解**: 该误差函数同样有解析解。
  $$
  w = (\lambda I + \Phi^T\Phi)^{-1}\Phi^T\mathbf{t} \quad (4.27)
  $$

---

## 2 决策论 (Decision Theory)

- **目标**: 将概率预测 $p(t|x)$ 转换为一个具体的点预测 $f(x)$。
- **损失函数**: 衡量预测 $f(x)$ 与真实值 $t$ 之间差异的函数，例如平方损失 $L(t, f(x)) = \{f(x) - t\}^2$。
- **期望损失**: 我们的目标是最小化所有数据上的平均损失。
  $$
  \mathbb{E}[L] = \iint \{f(x) - t\}^2 p(x, t)dx dt \quad (4.35)
  $$
- **最优解 (回归函数)**: 最小化期望平方损失的最优预测是**条件均值**。
  $$
  f^*(x) = \mathbb{E}_{t}[t|x] = \int t p(t|x)dt \quad (4.37)
  $$
- **期望损失的分解**: 期望损失可以分解为两部分。
  $$
  \mathbb{E}[L] = \int\{f(x)-\mathbb{E}[t|x]\}^{2}p(x)dx + \int \text{var}[t|x]p(x)dx \quad (4.39)
  $$
  第一项取决于我们的预测 $f(x)$，第二项是数据固有的噪声，是不可约减的误差。
- **Minkowski 损失**: 平方损失的一种泛化形式。
  $$
  \mathbb{E}[L_q] = \iint |f(x) - t|^q p(x, t)dx dt \quad (4.40)
  $$

---

## 3 偏置-方差权衡 (The Bias-Variance Trade-off)

- **核心思想**: 模型的期望损失可以分解为偏差、方差和噪声。
- **真实函数**: 设 $h(x) = \mathbb{E}[t|x]$ 为理想的最优预测函数。
- **期望损失分解**:
  $$
  \text{expected loss} = (\text{bias})^2 + \text{variance} + \text{noise} \quad (4.46)
  $$
- **各项定义**:
  - **偏差平方 (Bias^2)**: 模型在**所有可能训练集**上的平均预测与真实函数之间的差距。衡量模型的**拟合能力**。
    $$
    (\text{bias})^2 = \int \{\mathbb{E}_{\mathcal{D}}[f(x; \mathcal{D})] - h(x)\}^2 p(x)dx \quad (4.47)
    $$
  - **方差 (Variance)**: 模型对于**不同训练集**的预测结果的波动程度。衡量模型对数据扰动的**敏感度**。
    $$
    \text{variance} = \int \mathbb{E}_{\mathcal{D}}[\{f(x; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[f(x; \mathcal{D})]\}^2] p(x)dx \quad (4.48)
    $$
  - **噪声 (Noise)**: 数据本身固有的、不可避免的误差。
    $$
    \text{noise} = \iint \{h(x) - t\}^2 p(x, t)dx dt \quad (4.49)
    $$
- **权衡**:
  - **简单模型**: 高偏差，低方差 (欠拟合)。
  - **复杂模型**: 低偏差，高方差 (过拟合)。

