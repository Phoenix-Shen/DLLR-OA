# Decentralized Learning with Limited Subcarriers through Over-the-Air Computation

## 方法介绍

### 去中心化模型

考虑去中心化的学习，有$n$个客户端的集合$\mathcal{V}$，有个通信矩阵$\mathcal{W} = (W_{ij})_{n * n}$里面每个元素就代表客户端$i,j$能不能通信。

时间被分为很多同步轮，在每一轮中，每个客户端都汇聚它们邻居节点的信息，然后以本地数据和邻居信息来进行模型更新。

我们总的优化目标还是找到一个参数$\theta$使每个客户端损失的和最小。
$$
\min f(\theta) = \frac{1}{n} \sum_{n=1}^n \mathbb{E}_{\xi_i \sim \mathcal{D}_i}[F_i(\theta,\xi_i)]
$$

在这篇文章中，我们考虑动态的情况，即$\mathcal{W}$是一直在变的，有些节点可能会掉线，有些节点可能会半路加进来。

### 空中汇聚

客户端通过**无线多重访问信道**(MAC）进行通信，并汇聚邻居的信息。

客户端$i$所需要的参数的每个分量(component)都是被认为是由一个子载波承载的，因此，在去中心化学习的第$t$轮中，客户端$i$收到的信号子载波$k$中的信号可以表示为
$$
y_i^t(k) = \sum_{j\in N_i^t} b_{ij}^t(k)h_{ij}^t(k)x_{ij}^t(k)+n_i^t(k)
$$
