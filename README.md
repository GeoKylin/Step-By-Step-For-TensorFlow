# 深度学习库——TensorFlow 完全自学手册
## 简介
该项目旨在介绍 TensorFlow 的基本使用方法，第一部分介绍 TensorFlow 的基础知识，第二部分介绍 TensorFlow 计算图的有关操作，第三部分为 TensorFlow 的具体应用实例。

阅读 PDF 文档，[请点我](./Doc/Manual.pdf)。

## TensorFlow 简介
[Tensorflow](https://tensorflow.google.cn) 是广泛使用的实现机器学习以及其它涉及大量数学运算的算法库之一。它由 Google 开发，2015 年 11 月，Google 公司开源了 TensorFlow ，随后不久 TensorFlow 成为 [GitHub](https://github.com/tensorflow/tensorflow) 上最受欢迎的机器学习库。Google 几乎在所有应用程序中都使用 Tensorflow 来实现机器学习。例如，如果你使用到了 Google 照片或 Google 语音搜索，那么你就间接使用了 Tensorflow 模型。它们在大型 Google 硬件集群上工作，在感知任务方面功能强大。

Google 著名人工智能程序 [AlphaGo](https://deepmind.com/research/alphago/) 于 2017 年年初化身 Master，在弈城和野狐等平台上连胜中日韩围棋高手，其中包括围棋世界冠军井山裕太、朴廷桓、柯洁等，还有棋圣聂卫平，总计取得 60 连胜，未尝败绩。而 AlphaGo 背后神秘的推动力就是 TensorFlow。DeepMind 宣布全面迁移到 TensorFlow 后，AlphaGo 的算法训练任务就全部放在了 TensorFlow 这套分布式框架上。

TensorFlow 采用数据流图（data flow graphs）用于数值计算，这些数据流图也称计算图是有向无环图，并且支持并行计算。节点（nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。TensorFlow 创建计算图、自动求导和定制化的方式使得其能够很好地解决许多不同的机器学习问题，它灵活的架构让你可以在多种平台上展开计算，例如台式计算机中的一个或多个 CPU（或 GPU）、服务器和移动设备等。

![Alt tensors_flowing](./Doc/tensors_flowing.gif "tensors_flowing")

## 目录
1. [TensorFlow 基础](./TensorFlow_Base/)
	- [在 TensorFlow 中创建张量](./TensorFlow_Base/tensorflow_tensor.py)
	- [使用 TensorFlow 的变量和占位符](./TensorFlow_Base/tensorflow_placeholders_and_variables.py)
	- [TensorFlow 操作矩阵](./TensorFlow_Base/tensorflow_matrices.py)
	- [TensorFlow 张量的基本操作](./TensorFlow_Base/tensorflow_tensor_operations.py)
	- [TensorFlow 的激励函数](./TensorFlow_Base/tensorflow_activation_functions.py)
	- [通过 TensorFlow 和 Python 访问各种数据源](./TensorFlow_Base/tensorflow_get_datasources.py)
2. [TensorFlow 计算图](./TensorFlow_Graph/)
	- [TensorFlow 计算图中对象和层的操作](./TensorFlow_Graph/tensorflow_graph_and_layer.py)
	- [TensorFlow 实现损失函数](./TensorFlow_Graph/tensorflow_loss_functions.py)
	- [TensorFlow 实现反向传播](./TensorFlow_Graph/tensorflow_back_propagation.py)
	- [TensorFlow 实现批量训练和随机训练](./TensorFlow_Graph/tensorflow_batch_and_stochastic_training.py)
	- [TensorFlow 实现模型评估](./TensorFlow_Graph/tensorflow_evaluating_models.py)
	- [TensorFlow 的可视化: Tensorboard](./TensorFlow_Graph/tensorflow_use_tensorboard.py)
3. [TensorFlow 应用实例](./TensorFlow_Sample/)
	- [MNIST 手写数字问题](./TensorFlow_Sample/tensorflow_mnist.py)

## 如何使用

### 环境
- Python 3.x（推荐，部分操作系统 Python 2.x 也可以）
- TensorFlow 1.x
- Matplotlib
- Numpy

### 配置 TensorFlow
使用 pip 命令：

```
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果你使用的是 Anaconda：

```
conda install tensorflow
```

## 更新日志
- 2019.05.02 首次上传