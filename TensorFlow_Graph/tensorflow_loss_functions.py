# -*- coding: utf-8 -*-
"""
TensorFlow 实现损失函数
    回归算法的损失函数
        L2 正则损失函数
        L1 正则损失函数
        Pseudo-Huber 损失函数
    分类算法的损失函数
        Hinge 损失函数
        交叉熵损失函数（Cross-entropy loss）
        Sigmoid 交叉熵损失函数（Sigmoid cross entropy loss）
        加权交叉熵损失函数（Weighted cross entropy loss）
        Softmax 交叉熵损失函数（Softmax cross-entropy loss）
        稀疏 Softmax 交叉熵损失函数（Sparse softmax cross-entropy loss）
    用 matplotlib 绘制损失函数

WangKai 编写于 2019/04/30 13:00:00 (UTC+08:00)
  中国科学院大学, 北京, 中国
  地球与行星科学学院
  Comments, bug reports and questions, please send to:
  wangkai185@mails.ucas.edu.cn

Versions:
  最近更新: 2019/04/30
      算法构建，测试
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

""" 回归算法的损失函数，回归算法是预测连续因变量的 """
""" L2 正则损失函数（即欧拉损失函数） """
# 初始化计算图
ops.reset_default_graph()
sess = tf.Session()
# 创建预测序列和目标序列作为张量
x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)
# L2 正则损失
l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)
print('L2 norm loss:')
print('l2_y_out = %f' % sess.run(tf.reduce_sum(l2_y_vals)))
# TensorFlow 内建的 L2 正则损失是实际 L2 正则的一半
l2_y_vals_nn = tf.nn.l2_loss(target - x_vals)
l2_y_out_nn = sess.run(l2_y_vals_nn)
print('l2_y_out_nn = %f' % sess.run(tf.reduce_sum(l2_y_vals_nn)))
print('l2_y_out / l2_y_out_nn = %f' % sess.run(tf.divide(tf.reduce_sum(l2_y_vals), l2_y_vals_nn)))

""" L1 正则损失函数（即绝对值损失函数） """
l1_y_vals = tf.abs(target - x_vals)
l1_y_out = sess.run(l1_y_vals)

""" Pseudo-Huber 损失函数 """
""" Huber 损失函数的连续、平滑估计，试图利用 L1 和 L2 正则削减极值处的陡峭，
使得目标值附近连续。它的表达式依赖参数 delta。 """
# delta = 0.25 时
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
phuber1_y_out = sess.run(phuber1_y_vals)
# delta = 5 时
delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals)/delta2)) - 1.)
phuber2_y_out = sess.run(phuber2_y_vals)

""" 分类算法的损失函数，分类损失函数是用来评估预测分类结果的 """
""" Hinge 损失函数，主要用来评估支持向量机算法，但有时也用来评估神经网络算法 """
# 创建预测序列和目标序列作为张量
x_vals = tf.linspace(-3., 5., 500)
target = tf.constant(1.)
targets = tf.fill([500,], 1.)
# Hinge 损失，使用目标值1，预测值离1越近，损失函数值越小
hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
hinge_y_out = sess.run(hinge_y_vals)

""" 交叉熵损失函数（Cross-entropy loss），有时也作为逻辑损失函数 """
xentropy_y_vals = - tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. -target), tf.log(1. - x_vals))
xentropy_y_out = sess.run(xentropy_y_vals)

""" Sigmoid 交叉熵损失函数（Sigmoid cross entropy loss） """
""" 先把 x_vals 值通过 sigmoid 函数转换，再计算交叉熵损失 """
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_vals, labels=targets)
xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)

""" 加权交叉熵损失函数（Weighted cross entropy loss） """
""" 是 Sigmoid 交叉熵损失函数的加权，对正目标加权 """
# 将正目标加权权重 0.5
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(logits=x_vals, targets=targets, pos_weight=weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

""" Softmax 交叉熵损失函数（Softmax cross-entropy loss） """
""" 通过 softmax 函数将输出结果转化成概率分布，然后计算真值概率分布的损失 """
unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=unscaled_logits, labels=target_dist)
print('\nSoftmax cross-entropy loss:')
print(sess.run(softmax_xentropy))
# [ 1.16012561]

""" 稀疏 Softmax 交叉熵损失函数（Sparse softmax cross-entropy loss） """
""" 把目标分类为 true 的转化成 index，而 Softmax 交叉熵损失函数将目标转成概率分布 """
unscaled_logits = tf.constant([[1., -3., 10.]])
sparse_target_dist = tf.constant([2])
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=sparse_target_dist)
print('\nSparse softmax cross-entropy loss:')
print(sess.run(sparse_xentropy))
# [ 0.00012564]

""" 用 matplotlib 绘制损失函数 """
# 回归算法的损失函数
x_array = sess.run(x_vals)
plt.figure()
plt.plot(x_array, l2_y_out, 'b-', label='L2 Loss')
plt.plot(x_array, l1_y_out, 'r--', label='L1 Loss')
plt.plot(x_array, phuber1_y_out, 'k-.', label='P-Huber Loss (0.25)')
plt.plot(x_array, phuber2_y_out, 'g:', label='P-Huber Loss (5.0)')
plt.ylim(-0.2, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()
# 分类算法的损失函数
plt.figure()
plt.plot(x_array, hinge_y_out, 'b-', label='Hinge Loss')
plt.plot(x_array, xentropy_y_out, 'r--', label='Cross Entropy Loss')
plt.plot(x_array, xentropy_sigmoid_y_out, 'k-.', label='Cross Entropy Sigmoid Loss')
plt.plot(x_array, xentropy_weighted_y_out, 'g:', label='Weighted Cross Enropy Loss (x0.5)')
plt.ylim(-1.5, 3)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()
