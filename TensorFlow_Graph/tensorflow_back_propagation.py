# -*- coding: utf-8 -*-
"""
TensorFlow 实现反向传播，通过调节模型变量来最小化损失函数
    回归算法的例子，拟合一个平面
    分类算法的例子，将正态分布分割成不同的两类

WangKai 编写于 2019/04/30 13:00:00 (UTC+08:00)
  中国科学院大学, 北京, 中国
  地球与行星科学学院
  Comments, bug reports and questions, please send to:
  wangkai185@mails.ucas.edu.cn

Versions:
  最近更新: 2019/04/30
      算法构建，测试
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

""" 回归算法的例子，拟合一个平面 """
# 初始化计算图
ops.reset_default_graph()
sess = tf.Session()
# 使用 NumPy 生成假数据(phony data), 总共 100 个点
xy_vals = np.float32(np.random.rand(2, 100)) # 随机输入二维因变量
z_vals = np.dot([0.100, 0.200], xy_vals) + 0.300 # 平面 z=0.1x+0.2y+0.3
# 创建占位符
xy_data = tf.placeholder(shape=[2, 100], dtype=tf.float32)
z_target = tf.placeholder(shape=[100], dtype=tf.float32)
# 创建变量 W,b 作为拟合参数
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
# 创建计算图操作，构造线性模型
z = tf.matmul(W, xy_vals) + b
# 创建 L2 正则损失函数
loss = tf.reduce_mean(tf.square(z - z_vals))
# 声明变量的优化器，标准梯度下降法，最小化方差
# 小学习率收敛慢、精度高；大学习率收敛快，精度低
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 启动图
sess.run(init)
# 训练算法，拟合平面。迭代 201 次，每 20 次迭代打印返回结果。
# 每次迭代将所有点坐标传入计算图中。TensorFlow 将自动地计算损失，调整 W,b 偏差来最小化损失
print('Regression example:')
for step in range(0, 200):
    sess.run(train, feed_dict={xy_data: xy_vals, z_target: z_vals})
    if (step+1)%20 == 0:
        print('Step #' + str(step+1) + ': W = ' + str(sess.run(W)) + '; b = ' + 
                str(sess.run(b)) + '; Loss = ' + 
                str(sess.run(loss, feed_dict={xy_data: xy_vals, z_target: z_vals})))
# 得到最佳拟合结果应接近于 W = [[0.100  0.200]]; b = [0.300]

""" 分类算法的例子，将正态分布分割成不同的两类 """
# 重置计算图
ops.reset_default_graph()
sess = tf.Session()
# 从正态分布 (N(-1,1), N(3,1)) 生成数据，总共 100 个点
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))[:,np.newaxis]
# 创建目标标签，N(-1,1) 为 '0' 类，N(3,1) 为 '1' 类，各 50 个点
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))[:,np.newaxis]
# 创建占位符，一次使用一个随机数据
x_data = tf.placeholder(shape=[100, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[100, 1], dtype=tf.float32)
# 创建变量 A 作为最佳聚类边界的负值，初始值为 10 附近的随机数，远离理论值 -(-1+3)/2 = -1
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))
# 创建计算图操作，模型算法是 sigmoid(x+A)
my_output = tf.add(x_data, A)
# 创建损失函数，使用 Sigmoid 交叉熵损失函数（Sigmoid cross entropy loss）
xentropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target))
# 声明变量的优化器，标准梯度下降法，最小化交叉熵
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(xentropy)
# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)
# 通过随机选择的数据迭代 101 次，相应地更新变量 A。每迭代 10 次打印出变量 A 和损失的返回值
print('\nClassification example:')
for i in range(100):
    sess.run(train_step, feed_dict={x_data: x_vals, y_target: y_vals})
    if (i+1)%10 == 0:
        print('Step #' + str(i+1) + ': A = ' + str(sess.run(A)) + '; Loss = ' +
                str(sess.run(xentropy, feed_dict={x_data: x_vals, y_target: y_vals})))
# 得到最佳拟合结果应趋近于 A = [-1.]
