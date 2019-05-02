# -*- coding: utf-8 -*-
"""
TensorFlow 实现批量训练和随机训练
    TensorFlow 批量训练
    TensorFlow 随机训练
    绘制回归算法的批量训练损失和随机训练损失图

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
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

""" TensorFlow 批量训练 """
""" 随机训练可能导致比较“古怪”的学习过程，但使用大批量的训练会造成计算成本昂贵 """
# 初始化计算图
ops.reset_default_graph()
sess = tf.Session()
# 声明批量大小，批量大小是指通过计算图一次传入训练数据的多少
batch_size = 20
# 声明模型的数据、占位符和变量
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))
# 在计算图中增加矩阵乘法操作
my_output = tf.matmul(x_data, A)
# 声明损失函数
loss = tf.reduce_mean(tf.square(my_output - y_target))
# 声明优化器
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 启动图
sess.run(init)
# 通过循环迭代优化模型算法。每间隔 5 次迭代保存损失函数，用来绘制损失值图
print('Batch Training:')
loss_batch = []
for i in range(100):
    # 产生批量训练数据集
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    # 训练模型
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%5==0:
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Step #' + str(i+1) + ': A = ' + str(sess.run(A)) + '; Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)

""" TensorFlow 随机训练 """
# 通过循环迭代优化模型算法。每间隔 5 次迭代保存损失函数，用来绘制损失值图
print('\nStochastic Training:')
loss_stochastic = []
for i in range(100):
    # 产生随机训练数据
    rand_index = np.random.choice(100)
    rand_x = np.array([x_vals[rand_index]])[:,np.newaxis]
    rand_y = np.array([y_vals[rand_index]])[:,np.newaxis]
    # 训练模型
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%5==0:
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Step #' + str(i+1) + ': A = ' + str(sess.run(A)) + '; Loss = ' + str(temp_loss))
        loss_stochastic.append(temp_loss)

""" 绘制回归算法的批量训练损失和随机训练损失图 """
plt.figure()
plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()
