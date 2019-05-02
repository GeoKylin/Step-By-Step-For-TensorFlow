# -*- coding: utf-8 -*-
"""
使用 TensorFlow 的占位符和变量
    创建变量并初始化
    按序进行变量的初始化
    占位符的使用

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

""" 创建变量并初始化 """
# 初始化计算图
ops.reset_default_graph()
sess = tf.Session()
# 创建张量
my_tsr = tf.zeros([2,3])
# 封装张量来作为变量
my_var = tf.Variable(my_tsr)
# 初始化变量
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
# 打印变量值和大小
print('\nmy_var:')
print(sess.run(my_var))
print(my_var.shape)

""" 按序进行变量的初始化 """
""" 说明：如果是基于已经初始化的变量进行初始化，则必须按序进行初始化 """
# 创建一个图会话
sess = tf.Session()
first_var = tf.Variable(tf.zeros([3,2]))
# 初始化 first_var
sess.run(first_var.initializer)
print('\nfirst_var:')
print(sess.run(first_var))
print(my_var.shape)
# second_var 依赖于 first_var
second_var = tf.Variable(tf.ones_like(first_var))
# 初始化 second_var
sess.run(second_var.initializer)
print('\nsecond_var:')
print(sess.run(second_var))
print(my_var.shape)

""" 占位符的使用 """
""" 说明：占位符仅仅声明数据位置，用于传入数据到计算图 """
# 创建一个图会话
sess = tf.Session()
# 创建占位符 x
x = tf.placeholder(tf.float32, shape=[2,2])
# y 作为一个虚拟节点来控制计算图的操作
y = tf.identity(x)
# x_vals 是要传入到计算图的数据
x_vals = np.random.rand(2,2)
# 传入数据到计算图
sess.run(y, feed_dict={x: x_vals})
