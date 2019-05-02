# -*- coding: utf-8 -*-
"""
TensorFlow 计算图中对象和层的操作
    计算图中对象的操作
    TensorFlow 嵌入 Layer 的操作
    TensorFlow 多层 Layer 的操作

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

""" 计算图中对象的操作 """
# 初始化计算图
ops.reset_default_graph()
sess = tf.Session()
# 创建要传入的数据
x_vals = np.array([1., 3., 5., 7., 9.])
# 创建占位符
x_data = tf.placeholder(tf.float32)
# 创建常量矩阵
m_const = tf.constant(3.)
# 声明操作，表示成计算图
my_product = tf.multiply(x_data, m_const)
# 通过计算图赋值
print('Operations in a Computational Graph:')
for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data: x_val}))
# 3.0
# 9.0
# 15.0
# 21.0
# 27.0

""" TensorFlow 嵌入 Layer 的操作 """
""" 在同一个计算图中进行多个操作。传入两个形状为 3×5 的 numpy 数组，
然后每个矩阵乘以常量矩阵（形状为：5×1），将返回一个形状为 3×1 的矩阵。
紧接着再乘以 1×1 的矩阵，返回的结果矩阵仍然为 3×1。
最后，加上一个 3×1 的矩阵。 """
# 创建要传入的数据
my_array = np.array([[1., 3., 5., 7., 9.],
                    [-2., 0., 2., 4., 6.],
                    [-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
# 创建占位符
x_data = tf.placeholder(tf.float32, shape=(3, 5))
# 创建常量矩阵
m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])
# 声明操作，表示成计算图
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)
# 通过计算图赋值
print('\nLayering Nested Operations:')
for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data: x_val}))
# [[ 102.]
# [ 66.]
# [ 58.]]
# [[ 114.]
# [ 78.]
# [ 70.]]

""" TensorFlow 多层 Layer 的操作 """
""" 连接传播数据的多层 Layer。生成随机图片数据，对2D图像进行滑动窗口平均，
然后通过自定义操作层 Layer 返回结果。 """
# 通过 numpy 创建 2D 图像，4×4 像素图片
# 注意：TensorFlow 的图像函数是处理四维图片的，这四维是：图片数量、高度、宽度和颜色通道
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)
# 创建占位符，用来传入图片
x_data = tf.placeholder(tf.float32, shape=x_shape)
# 创建过滤 4×4 像素图片的滑动平均窗口（第一层 Layer）
my_filter = tf.constant(0.25, shape=[2, 2, 1, 1]) # 滑动窗口
my_strides = [1, 2, 2, 1] # 滑动窗口的步幅
mov_avg_layer= tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Avg_Window')
# 定义一个自定义 Layer，操作滑动平均窗口的 2×2 的返回值（第二层 Layer）
def custom_layer(input_matrix):
    # 压缩尺寸为 1 的纬度
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    # 创建常量矩阵
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    # 声明操作，表示成计算图 Ax + b
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b)
    # 返回激励函数
    return(tf.nn.sigmoid(temp))
# 把新定义的 Layer 加入到计算图中，表示为 mov_avg_layer 层的下一层
with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)
# 为占位符传入 4×4 像素图片，执行计算图
print('\nWorking with Multiple Layers:')
print(sess.run(custom_layer1, feed_dict={x_data: x_val}))
