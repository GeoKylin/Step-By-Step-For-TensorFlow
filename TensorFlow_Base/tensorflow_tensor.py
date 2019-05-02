# -*- coding: utf-8 -*-
"""
在 TensorFlow 中创建张量
    查看张量的值和大小的函数
    固定张量
    相似形状的张量
    序列张量
    随机张量
    将其他数据变成张量

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

# 初始化计算图
ops.reset_default_graph()
# 创建计算图会话
sess = tf.Session()

""" 查看张量的值和大小的函数 """
def print_tensor(tensor_to_print):
    # 打印张量名
    print("\n" + tensor_to_print + ":")
    # 打印张量的值
    with tf.Session() as sess: # 创建一个图会话
        print(eval("sess.run(" + tensor_to_print + ")"))
    # 打印张量的大小
    print(eval(tensor_to_print + ".shape"))

""" 固定张量 """
# 创建指定维度的零张量
zero_tsr = tf.zeros([1, 2])
print_tensor('zero_tsr')
# 创建指定维度的单位张量
ones_tsr = tf.ones([2, 1])
print_tensor('ones_tsr')
# 创建指定维度的常数填充的张量
filled_tsr = tf.fill([2, 2], 42)
print_tensor('filled_tsr')
# 用已知常数张量创建一个张量
constant_tsr = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
print_tensor('constant_tsr')
# 广播一个值为张量
broadcast_tsr = tf.constant(42, shape=[1, 2])
print_tensor('broadcast_tsr')

""" 相似形状的张量 """
""" 注意：因为这些张量依赖给定的张量，所以初始化时需要按序进行。
如果打算一次性初始化所有张量，那么程序将会报错。 """
# 新建一个与给定的tensor类型大小一致的tensor，其所有元素为0或者1
zeros_similar = tf.zeros_like(constant_tsr)
print_tensor('zeros_similar')
ones_similar = tf.ones_like(constant_tsr)
print_tensor('ones_similar')

""" 序列张量 """
# 创建指定间隔的张量
linear_tsr = tf.linspace(start=0.0, stop=1.0, num=3) # 包含 stop
print_tensor('linear_tsr')
integer_seq_tsr = tf.range(start=6, limit=15, delta=3) # 不包含 limit
print_tensor('integer_seq_tsr')

""" 随机张量 """
# 生成均匀分布的随机数
randunif_tsr = tf.random_uniform([3, 2], minval=0, maxval=1)
print_tensor('randunif_tsr')
# 生成正态分布的随机数
randnorm_tsr = tf.random_normal([2, 3], mean=0.0, stddev=1.0)
print_tensor('randnorm_tsr')
# 生成带有指定边界的正态分布的随机数，其正态分布的随机数位于指定均值（期望）到两个标准差之间的区间
randtruncnorm_tsr = tf.truncated_normal([3, 3], mean=0.0, stddev=1.0)
print_tensor('randtruncnorm_tsr')
# 张量的随机化(沿着第一纬度)
randshuffled_tsr = tf.random_shuffle(constant_tsr)
print_tensor('randshuffled_tsr')
# 张量的随机剪裁
randcropped_tsr = tf.random_crop(constant_tsr, [1, 2])
print_tensor('randcropped_tsr')

""" 将其他数据变成张量 """
# 将列表转为张量
list_data = list([1,2,3])
listdata_tsr = tf.convert_to_tensor(list_data)
print_tensor('listdata_tsr')
# 将 numpy 数组转为张量
nparray_data = np.array([4,5,6])
nparraydata_tsr = tf.convert_to_tensor(nparray_data)
print_tensor('nparraydata_tsr')
