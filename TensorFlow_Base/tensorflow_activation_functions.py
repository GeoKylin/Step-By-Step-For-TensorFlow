# -*- coding: utf-8 -*-
"""
TensorFlow 的激励函数，位于神经网络（neural network，nn）库，
激励函数是作用在张量上的非线性操作
    查看张量的值和大小的函数
    整流线性单元（Rectifier linear unit，ReLU）
    ReLU6 函数
    sigmoid 函数
    双曲正切函数（hyper tangent，tanh）
    softsign 函数
    softplus 激励函数
    ELU 激励函数（Exponential Linear Unit，ELU）

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

""" 查看张量的值和大小的函数 """
def print_tensor(tensor_to_print):
    # 打印张量名
    print("\n" + tensor_to_print + ":")
    # 打印张量的值
    with tf.Session() as sess: # 创建一个图会话
        print(eval("sess.run(" + tensor_to_print + ")"))
    # 打印张量的大小
    print(eval(tensor_to_print + ".shape"))

""" 整流线性单元（Rectifier linear unit，ReLU） """
A = tf.constant([-3.,3.,10.])
print_tensor('A')
relu_A = tf.nn.relu(A)
print_tensor('relu_A')

""" ReLU6 函数，抵消 ReLU 激励函数的线性增长部分 """
relu6_A = tf.nn.relu6(A)
print_tensor('relu6_A')

""" sigmoid 函数，最常用的连续、平滑的激励函数，也被称作逻辑函数（Logistic函数） """
sigmoid_A = tf.nn.sigmoid(A)
print_tensor('sigmoid_A')

""" 双曲正切函数（hyper tangent，tanh） """
tanh_A = tf.nn.tanh(A)
print_tensor('tanh_A')

""" softsign 函数，是符号函数的连续估计 """
softsign_A = tf.nn.softsign(A)
print_tensor('softsign_A')

""" softplus 激励函数，是 ReLU 激励函数的平滑版 """
softplus_A = tf.nn.softplus(A)
print_tensor('softplus_A')

""" ELU 激励函数（Exponential Linear Unit，ELU） """
elu_A = tf.nn.elu(A)
print_tensor('elu_A')

