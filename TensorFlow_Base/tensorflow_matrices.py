# -*- coding: utf-8 -*-
"""
TensorFlow 操作矩阵
    查看矩阵的值和大小的函数
    创建矩阵
    矩阵转置
    矩阵乘法
    矩阵内积
    矩阵的加法和减法
    矩阵的行列式
    矩阵的逆矩阵
    矩阵的分解
    矩阵的切片与连接
    对称矩阵的特征值和特征向量

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

""" 查看矩阵的值和大小的函数 """
def print_matrix(matrix_to_print):
    # 打印矩阵名
    print("\n" + matrix_to_print + ":")
    # 打印矩阵的值
    with tf.Session() as sess: # 创建一个图会话
        print(eval("sess.run(" + matrix_to_print + ")"))
    # 打印矩阵的大小
    print(eval(matrix_to_print + ".shape"))

""" 创建矩阵 """
# 创建对角矩阵
diag_matrix = tf.diag([1.0, 1.0, 1.0])
print_matrix('diag_matrix')
# 从张量创建矩阵
truncated_matrix = tf.truncated_normal([2, 3])
print_matrix('truncated_matrix')
filled_matrix = tf.fill([2,3], 5.0)
print_matrix('filled_matrix')
randunif_matrix = tf.random_uniform([3,2])
print_matrix('randunif_matrix')
npconv_matrix = tf.convert_to_tensor(np.array([[1., 2., 3.],[-3., -7., -1.],[0., 5., -2.]]))
print_matrix('npconv_matrix')

""" 矩阵转置 """
A = tf.constant([[1.,2.,3.],[4.,5.,6.]])
print_matrix('A')
A_trans = tf.transpose(A)
print_matrix('A_trans')

""" 矩阵乘法 """
B = tf.constant([[4.,5.,6.],[7.,8.,9.]])
print_matrix('B')
AT_matmul_B = tf.matmul(A_trans,B)
print_matrix('AT_matmul_B')

""" 矩阵内积 """
# 矩阵内积有两种方式
A_mul_B = A*B
print_matrix('A_mul_B')
A_mul_B_tf = tf.multiply(A, B)
print_matrix('A_mul_B_tf')

""" 矩阵的加法和减法 """
# 加法有两种方式
A_add_B = A + B
print_matrix('A_add_B')
A_add_B_tf = tf.add(A, B)
print_matrix('A_add_B_tf')
# 减法有两种方式
A_subtract_B = A - B
print_matrix('A_subtract_B')
A_subtract_B_tf = tf.subtract(A, B)
print_matrix('A_subtract_B_tf')

""" 矩阵的行列式 """
C = tf.constant([[1.,2.,3.],[5.,6.,4.],[9.,7.,8.]])
print_matrix('C')
C_det = tf.matrix_determinant(C)
print_matrix('C_det')

""" 矩阵的逆矩阵 """
""" 注意：TensorFlow 中的矩阵求逆方法是 Cholesky 矩阵分解法（又称为平方根法），
矩阵需要为对称正定矩阵或者可进行 LU 分解，否则报错"""
C_inv = tf.matrix_inverse(C)
print_matrix('C_inv')

""" 矩阵的分解 """
D = tf.diag([1.0, 2.0, 3.0])
print_matrix('D')
D_chol = tf.cholesky(D)
print_matrix('D_chol')
# 检验是否正确
D_test = tf.matmul(D_chol, tf.transpose(D_chol))
print_matrix('D_test')

""" 矩阵的切片与连接 """
# 矩阵切片有两种方式
D_slice = D[0,:]
print_matrix('D_slice')
D_slice_tf = tf.slice(D, begin=[0,0], size=[1,3])
print_matrix('D_slice_tf')
# 矩阵连接
D_concat = tf.concat([C,D], axis=0)
print_matrix('D_concat')

""" 对称矩阵的特征值和特征向量 """
""" 注意：tensorflow 只能计算对称矩阵的特征值与特征向量，与 np.linalg.eigh 相同，
非对称矩阵求取的特征值与特征向量并非真正意义上的特征值与特征向量 """
E = tf.constant([[1.,2.,3.],[2.,4.,5.],[3.,5.,6.]]) # 对称矩阵
print_matrix('E')
E_eigenvalues, E_eigenvectors = tf.self_adjoint_eig(E)
print_matrix('E_eigenvalues')
print_matrix('E_eigenvectors')
# 检验是否正确
E_test = tf.matmul(tf.matmul(E_eigenvectors,tf.diag(E_eigenvalues)),tf.matrix_inverse(E_eigenvectors))
print_matrix('E_test')
