# -*- coding: utf-8 -*-
"""
TensorFlow 张量的基本操作
    查看张量的值和大小的函数
    张量的加减乘除
    数学函数
    特殊数学函数
    组合基本函数生成自定义函数

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

""" 张量的加减乘除 """
A = tf.constant([1,2])
print_tensor('A')
B = tf.constant([3,4])
print_tensor('B')
# 张量加法
A_add_B = A + B
print_tensor('A_add_B')
A_add_B_tf = tf.add(A, B)
print_tensor('A_add_B_tf')
# 张量减法
A_subtract_B = A - B
print_tensor('A_subtract_B')
A_subtract_B_tf = tf.subtract(A, B)
print_tensor('A_subtract_B_tf')
# 张量元素间乘法
A_multiply_B = A*B
print_tensor('A_multiply_B')
A_multiply_B_tf = tf.multiply(A, B)
print_tensor('A_multiply_B_tf')
# 张量元素间除法
A_divide_B = tf.divide(A, B)
print_tensor('A_divide_B')
# 求商
rem_A_divide_B = A/B
print_tensor('rem_A_divide_B')
rem_A_divide_B_tf = tf.floordiv(A, B)
print_tensor('rem_A_divide_B_tf')
# 求余数
mod_A_divide_B = A%B
print_tensor('mod_A_divide_B')
mod_A_divide_B_tf = tf.mod(A, B)
print_tensor('mod_A_divide_B_tf')
# 三维向量的外积（只能是三维向量）
C = tf.constant([1.,0.,0.])
print_tensor('C')
D = tf.constant([0.,1.,0.])
print_tensor('D')
C_cross_D = tf.cross(C, D)
print_tensor('C_cross_D')

""" 数学函数 """
E = tf.constant([-1.2,0.3,4.5])
print_tensor('E')
# 绝对值
abs_E = tf.abs(E)
print_tensor('abs_E')
# 倒数
rec_E = tf.reciprocal(E)
print_tensor('rec_E')
# 相反数
neg_E = tf.negative(E)
print_tensor('neg_E')
# 最大值
max_E = tf.reduce_max(E)
print_tensor('max_E')
# 最小值
min_E = tf.reduce_min(E)
print_tensor('min_E')
# 平均值
mean_E = tf.reduce_mean(E)
print_tensor('mean_E')
# 求和
sum_E = tf.reduce_sum(E)
print_tensor('sum_E')
# 符号函数
sign_E = tf.sign(E)
print_tensor('sign_E')
# 向上取整
ceil_E = tf.ceil(E)
print_tensor('ceil_E')
# 向下取整
floor_E = tf.floor(E)
print_tensor('floor_E')
# 四舍五入
round_E = tf.round(E)
print_tensor('round_E')
# 两个张量中对应元素的最大值
max_D_E = tf.maximum(D, E)
print_tensor('max_D_E')
# 两个张量中对应元素的最小值
min_D_E = tf.minimum(D, E)
print_tensor('min_D_E')
# 正弦函数
sin_E = tf.sin(E)
print_tensor('sin_E')
# 余弦函数
cos_E = tf.cos(E)
print_tensor('cos_E')
# 自然底数的指数函数
exp_E = tf.exp(E)
print_tensor('exp_E')
# 自然对数
log_E = tf.log(E)
print_tensor('log_E')
# 对应元素的幂
E_pow_D = tf.pow(E, D)
print_tensor('E_pow_D')
# 平方
square_E = tf.square(E)
print_tensor('square_E')
# 平方根
sqrt_E = tf.sqrt(E)
print_tensor('sqrt_E')
# 平方根的倒数
rsqrt_E = tf.rsqrt(E)
print_tensor('rsqrt_E')

""" 特殊数学函数 """
# 两个张量差的平方
sqdiff_D_E = tf.squared_difference(D, E)
print_tensor('sqdiff_D_E')
# 高斯误差函数
erf_E = tf.erf(E)
print_tensor('erf_E')
# 互补误差函数
erfc_E = tf.erfc(E)
print_tensor('erfc_E')
# 下不完全伽马函数
igamma_D_E = tf.igamma(D, E)
print_tensor('igamma_D_E')
# 上不完全伽马函数
igammac_D_E = tf.igammac(D, E)
print_tensor('igammac_D_E')
# 贝塔函数绝对值的自然对数，纬度降1
lbeta_E = tf.lbeta(E)
print_tensor('lbeta_E')
# 伽马函数绝对值的自然对数
lgamma_E = tf.lgamma(E)
print_tensor('lgamma_E')
# Psi 函数，lgamma 函数的导数
Psi_E = tf.digamma(E)
print_tensor('Psi_E')

""" 组合基本函数生成自定义函数 """
# 正切函数
F = tf.constant([0.,3.1416/4.])
print_tensor('F')
tan_F = tf.divide(tf.sin(F), tf.cos(F))
print_tensor('tan_F')
# 二次多项式函数
def my_polynomial(value): # 3x^2-x+10
    return(tf.subtract(3 * tf.square(value), value) + 10)
poly_F = my_polynomial(F)
print_tensor('poly_F')
