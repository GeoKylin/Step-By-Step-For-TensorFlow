# -*- coding: utf-8 -*-
"""
TensorFlow 实现模型评估
    评估回归算法模型
    评估分类算法模型

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

""" 评估回归算法模型 """
""" 不管算法模型预测的如何，我们都需要测试算法模型，这点相当重要。
在训练数据和测试数据上都进行模型评估，以搞清楚模型是否过拟合。 """
# 初始化计算图
ops.reset_default_graph()
sess = tf.Session()
# 创建数据，要拟合的事实上是数据中心点与原点所在直线的斜率
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
# 分割训练数据集和测试数据集
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals)*0.8)), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
# 创建占位符
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# 声明变量，作为数据中心点与原点所在直线的斜率，理论值为 10.
A = tf.Variable(tf.random_normal(shape=[1,1]))
# 声明批量训练集大小
batch_size = 25
# 在计算图中声明算法模型
my_output = tf.matmul(x_data, A)
# 声明损失函数
loss = tf.reduce_mean(tf.square(my_output - y_target))
# 声明优化器算法
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)
# 初始化模型变量 A
init = tf.global_variables_initializer()
# 启动图
sess.run(init)
# 迭代训练模型
print('Evaluate the regression model:')
for i in range(100):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target:
    rand_y})
    if (i+1)%25==0:
        print('Step #' + str(i+1) + ': A = ' + str(sess.run(A)) + '; Loss = ' +
                str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))
# 得到最佳拟合结果应接近于 A = [10.]
# 打印训练数据集和测试数据集训练的 MSE 损失函数值，以评估训练模型
mse_train = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
mse_test = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
print('MSE on train set: ' + str(np.round(mse_train, 2)))
print('MSE on test set: ' + str(np.round(mse_test, 2)))

""" 评估分类算法模型 """
""" 对于分类模型的例子，创建准确率函数（accuracy function），
分别调用 sigmoid 来测试分类是否正确。 """
# 重置计算图
ops.reset_default_graph()
sess = tf.Session()
# 从正态分布 (N(-1,1), N(2,1)) 生成数据，总共 100 个点
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(2, 1, 50)))
# 创建目标标签，N(-1,1) 为 '0' 类，N(2,1) 为 '1' 类，各 50 个点
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
# 分割训练数据集和测试数据集
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals)*0.8)), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
# 创建占位符
x_data = tf.placeholder(shape=[1, None], dtype=tf.float32)
y_target = tf.placeholder(shape=[1, None], dtype=tf.float32)
# 创建变量 A 作为最佳聚类边界的负值，初始值为 10 附近的随机数，远离理论值 -(-1+2)/2 = -0.5
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))
# 声明批量训练集大小
batch_size = 25
# 在计算图中声明算法模型
my_output = tf.add(x_data, A)
# 声明损失函数
xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target))
# 声明优化器算法
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)
# 初始化变量 A
init = tf.global_variables_initializer()
sess.run(init)
# 迭代训练
print('\nEvaluate the classification model:')
for i in range(1800):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = [x_vals_train[rand_index]]
    rand_y = [y_vals_train[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%200 == 0:
        print('Step #' + str(i+1) + ': A = ' + str(sess.run(A)) + '; Loss = ' +
                str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))
# 得到最佳拟合结果应趋近于 A = [-0.5]
# 为了评估训练模型，我们创建预测操作。用 squeeze() 函数封装预测操作，
# 使得预测值和目标值有相同的维度。然后用 equal() 函数检测是否相等，
# 把得到的 true 或 false 的 boolean 型张量转化成 float32 型，再对其取平均值，
# 得到一个准确度值。我们将用这个函数评估训练模型和测试模型。
y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_data, A))))
correct_prediction = tf.equal(y_prediction, y_target)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_value_test = sess.run(accuracy, feed_dict={x_data: [x_vals_test], y_target: [y_vals_test]})
acc_value_train = sess.run(accuracy, feed_dict={x_data: [x_vals_train], y_target: [y_vals_train]})
print('Accuracy on train set: ' + str(acc_value_train))
print('Accuracy on test set: ' + str(acc_value_test))
# 绘制分类学习模型图
A_result = sess.run(A)
bins = np.linspace(-5, 5, 50)
plt.figure()
plt.hist(x_vals[0:50], bins, alpha=0.5, label='N(-1,1)', color='blue')
plt.hist(x_vals[50:100], bins[0:50], alpha=0.5, label='N(2,1)', color='red')
plt.plot((-A_result, -A_result), (0, 8), 'k--', linewidth=3, label='-A = '+ str(np.round(-A_result, 2)))
plt.legend(loc='upper right')
plt.title('Binary Classifier, Accuracy=' + str(np.round(acc_value_test, 2)))
plt.show()
