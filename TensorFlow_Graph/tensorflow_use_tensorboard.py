# -*- coding: utf-8 -*-
"""
TensorFlow 的可视化: Tensorboard
    拟合一个线性回归模型，并将迭代训练结果写入 Tensorboard 汇总数据
    将图片插入到 Tensorboard
    启动并查看 Tensorboard

WangKai 编写于 2019/04/30 13:00:00 (UTC+08:00)
  中国科学院大学, 北京, 中国
  地球与行星科学学院
  Comments, bug reports and questions, please send to:
  wangkai185@mails.ucas.edu.cn

Versions:
  最近更新: 2019/04/30
      算法构建，测试
"""
import os
import io
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

""" 拟合一个线性回归模型，并将迭代训练结果写入 Tensorboard 汇总数据 """
# 初始化计算图
ops.reset_default_graph()
sess = tf.Session()
# 设置要拟合直线的真实斜率
true_slope = 2.
# 创建随机数据，在直线 y=2x 附近，共 1000 个点
x_data = np.arange(1000)/10.
y_data = x_data * true_slope + np.random.normal(loc=0.0, scale=25, size=1000)
# 分割数据集为测试集和训练集
train_ix = np.random.choice(len(x_data), size=int(len(x_data)*0.9), replace=False)
test_ix = np.setdiff1d(np.arange(1000), train_ix)
x_data_train, y_data_train = x_data[train_ix], y_data[train_ix]
x_data_test, y_data_test = x_data[test_ix], y_data[test_ix]
# 创建占位符
x_graph_input = tf.placeholder(tf.float32, [None])
y_graph_input = tf.placeholder(tf.float32, [None])
# 创建变量 m 作为拟合直线的斜率估计
m = tf.Variable(tf.random_normal([1], 10, dtype=tf.float32), name='Slope')
# 声明批量训练数据集大小
batch_size = 1000
# 声明迭代次数为 generations+1
generations = 30
# 创建模型操作
output = tf.multiply(m, x_graph_input, name='Batch_Multiplication')
# 创建损失函数
residuals = output - y_graph_input
loss = tf.reduce_mean(tf.abs(residuals), name="L1_Loss")
# 创建优化器操作
my_optim = tf.train.GradientDescentOptimizer(0.01)
train_step = my_optim.minimize(loss)
# 创建时间戳，并创建当前 Tensorboard 日志文件夹
time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
log_dir = 'tensorboard/log_' + time_stamp
# 创建 summary_writer，将 Tensorboard summary 写入到当前日志文件夹
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
# 确保 summary_writer 写入的文件夹存在
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# 创建 Tensorboard 操作汇总一个标量值，该汇总的标量值为模型的斜率估计
with tf.name_scope('Slope_Estimate'):
    tf.summary.scalar('Slope_Estimate', tf.squeeze(m))
# 添加到 Tensorboard 的另一个汇总数据是直方图汇总，该直方图汇总输入张量，输出曲线图和直方图
with tf.name_scope('Loss_and_Residuals'):
    tf.summary.histogram('Histogram_Errors', loss)
    tf.summary.histogram('Histogram_Residuals', residuals)
# 创建完这些汇总操作之后，创建汇总合并操作综合所有的汇总数据
summary_op = tf.summary.merge_all()
# 初始化模型变量
init = tf.global_variables_initializer()
# 启动图
sess.run(init)
# 训练线性回归模型，每隔 2 次将迭代训练结果写入汇总数据
for i in range(generations):
    batch_indices = np.random.choice(len(x_data_train), size=batch_size)
    x_batch = x_data_train[batch_indices]
    y_batch = y_data_train[batch_indices]
    _, train_m, train_loss, summary = sess.run([train_step, m, loss, summary_op], feed_dict={x_graph_input: x_batch, y_graph_input: y_batch})
    test_loss, test_resids = sess.run([loss, residuals], feed_dict={x_graph_input: x_data_test, y_graph_input: y_data_test})
    if (i+1)%2 == 0:
        print('Step #{}: m = {}; Train Loss = {:.8}; Test Loss = {:.8}'.format(i+1, train_m, train_loss, test_loss))
        summary_writer.add_summary(summary, i+1)
        summary_writer.flush()

""" 将图片插入到 Tensorboard """
# 创建函数输出 protobuff 格式的图形，用来可视化数据点拟合的线性回归模型
def gen_linear_plot(slope):
    linear_prediction = x_data * slope
    plt.figure()
    plt.plot(x_data, y_data, 'b.', label='data')
    plt.plot(x_data, linear_prediction, 'r-', linewidth=3, label='predicted line')
    plt.legend(loc='upper left')
    plt.title('Slope = ' + str(slope))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return(buf)
# 创建将 protobuff 格式的图形
slope = sess.run(m)
plot_buf = gen_linear_plot(slope[0])
# 将 protobuff 格式的图形转为 Tensorboard 格式图片张量
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
# 增加图片张量的纬度
image = tf.expand_dims(image, 0)
# 将图片增加到 Tensorboard
image_summary_op = tf.summary.image("Linear_Plot", image)
image_summary = sess.run(image_summary_op)
summary_writer.add_summary(image_summary, i+1)
summary_writer.close()

""" 启动并查看 Tensorboard """
# 启动 Tensorboard，在浏览器中输入 http://localhost:6006 即可打开 Tensorboard 面板，
# 在命令行按下 Ctrl+C 则关闭 Tensorboard。
os.system("tensorboard --logdir " + log_dir + " --host localhost --port 6006")
