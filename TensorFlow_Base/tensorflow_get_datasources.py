# -*- coding: utf-8 -*-
"""
通过 TensorFlow 和 Python 访问各种数据源
    鸢尾花卉数据集（Iris data）
    出生体重数据（Birth weight data）
    波士顿房价数据（Boston Housing data）
    MNIST 手写体字库
    垃圾短信文本数据集（Spam-ham text data）
    影评样本数据集（Movie review data）
    莎士比亚著作文本数据集（Shakespeare text data）
    英德句子翻译样本集（English-German sentence translation data）

WangKai 编写于 2019/04/30 13:00:00 (UTC+08:00)
  中国科学院大学, 北京, 中国
  地球与行星科学学院
  Comments, bug reports and questions, please send to:
  wangkai185@mails.ucas.edu.cn

Versions:
  最近更新: 2019/04/30
      算法构建，测试
"""

""" 鸢尾花卉数据集（Iris data） """
from sklearn import datasets
iris = datasets.load_iris()
print('Iris data:')
print(len(iris.data))
# 150
print(len(iris.target))
# 150
print(iris.data[0]) # Sepal length, Sepal width, Petal length, Petal width
# [ 5.1 3.5 1.4 0.2]
print(set(iris.target)) # I. setosa, I. virginica, I. versicolor
# {0, 1, 2}

""" 出生体重数据（Birth weight data），from 马萨诸塞大学（University of Massachusetts，UMASS） """
import requests
birthdata_url = 'https://raw.githubusercontent.com/GeoKylin/GA---Data-Science/master/Week%206/lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\n')
birth_header = [x for x in birth_data[0].split() if len(x)>=1]
birth_data = [[float(x) for x in y.split() if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
print('\nBirth weight data:')
print(len(birth_data))
# 189
print(len(birth_data[0]))
# 11

""" 波士顿房价数据（Boston Housing data），from 加州大学欧文分校（University of California Irvine，UCI） """
import requests
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split() if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
print('\nBoston Housing data:')
print(len(housing_data))
# 506
print(len(housing_data[0]))
# 14

""" MNIST 手写体字库 """
print('\nMNIST data:')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(len(mnist.train.images))
# 55000
print(len(mnist.test.images))
# 10000
print(len(mnist.validation.images))
# 5000
print(mnist.train.labels[1,:]) # The first label is a '3'
# [ 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]

""" 垃圾短信文本数据集（Spam-ham text data），from 加州大学欧文分校（University of California Irvine，UCI） """
import requests
import io
from zipfile import ZipFile
zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
r = requests.get(zip_url)
z = ZipFile(io.BytesIO(r.content))
zfile = z.read('SMSSpamCollection')
text_data = [x.split('\t') for x in zfile.split('\n') if len(x)>=1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]
print('\nSpam-ham text data:')
print(len(text_data_train))
# 5574
print(set(text_data_target))
# {'ham', 'spam'}
print(text_data_train[1])
# Ok lar... Joking wif u oni...

""" 影评样本数据集（Movie review data），from 康奈尔大学（Cornell University） """
import requests
import io
import tarfile
movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
r = requests.get(movie_data_url)
# Stream data into temp object
stream_data = io.BytesIO(r.content)
tmp = io.BytesIO()
while True:
    s = stream_data.read(16384)
    if not s:
        break
    tmp.write(s)
stream_data.close()
tmp.seek(0)
# Extract tar file
tar_file = tarfile.open(fileobj=tmp, mode="r:gz")
pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
# Save pos/neg reviews (Also deal with encoding)
pos_data = []
for line in pos:
    pos_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
neg_data = []
for line in neg:
    neg_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
tar_file.close()
print('\nMovie review data:')
print(len(pos_data))
# 5331
print(len(neg_data))
# 5331
print(neg_data[0])
# simplistic , silly and tedious .

""" 莎士比亚著作文本数据集（Shakespeare text data） """
import requests
shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
# Get Shakespeare text
response = requests.get(shakespeare_url)
shakespeare_file = response.content
# Decode binary into string
shakespeare_text = shakespeare_file.decode('utf-8')
# Drop first few descriptive paragraphs.
shakespeare_text = shakespeare_text[7675:]
print('\nShakespeare text data:')
print(len(shakespeare_text)) # Number of characters
# 5582212

""" 英德句子翻译样本集（English-German sentence translation data） """
import requests
import io
from zipfile import ZipFile
sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
r = requests.get(sentence_url)
z = ZipFile(io.BytesIO(r.content))
zfile = z.read('deu.txt')
# Format Data
eng_ger_data = [x.split('\t') for x in zfile.split('\n') if len(x)>=1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
print('\nEnglish-German sentence translation data:')
print(len(english_sentence))
# 192881
print(len(german_sentence))
# 192881
print(eng_ger_data[10])
# ['Go on.', 'Mach weiter.']
