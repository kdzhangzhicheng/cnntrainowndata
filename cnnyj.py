import scipy.io as sio
import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import *
from time import *
import os
import matplotlib.pyplot as plt
from pylab import *
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

###下面是讲解python怎么读取.mat文件以及怎么处理得到的结果###
load_fn = 'F:/zhicheng/张森师兄训练数据/ToZZC/feature_10.mat'
print(load_fn)
load_data = sio.loadmat(load_fn)
# load_matrix = load_data['matrix']
#  假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');
# 当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
# load_matrix_row = load_matrix[0]
#  取了当时matlab中matrix的第一行，python中数组行排
# print(np.split(load_data['feature.P']))
a = load_data['feature'][0]
a = a[0]
temp = a[1]
yj = a[2]
yj = yj[0]

fwj = a[3]
fwj = fwj[0]
# print("方位角：", fwj)

input_data = np.transpose(temp)  # 整理输入
# print("input_data", input_data)
# print("仰角", yj)

train_data = []
test_data = []
fwjtrainlabel = []
fwjtestlabel = []
yjtrainlabel = []
yjtestlabel = []
test_accuracy_list=[]
CKPT_DIR = 'C:/Users/Administrator/Desktop/owndatatest1/'

filename = 'C:/Users/Administrator/Desktop/owndatatest2/'
filename2 = 'C:/Users/Administrator/Desktop/owndatatest3/'
for pic in os.listdir(filename):
    im = Image.open(filename + pic)
    im2 = np.array(im)
    train_data.append(im2)
train_data = np.array(train_data)


for pic in os.listdir(filename2):
    im = Image.open(filename + pic)
    im2 = np.array(im)
    test_data.append(im2)
test_data = np.array(test_data)

for i in range(0, 3360):
    yjtrainlabel.append(yj[i])
for i in range(3360, 3840):
    yjtestlabel.append(yj[i])  # 整理标签


data1 = np.array(yjtrainlabel)
values1 = data1
label_encoder1 = LabelEncoder()
integer_encoded1 = label_encoder1.fit_transform(values1)
# print(integer_encoded)

onehot_encoder1 = OneHotEncoder(sparse=False)
integer_encoded1 = integer_encoded1.reshape(len(integer_encoded1), 1)
onehot_encoded1 = onehot_encoder1.fit_transform(integer_encoded1)
Ytrain_onehot = np.array(onehot_encoded1)

print("Ytrain_onehot-----------------", Ytrain_onehot)


data2 = yjtestlabel
values2 = np.array(data2)
# print(values)

label_encoder2 = LabelEncoder()
integer_encoded2 = label_encoder2.fit_transform(values2)
# print(integer_encoded)

onehot_encoder2 = OneHotEncoder(sparse=False)
integer_encoded2 = integer_encoded2.reshape(len(integer_encoded2), 1)
onehot_encoded2 = onehot_encoder2.fit_transform(integer_encoded2)
print("onehot_encoded2----------------", onehot_encoded2)


# 初始化过滤器
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


# 初始化偏置，初始化时，所有值是0.1
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# 卷积运算，strides表示每一维度滑动的步长，一般strides[0]=strides[3]=1
# 第四个参数可选"Same"或"VALID"，“Same”表示边距使用全0填充
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# 池化运算
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 创建x占位符，用于临时存放MNIST图片的数据，
# [None, 784]中的None表示不限长度，而784则是一张图片的大小（28×28=784）
x = tf.placeholder(tf.float32, [None, 28, 28])
# y_存的是实际图像的标签，即对应于每张输入图片实际的值
y_ = tf.placeholder(tf.float32, [None, 8])

# 将图片从784维向量重新还原为28×28的矩阵图片,
# 原因参考卷积神经网络模型图，最后一个参数代表深度，
# 因为MNIST是黑白图片，所以深度为1,
# 第一个参数为-1,表示一维的长度不限定，这样就可以灵活设置每个batch的训练的个数了
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积
# 将过滤器设置成5×5×1的矩阵，
# 其中5×5表示过滤器大小，1表示深度，因为MNIST是黑白图片只有一层。所以深度为1
# 32表示卷积在经过每个5×5大小的过滤器后可以算出32个特征，即经过卷积运算后，输出深度为32
W_conv1 = weight_variable([5, 5, 1, 32])
# 有多少个输出通道数量就有多少个偏置
b_conv1 = bias_variable([32])
# 使用conv2d函数进行卷积计算，然后再用ReLU作为激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv3 = weight_variable([5, 5, 32, 32])
# 有多少个输出通道数量就有多少个偏置
b_conv3 = bias_variable([32])
# 使用conv2d函数进行卷积计算，然后再用ReLU作为激活函数
h_conv3 = tf.nn.relu(conv2d(h_conv1, W_conv3) + b_conv3)

W_conv5 = weight_variable([5, 5, 32, 32])
# 有多少个输出通道数量就有多少个偏置
b_conv5 = bias_variable([32])
# 使用conv2d函数进行卷积计算，然后再用ReLU作为激活函数
h_conv5 = tf.nn.relu(conv2d(h_conv3, W_conv5) + b_conv5)


h_pool1=max_pool_2x2(h_conv5)
# 卷积以后再经过池化操作
#h_pool1 = max_pool_2x2(h_conv1)





# 第二层卷积
# 因为经过第一层卷积运算后，输出的深度为32,所以过滤器深度和下一层输出深度也做出改变
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

W_conv4 = weight_variable([5, 5, 64, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.relu(conv2d(h_conv2, W_conv4) + b_conv4)

h_pool2 = max_pool_2x2(h_conv4)

# 全连接层
# 经过两层卷积后，图片的大小为7×7（第一层池化后输出为（28/2）×（28/2），
# 第二层池化后输出为（14/2）×（14/2））,深度为64，
# 我们在这里加入一个有1024个神经元的全连接层，所以权重W的尺寸为[7 * 7 * 64, 1024]
W_fc1 = weight_variable([7 * 7 * 64, 1024])
# 偏置的个数和权重的个数一致
b_fc1 = bias_variable([1024])
# 这里将第二层池化后的张量（长：7 宽：7 深度：64） 变成向量（跟上一节的Softmax模型的输入一样了）
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 使用ReLU激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
# 为了减少过拟合，我们在输出层之前加入dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
# 全连接层输入的大小为1024,而我们要得到的结果的大小是10（0～9），
# 所以这里权重W的尺寸为[1024, 10]
W_fc2 = weight_variable([1024, 8])
b_fc2 = bias_variable([8])
# 最后都要经过Softmax函数将输出转化为概率问题
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 损失函数和损失优化
#cross_entropy = tf.reduce_sum(y_ * tf.log(y_conv))
# coss_entropy=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=y_conv)
cross_entropy = tf.reduce_mean (
    tf.nn.softmax_cross_entropy_with_logits (labels = y_, logits = y_conv))#损失函数,交叉熵方法
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step= tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

# 测试准确率,跟Softmax回归模型的一样
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 开始训练
with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver(max_to_keep=3)
    # 训练两万次
    for i in range(20000):
        # # 每次获取50张图片数据和对应的标签
        # batch1 = train_data.next_batch(50)
        # batch2=test_data.next_batch(50)
        # # 每训练100次，我们打印一次训练的准确
        train_accuracy = sess.run(accuracy, feed_dict={x: train_data, y_: Ytrain_onehot, keep_prob: 0.5})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: train_data, y_: Ytrain_onehot, keep_prob: 0.5})# 这里是真的训练，将数据传入
        #train_step.run(feed_dict={x: train_data, y_: Ytrain_onehot, keep_prob: 0.5})
        test_accuracy = sess.run(accuracy, feed_dict={x: test_data, y_: onehot_encoded2, keep_prob: 1.0})
        test_accuracy_list.append(test_accuracy)
        if i%5000 == 0:
            saver.save(sess, CKPT_DIR+'model.ckpt',global_step=i)
        if i % 100 == 0:
            print(test_accuracy)


































