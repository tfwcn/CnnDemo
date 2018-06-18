# coding:utf-8
#import input_data
import tensorflow as tf
import os
import scipy.io as sio
import scipy.misc as misc
import numpy
import math
import pandas as pd
from nets import inception
from nets import alexnet

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('data_dir', 'e:/imdb_crop',
#                            """Path to the MNIST data directory.""")
# tf.app.flags.DEFINE_string('data_dir', './imdb_crop',
#                            """Path to the MNIST data directory.""")
tf.app.flags.DEFINE_string('data_dir', '/home/hillsun/jzy/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification',
                           """Path to the MNIST data directory.""")

# mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)#加载图片


def weight_variable(shape, name):
    """初始化权重"""
    initial = tf.truncated_normal(shape, stddev=0.01)  # 0.1的正态分布
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    """初始化偏置"""
    initial = tf.constant(0.1, shape=shape)  # 所有值0.1
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    """卷积，步长为1，,padding为0"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, name):
    """池化，2*2，步长为2，,padding为0"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name=name)


def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = len(labels_dense)  # 数组长度
    index_offset = numpy.arange(num_labels) * num_classes  # 根据数组生成序号
    labels_one_hot = numpy.zeros([num_labels, num_classes])  # 初始化二维数组
    for i in range(num_labels):  # 赋值
        if math.isnan(labels_dense[i]):
            continue
        labels_one_hot.flat[int(
            index_offset[i] + labels_dense[i])] = 1  # 一维化赋值
    return labels_one_hot.tolist()


def readFiles():
        # 读取地址和性别
    dataset = sio.loadmat(FLAGS.data_dir+"/imdb.mat")
    filepath = dataset["imdb"][0, 0]["full_path"][0]
    gender = dataset["imdb"][0, 0]["gender"][0]
    # print('filepath:%s' % len(filepath))
    # print('gender:%s' % len(gender))
    # 整理
    TEST_IMAGE_PATHS = [os.path.join(
        FLAGS.data_dir, filepath[i][0]) for i in range(0, len(gender))]
    TEST_GENDER = [gender[i] for i in range(0, len(gender))]
    TEST_GENDER = dense_to_one_hot(TEST_GENDER, 2)  # 转化成1维数组

    # slice_input_producer会产生一个文件名队列
    # filename_queue = tf.train.string_input_producer(TEST_IMAGE_PATHS) #文件列表
    filename_queue, label_queue = tf.train.slice_input_producer(
        [TEST_IMAGE_PATHS, TEST_GENDER], shuffle=True, num_epochs=None)  # 文件列表
    # 读取文件
    #reader = tf.WholeFileReader()
    #key, value = reader.read(filename_queue)
    value = tf.read_file(filename_queue)
    label = label_queue
    image = value
    # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.grayscale_to_rgb(image)#转彩色
    image = tf.image.resize_images(image, [160, 160])  # 缩放图片
    # 随机
    num_preprocess_threads = 4  # 读取线程数
    batch_size = 50  # 每次训练数据量
    min_queue_examples = 10000  # 最小数据量
    print(label._shape)
    print(image._shape)
    image, label = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    print(label._shape)
    print(image._shape)
    image= tf.divide(image,255.0)
    return label, image


def readFiles2():
    # 读取地址和性别
    dataset0 = pd.read_csv(FLAGS.data_dir+"/fold_0_data.txt", sep="\t")
    dataset1 = pd.read_csv(FLAGS.data_dir+"/fold_1_data.txt", sep="\t")
    dataset2 = pd.read_csv(FLAGS.data_dir+"/fold_2_data.txt", sep="\t")
    dataset3 = pd.read_csv(FLAGS.data_dir+"/fold_3_data.txt", sep="\t")
    dataset4 = pd.read_csv(FLAGS.data_dir+"/fold_4_data.txt", sep="\t")
    dataset = pd.concat([dataset0, dataset1, dataset2,
                         dataset3, dataset4], axis=0)
    dataset = dataset[dataset.gender.notnull() & dataset.face_id.notnull()]
    dataset1 = dataset[(dataset.gender=='f') & (-dataset.age.isin(["(0,2)","(0, 2)"]))].iloc[0:5000]
    dataset2 = dataset[(dataset.gender=='m') & (-dataset.age.isin(["(0,2)","(0, 2)"]))].iloc[0:5000]
    dataset = pd.concat([dataset1, dataset2], axis=0)
    user_id = dataset["user_id"]
    original_image = dataset["original_image"]
    face_id = dataset["face_id"]
    gender = dataset["gender"]
    # 整理
    # print(len(dataset))
    # print(user_id.iloc[0])
    # print(original_image)
    # print(face_id)
    # print(gender)
    TEST_IMAGE_PATHS = [os.path.join(FLAGS.data_dir, 'aligned/%s/landmark_aligned_face.%s.%s' % (
        user_id.iloc[i], face_id.iloc[i], original_image.iloc[i])) for i in range(0, len(dataset))]
    TEST_GENDER = [1 if gender.iloc[i] ==
                   'f' else 0 for i in range(0, len(dataset))]
    TEST_GENDER = dense_to_one_hot(TEST_GENDER, 2)  # 转化成1维数组

    # print(TEST_IMAGE_PATHS)
    # print(TEST_GENDER)

    # slice_input_producer会产生一个文件名队列
    # filename_queue = tf.train.string_input_producer(TEST_IMAGE_PATHS) #文件列表
    filename_queue, label_queue = tf.train.slice_input_producer(
        [TEST_IMAGE_PATHS, TEST_GENDER], shuffle=True, num_epochs=None)  # 文件列表
    # 读取文件
    #reader = tf.WholeFileReader()
    #key, value = reader.read(filename_queue)
    value = tf.read_file(filename_queue)
    label = label_queue
    image = value
    # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.grayscale_to_rgb(image)#转彩色
    image = tf.image.resize_images(image, [160, 160])  # 缩放图片
    # 随机
    num_preprocess_threads = 4  # 读取线程数
    batch_size = 50  # 每次训练数据量
    min_queue_examples = 5000  # 最小数据量
    image, label = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    # print(label._shape)
    # print(image._shape)
    image= tf.divide(image,255.0)
    return label, image

def readFiles4():
    # 读取文件列表
    # file_dir="/home/hillsun/dxh/images_hyj_7243/man_4314/"
    file_dir="/home/hillsun/dxh/images_yzr_avg/"
    # pathDir = os.listdir(file_dir)[0:2900]
    pathDir = os.listdir(file_dir)
    # 定义特征与标签
    TEST_IMAGE_PATHS = [file_dir+path for path in pathDir]
    TEST_GENDER = [int(path.split('_')[1])-1 for path in pathDir]
    # # 读取文件列表
    # file_dir="/home/hillsun/dxh/images_hyj_7243/women_2929/"
    # pathDir = os.listdir(file_dir)[0:2900]
    # # 定义特征与标签
    # TEST_IMAGE_PATHS += [file_dir+path for path in pathDir]
    # TEST_GENDER += [int(path.split('_')[1])-1 for path in pathDir]
    TEST_GENDER = dense_to_one_hot(TEST_GENDER, 2)  # 转化成1维数组

    # print(TEST_IMAGE_PATHS)
    # print(TEST_GENDER)

    # slice_input_producer会产生一个文件名队列
    # filename_queue = tf.train.string_input_producer(TEST_IMAGE_PATHS) #文件列表
    filename_queue, label_queue = tf.train.slice_input_producer(
        [TEST_IMAGE_PATHS, TEST_GENDER], shuffle=True, num_epochs=None)  # 文件列表
    # 读取文件
    #reader = tf.WholeFileReader()
    #key, value = reader.read(filename_queue)
    value = tf.read_file(filename_queue)
    label = label_queue
    image = value
    # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.grayscale_to_rgb(image)#转彩色
    image = tf.image.resize_images(image, [160, 160])  # 缩放图片
    # 随机
    num_preprocess_threads = 4  # 读取线程数
    batch_size = 50  # 每次训练数据量
    min_queue_examples = 5000  # 最小数据量
    image, label = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    # print(label._shape)
    # print(image._shape)
    image= tf.divide(image,255.0)
    return label, image

def readFiles3():
    """识别用"""
    filename = tf.placeholder(tf.string, name="filename")  # 定义变量，输入值
    # 读取文件
    image = tf.read_file(filename)
    # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.grayscale_to_rgb(image)#转彩色
    image = tf.image.resize_images(image, [160, 160])  # 缩放图片
    image = tf.reshape(image, [-1, 160, 160, 3], name="input_image")
    image= tf.divide(image,255.0)
    return filename, image

def init():
    """初始化神经网络"""
    x = tf.placeholder(
        "float", shape=[None, 160, 160, 3], name="x")  # 定义变量，输入值
    y_ = tf.placeholder("float", shape=[None, 2], name="y_")  # 定义变量，输出值
    keep_prob = tf.placeholder("float", name="keep_prob")  # dropout概率
    # x = image
    # y_ = label
    # 第一层160*160*36
    # x_image = tf.reshape(x,
    # [-1,160,160,3],name="x_image")#转成四维向量，大小160*160，颜色通道3，对应输入数量

    h_conv1 = slim.conv2d(x, 20, [7, 7], [4, 4], scope='Conv2d_1_5x5',
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01))
    h_pool1 = slim.max_pool2d(h_conv1, [2, 2], scope='MaxPool_1_2x2', stride=2)
    h_pool1 = tf.nn.local_response_normalization(
        h_pool1, 5, alpha=0.0001, beta=0.75, name='norm1')
    # h_pool1 = tf.nn.dropout(h_pool1, keep_prob, name="h_pool1_drop")
    # print(h_pool1.shape)

    # 第二层40*40*64
    h_conv2 = slim.conv2d(h_pool1, 30, [5, 5], scope='Conv2d_2_5x5',
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01))
    h_pool2 = slim.max_pool2d(h_conv2, [2, 2], scope='MaxPool_2_2x2', stride=2)
    h_pool2 = tf.nn.local_response_normalization(
        h_pool2, 5, alpha=0.0001, beta=0.75, name='norm2')
    # h_pool2 = tf.nn.dropout(h_pool2, keep_prob, name="h_pool2_drop")

    # print(h_pool2.shape)
    # 20 x 20 x 320
    end_point = 'Mixed_4a'
    with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
            # 128
            branch_0 = slim.conv2d(h_pool2, 30, [1, 1],
                                   weights_initializer=tf.truncated_normal_initializer(
                                       0.0, 0.01),
                                   scope='Conv2d_0a_1x1')
            # 160
            branch_0 = slim.conv2d(branch_0, 40, [3, 3], stride=2,
                                   scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
            # 64
            branch_1 = slim.conv2d(h_pool2, 25, [1, 1],
                                   weights_initializer=tf.truncated_normal_initializer(
                                       0.0, 0.01),
                                   scope='Conv2d_0a_1x1')
            # 96
            branch_1 = slim.conv2d(branch_1, 30, [3, 3], scope='Conv2d_0b_3x3')
            branch_1 = slim.conv2d(
                branch_1, 30, [3, 3], stride=2, scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.max_pool2d(
                h_pool2, [2, 2], stride=2, scope='MaxPool_1a_3x3')
        print(branch_0.shape)
        print(branch_1.shape)
        print(branch_2.shape)
        mixed_4a = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

    # 全连接层20*20*320
    W_fc1 = weight_variable([5 * 5 * 100, 100], name="W_fc1")  # 定义权重，输出数量1024
    b_fc1 = bias_variable([100], name="b_fc1")  # 定义偏置，输出数量1024

    h_pool2_flat = tf.reshape(
        mixed_4a, [-1, 5 * 5 * 100], name="h_pool2_flat")  # 池化结果7*7*64转一维数组
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) +
                       b_fc1, name="h_fc1")  # 用relu激活函数

    # print(h_fc1.shape)
    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")

    # 输出层也是全连接层
    W_fc2 = weight_variable([100, 2], name="W_fc2")
    b_fc2 = bias_variable([2], name="b_fc2")

    # print(b_fc2.shape)
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) +
                           b_fc2, name="y_conv")  # 用softmax激活函数
    # print(y_.shape)
    # print(y_conv.shape)
    cross_entropy = tf.multiply(tf.reduce_sum(
        y_ * tf.log(y_conv)), -1, name="cross_entropy")  # 损失函数，交叉熵
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,name="train_step")#ADAM优化器来做梯度最速下降，自动调整里面的变量
    train_step = tf.train.AdamOptimizer(
        1e-4).minimize(cross_entropy, name="train_step")  # ADAM优化器来做梯度最速下降，自动调整里面的变量
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(
        y_, 1), name="correct_prediction")  # 比较结果
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, "float"), name="accuracy")  # 求平均
    return

def init2():
    """初始化神经网络"""
    #tf.reset_default_graph()#重置图
    x = tf.placeholder(
        "float", shape=[None, 227, 227, 3], name="x")  # 定义变量，输入值
    y_ = tf.placeholder("float", shape=[None, 2], name="y_")  # 定义变量，输出值
    # x = image
    # y_ = label
    
    #第一层160*160*32
    h_conv1 = slim.conv2d(x, 96, [7, 7], [4, 4], padding='VALID', scope='conv1',
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                            biases_initializer=tf.constant_initializer(0.0))
    h_pool1 = slim.max_pool2d(h_conv1, 3, 2, padding='VALID', scope='pool1')
    h_pool1 = tf.nn.local_response_normalization(h_pool1, 5, alpha=0.0001, beta=0.75, name='norm1')  

    #80*80*64
    h_conv2 = slim.conv2d(h_pool1, 256, [5, 5],[1,1], padding='SAME', scope='Conv2d_2_5x5',
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                            biases_initializer=tf.constant_initializer(0.0))
    h_pool2 = slim.max_pool2d(h_conv2, 3, 2, padding='VALID', scope='MaxPool_2_2x2')
    h_pool2 = tf.nn.local_response_normalization(h_pool2, 5, alpha=0.0001, beta=0.75, name='norm2')  

    h_conv3 = slim.conv2d(h_pool2, 384, [3, 3],[1,1], padding='SAME', scope='Conv2d_3_5x5',
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                            biases_initializer=tf.constant_initializer(0.0))
    h_pool3 = slim.max_pool2d(h_conv3, 3, 2, padding='VALID', scope='MaxPool_3_2x2')
    h_pool3 = tf.nn.local_response_normalization(h_pool3, 5, alpha=0.0001, beta=0.75, name='norm3')  
    
    #全连接层40*40*200
    W_fc1 = weight_variable([6 * 6 * 384, 512],name="W_fc1")#定义权重，输出数量1024
    b_fc1 = bias_variable([512],name="b_fc1")#定义偏置，输出数量1024

    h_pool2_flat = tf.reshape(h_pool3, [-1, 6 * 6 * 384],name="h_pool2_flat")#池化结果7*7*64转一维数组
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,name="h_fc1")#用relu激活函数


    #dropout
    keep_prob = tf.placeholder("float",name="keep_prob")#dropout概率
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name="h_fc1_drop")

    #全连接层
    W_fc2 = weight_variable([512, 512],name="W_fc2")
    b_fc2 = bias_variable([512],name="b_fc2")
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name="h_fc2")#用relu激活函数
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob,name="h_fc2_drop")

    #输出层也是全连接层
    W_fc3 = weight_variable([512, 2],name="W_fc3")
    b_fc3 = bias_variable([2],name="b_fc3")

    y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3,name="y_conv")#用softmax激活函数

    cross_entropy = tf.multiply(tf.reduce_sum(y_ * tf.log(y_conv)),-1,name="cross_entropy")#损失函数，交叉熵
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,name="train_step")#ADAM优化器来做梯度最速下降，自动调整里面的变量
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1),name="correct_prediction")#比较结果
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name="accuracy")#求平均
    return

def init3():
    """初始化神经网络"""
    x = tf.placeholder(
        "float", shape=[None, 160, 160, 3], name="x")  # 定义变量，输入值
    y_ = tf.placeholder("float", shape=[None, 2], name="y_")  # 定义变量，输出值
    keep_prob = tf.placeholder("float", name="keep_prob")  # dropout概率
    # x = image
    # y_ = label
    # 第一层160*160*36
    # x_image = tf.reshape(x,
    # [-1,160,160,3],name="x_image")#转成四维向量，大小160*160，颜色通道3，对应输入数量

    h_conv1 = slim.conv2d(x, 50, [7, 7], [4, 4], scope='Conv2d_1_5x5',
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01))
    h_pool1 = slim.max_pool2d(h_conv1, [3, 3], scope='MaxPool_1_2x2', stride=2)
    h_pool1 = tf.nn.local_response_normalization(
        h_pool1, 5, alpha=0.0001, beta=0.75, name='norm1')
    h_pool1 = tf.nn.dropout(h_pool1, keep_prob, name="h_pool1_drop")
    # print(h_pool1.shape)

    # 第二层40*40*64
    h_conv2 = slim.conv2d(h_pool1, 80, [5, 5], scope='Conv2d_2_5x5',
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01))
    h_pool2 = slim.max_pool2d(h_conv2, [3, 3], scope='MaxPool_2_2x2', stride=2)
    h_pool2 = tf.nn.local_response_normalization(
        h_pool2, 5, alpha=0.0001, beta=0.75, name='norm2')
    h_pool2 = tf.nn.dropout(h_pool2, keep_prob, name="h_pool2_drop")

    # 第三层40*40*64
    h_conv3 = slim.conv2d(h_pool2, 120, [5, 5], scope='Conv2d_3_5x5',
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01))
    h_pool3 = slim.max_pool2d(h_conv3, [3, 3], scope='MaxPool_3_2x2', stride=2)
    h_pool3 = tf.nn.local_response_normalization(
        h_pool3, 5, alpha=0.0001, beta=0.75, name='norm3')
    h_pool3 = tf.nn.dropout(h_pool3, keep_prob, name="h_pool3_drop")

    print(h_pool3.shape)
    # 全连接层20*20*320
    W_fc1 = weight_variable([4 * 4 * 120, 100], name="W_fc1")  # 定义权重，输出数量1024
    b_fc1 = bias_variable([100], name="b_fc1")  # 定义偏置，输出数量1024

    h_pool3_flat = tf.reshape(
        h_pool3, [-1, 4 * 4 * 120], name="h_pool2_flat")  # 池化结果7*7*64转一维数组
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) +
                       b_fc1, name="h_fc1")  # 用relu激活函数

    # print(h_fc1.shape)
    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")

    # 输出层也是全连接层
    W_fc2 = weight_variable([100, 2], name="W_fc2")
    b_fc2 = bias_variable([2], name="b_fc2")

    # print(b_fc2.shape)
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) +
                           b_fc2, name="y_conv")  # 用softmax激活函数
    # print(y_.shape)
    # print(y_conv.shape)
    cross_entropy = tf.multiply(tf.reduce_sum(
        y_ * tf.log(y_conv)), -1, name="cross_entropy")  # 损失函数，交叉熵
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,name="train_step")#ADAM优化器来做梯度最速下降，自动调整里面的变量
    train_step = tf.train.AdamOptimizer(
        1e-4).minimize(cross_entropy, name="train_step")  # ADAM优化器来做梯度最速下降，自动调整里面的变量
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(
        y_, 1), name="correct_prediction")  # 比较结果
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, "float"), name="accuracy")  # 求平均
    return

def init4():
    """初始化神经网络"""
    x = tf.placeholder(
        "float", shape=[None, 160, 160, 3], name="x")  # 定义变量，输入值
    y_ = tf.placeholder("float", shape=[None, 2], name="y_")  # 定义变量，输出值
    keep_prob = tf.placeholder("float", name="keep_prob")  # dropout概率

    # logits, endpoints = inception.inception_resnet_v2(x, 2)
    logits, endpoints = inception.inception_v4(x, 2)

    print(logits.name)
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) +
    #                        b_fc2, name="y_conv")  # 用softmax激活函数
    # print(y_.shape)
    # print(y_conv.shape)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = y_, logits = logits)
    cross_entropy = tf.losses.get_total_loss(name="cross_entropy")    #obtain the regularization losses as well
    # cross_entropy = tf.multiply(tf.reduce_sum(
    #     y_ * tf.log(logits)), -1, name="cross_entropy")  # 损失函数，交叉熵
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,name="train_step")#ADAM优化器来做梯度最速下降，自动调整里面的变量
    train_step = tf.train.AdamOptimizer(
        1e-4).minimize(cross_entropy, name="train_step")  # ADAM优化器来做梯度最速下降，自动调整里面的变量
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(
        y_, 1), name="correct_prediction")  # 比较结果
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, "float"), name="accuracy")  # 求平均
    return

def train():
    """训练"""
    tf.reset_default_graph()  # 重置图
    label, image = readFiles4()  # 读取文件
    readFiles3()
    init3()  # 初始化
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)) as sess:  # 启动Session，与底层通信
        graph = sess.graph
        # 加载变量和操作
        x = graph.get_tensor_by_name("x:0")  # 定义变量，输入值
        y_ = graph.get_tensor_by_name("y_:0")  # 定义变量，输出值
        keep_prob = graph.get_tensor_by_name("keep_prob:0")  # dropout概率
        cross_entropy = graph.get_tensor_by_name("cross_entropy:0")  # 损失函数，交叉熵
        train_step = graph.get_operation_by_name("train_step")  # ADAM优化器来做梯度最速下降，自动调整里面的变量
        accuracy = graph.get_tensor_by_name("accuracy:0")  # 求平均
        y_conv = graph.get_tensor_by_name("y_conv:0")  # 定义变量，输出值
        # y_conv = graph.get_tensor_by_name("InceptionResnetV2/Logits/Logits/BiasAdd:0")  # 定义变量，输出值
        # y_conv = graph.get_tensor_by_name("InceptionV4/Logits/Logits/BiasAdd:0")  # 定义变量，输出值
        input_image = graph.get_tensor_by_name("input_image:0")  # 读单张图片
        filename = graph.get_tensor_by_name("filename:0")  # 图片路径
        sex_value = tf.argmax(y_conv, 1)
        saver = tf.train.Saver()
        # sess.run(tf.initialize_all_variables())#初始化变量
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()  # 初始化变量
        # 加载已训练数据
        ckpt = tf.train.get_checkpoint_state('.')
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        # 记录训练数据
        # tf.summary.histogram(cross_entropy.name + '/activations',
        # cross_entropy)
        #tf.summary.scalar(cross_entropy.name + '/sparsity',cross_entropy)
        #merged_summary_op = tf.summary.merge_all()
        #summary_writer = tf.summary.FileWriter('.', graph)
        # 使用start_queue_runners之后，才会开始填充队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 训练
        for i in range(100000):
            image_value, label_value = sess.run([image, label])
            # y_value= sess.run(y_conv,
            #         feed_dict={x: image_value, y_: label_value, keep_prob: 1.0})
            # print("y_conv:%g,%g" % (y_value[0][0],y_value[0][1]))  # 识别
            if i % 100 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={x: image_value, y_: label_value, keep_prob: 1.0})  # 识别
                print("step %d, training accuracy %g" %
                        (i, train_accuracy))
                print('训练数：%d %s' %
                        (i, cross_entropy.eval(feed_dict={x: image_value, y_: label_value, keep_prob: 1})))
            if i % 100 == 0:
                # 识别
                # file_dir="/home/hillsun/dxh/images_test2_500/"
                # file_dir="/home/hillsun/dxh/images_test_400/"
                # file_dir="/home/hillsun/dxh/images_test_jzy_500/female_500/"
                # file_dir="/home/hillsun/dxh/images_gs_71/"
                file_dir="/home/hillsun/dxh/images_align_240/"
                sum=0
                true_num=0
                for root, dirs, files in os.walk(file_dir):
                    for file in files:
                        file_path = file_dir + file
                        image2 = sess.run(input_image, feed_dict={filename:file_path})
                        a=sex_value.eval(feed_dict={x: image2, keep_prob: 1.0})
                        # if (a[0]==0 and file.split('_')[1]=="0") or (a[0]==1 and file.split('_')[1]=="1"):
                        if (a[0]==0 and file.split('_')[1]=="1") or (a[0]==1 and file.split('_')[1]=="2"):
                            true_num+=1
                        sum+=1
                # file_dir='/home/hillsun/dxh/images_test_jzy_500/male_500/'
                # for root, dirs, files in os.walk(file_dir):
                #     for file in files:
                #         file_path = file_dir + file
                #         image2 = sess.run(input_image, feed_dict={filename:file_path})
                #         a=sex_value.eval(feed_dict={x: image2, keep_prob: 1.0})
                #         # if (a[0]==0 and file.split('_')[1]=="0") or (a[0]==1 and file.split('_')[1]=="1"):
                #         if (a[0]==0 and file.split('_')[1]=="1") or (a[0]==1 and file.split('_')[1]=="2"):
                #             true_num+=1
                #         sum+=1
                print("准确率：%g" % (true_num/sum))
                # 保存模型
                checkpoint_path = os.path.join('.', 'model.ckpt')
                saver.save(sess, checkpoint_path)
            #summary_str = sess.run(merged_summary_op,feed_dict={keep_prob: 1.0})
            #summary_writer.add_summary(summary_str, i)
            train_step.run(
                feed_dict={x: image_value, y_: label_value, keep_prob: 0.3})  # 训练
        # 停止填充队列
        coord.request_stop()
        coord.join(threads)
        # Create a saver.
        #saver = tf.train.Saver(tf.all_variables())
        checkpoint_path = os.path.join('.', 'model.ckpt')
        saver.save(sess, checkpoint_path)
    return


def predict():
    """识别"""
    tf.reset_default_graph()  # 重置图
    init3()  # 初始化
    readFiles3()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)) as sess:  # 启动Session，与底层通信
        graph = sess.graph
        # 加载变量和操作
        x = graph.get_tensor_by_name("x:0")  # 定义变量，输入值
        y_ = graph.get_tensor_by_name("y_:0")  # 定义变量，输入值
        y_conv = graph.get_tensor_by_name("y_conv:0")  # 定义变量，输出值
        # y_conv = graph.get_tensor_by_name("InceptionResnetV2/Logits/Logits/BiasAdd:0")  # 定义变量，输出值
        # y_conv = graph.get_tensor_by_name("InceptionV4/Logits/Logits/BiasAdd:0")  # 定义变量，输出值
        keep_prob = graph.get_tensor_by_name("keep_prob:0")  # dropout概率
        input_image = graph.get_tensor_by_name("input_image:0")  # 读单张图片
        filename = graph.get_tensor_by_name("filename:0")  # 图片路径
        sex_value = tf.argmax(y_conv, 1)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('.')
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        # 记录训练数据
        # merged_summary_op = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter('.', graph)
        # file_dir='/home/hillsun/dxh/images_gs_71/'
        # file_dir='/home/hillsun/dxh/images_waiguoren/man_100/'
        # file_dir='/home/hillsun/dxh/images_waiguoren/women_100/'
        # file_dir='/home/hillsun/dxh/images_test_lbc_500/lbc_male/'
        # file_dir='/home/hillsun/dxh/images_test_lbc_500/lbc_female/'
        file_dir="/home/hillsun/dxh/images_align_240/"
        sum=0
        true_num=0
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                file_path = file_dir + file
                image = sess.run(input_image, feed_dict={filename:file_path})
                # print(image.shape)
                # print(x.shape)
                # accuracy = graph.get_tensor_by_name("accuracy:0")  # 求平均
                # train_accuracy = accuracy.eval(
                #     feed_dict={x: image, y_: [[1,0]], keep_prob: 1.0})  # 识别
                # print("%s training accuracy %g" %
                #         (file,train_accuracy))
                a=sex_value.eval(feed_dict={x: image, keep_prob: 1.0})
                if (a[0]==0 and file.split('_')[1]=="1") or (a[0]==1 and file.split('_')[1]=="2"):
                # if (a[0]==0 and file[3]=="1") or (a[0]==1 and file[3]=="2"):
                    true_num+=1
                sum+=1
                print("%s test accuracy %g" % (file,a))  # 识别
                if sum % 100 == 0:
                    print("%g 准确率：%g" % (sum,true_num/sum))
        print("准确率：%g" % (true_num/sum))
    return


def saveTest():
    """保存变量测试"""
    tf.reset_default_graph()  # 重置图
    # Create some variables.
    v1 = tf.Variable(tf.constant(-1, shape=[2, 3]), name="v1")
    v2 = tf.Variable(tf.constant(1, shape=[2, 3]), name="v2")
    # ...
    # Add an op to initialize the variables.
    init_op = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, save the
    # variables to disk.
    with tf.Session() as sess:
        sess.run(init_op)
        # Do some work with the model.
        # ..
        # Save the variables to disk.
        save_path = saver.save(sess,  FLAGS.data_dir + '/model.ckpt')
        print("Model saved in file: ", save_path)


def loadTest():
    """读取变量测试"""
    tf.reset_default_graph()  # 重置图
    # Create some variables.
    v1 = tf.Variable(tf.constant(0, shape=[2, 3]), name="v1")
    v2 = tf.Variable(tf.constant(0, shape=[2, 3]), name="v2")
    # ...

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk,
    # and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, FLAGS.data_dir + '/model.ckpt')
        print("Model restored.")
        # Do some work with the model
        print(v1.eval())
        print(v2.eval())


def main(argv=None):  # 运行
    train()
    # predict()
    # saveTest()
    # loadTest()

if __name__ == '__main__':
    tf.app.run()
