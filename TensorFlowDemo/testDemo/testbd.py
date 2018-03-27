import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./models')
from models.BaiduModel import BaiduModel

if __name__ == "__main__":
    char_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z']
    print(len(char_list))
    # 读取文件列表
    pathDir = os.listdir('./img')
    # 定义特征与标签
    features = [path for path in pathDir]
    labels = [path[0:4] for path in features]
    features = pd.Series(features)
    labels = pd.Series(labels)
    # 把列添加到表格
    data = pd.DataFrame({'文件路径': features, '标签': labels})
    # 随机排序
    data = data.reindex(np.random.permutation(data.index))
    # print(data['文件路径'][0:10])
    # print(data['标签'][0:10])
    # 取后500条记录昨测试集，shuffle=True随机文件列表
    train_features, train_labels = tf.train.slice_input_producer(
        [data['文件路径'][0:-500].values, data['标签'][0:-500].values], shuffle=True)
    train_features = tf.read_file(train_features)
    test_features, test_labels = tf.train.slice_input_producer(
        [data['文件路径'][-500:].values, data['标签'][-500:].values], shuffle=True)
    test_features = tf.read_file(test_features)
    # 读取图片
    train_features = tf.image.decode_jpeg(
        train_features, channels=3)  # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    train_features = tf.image.rgb_to_grayscale(train_features)  # 转灰度
    train_features = tf.image.resize_images(train_features, [160, 60])  # 缩放图片

    test_features = tf.image.decode_jpeg(
        test_features, channels=3)  # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    test_features = tf.image.rgb_to_grayscale(train_features)  # 转灰度
    test_features = tf.image.resize_images(test_features, [160, 60])  # 缩放图片

    # 分批训练
    num_preprocess_threads = 4  # 读取线程数
    batch_size = 20  # 每次训练数据量
    min_queue_examples = 1000  # 最小数据量
    train_features, train_labels = tf.train.shuffle_batch(
        [train_features, train_labels],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    test_features, test_labels = tf.train.shuffle_batch(
        [test_features, test_labels],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    # 生成神经网络结构
    xs = tf.placeholder(tf.float32, [None, 160, 60, 1])
    ys = tf.placeholder(tf.float32, [None, 4*36])
    baiduModel = BaiduModel()
    y = baiduModel.create_model(xs, 0.5)

    cross_entropy = tf.multiply(tf.reduce_sum(
        ys * tf.log(y)), -1, name="cross_entropy")  # 损失函数，交叉熵
    train_step = tf.train.AdamOptimizer(
        1e-4).minimize(cross_entropy, name="train_step")  # ADAM优化器来做梯度最速下降，自动调整里面的变量
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(
        ys, 1), name="correct_prediction")  # 比较结果
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, "float"), name="accuracy")  # 求平均

    # 开始训练
    with tf.Session() as sess:
        sess.run(train_step, feed_dict={xs: train_features, ys: train_labels})
        for step in range(1000):
            if step % 100 == 0:
                print(step, sess.run(accuracy, feed_dict={
                      xs: test_features, ys: test_labels}))
