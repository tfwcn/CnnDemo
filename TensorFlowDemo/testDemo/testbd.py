import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./models')
from models.BaiduModel import BaiduModel
from models.ModelHelper import ModelHelper

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', './data',
                           """Path to the MNIST data directory.""")

if __name__ == "__main__":
    char_list = "0123456789abcdefghijklmnopqrstuvwxyz"
    modelHelper = ModelHelper()
    # 读取文件列表
    pathDir = os.listdir('./img')
    # 定义特征与标签
    features = ['./img/'+path for path in pathDir]
    labels = [path[0:4] for path in pathDir]
    features = pd.Series(features)
    labels = pd.Series(labels)
    # 把列添加到表格
    data = pd.DataFrame({'文件路径': features, '标签': labels})
    # 随机排序
    data = data.reindex(np.random.permutation(data.index))

    features = data['文件路径'].values
    labels = data['标签'].values
    # print(len(labels), len(char_list))
    labels = modelHelper.hot_one(labels, char_list, 4)
    # print(labels[0])
    # print(data['文件路径'][0:10])
    # print(data['标签'][0:10])
    # 取后500条记录昨测试集，shuffle=True随机文件列表
    train_features, train_labels = tf.train.slice_input_producer(
        [features[0:-500], labels[0:-500]], shuffle=True)
    train_features = tf.read_file(train_features)
    test_features, test_labels = tf.train.slice_input_producer(
        [features[-500:], labels[-500:]], shuffle=True)
    test_features = tf.read_file(test_features)
    # 读取图片
    train_features = tf.image.decode_jpeg(
        train_features, channels=3)  # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    train_features = tf.image.rgb_to_grayscale(train_features)  # 转灰度
    train_features = tf.image.resize_images(train_features, [160, 60])  # 缩放图片
    train_features = tf.div(train_features, 255)

    test_features = tf.image.decode_jpeg(
        test_features, channels=3)  # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    test_features = tf.image.rgb_to_grayscale(test_features)  # 转灰度
    test_features = tf.image.resize_images(test_features, [160, 60])  # 缩放图片
    test_features = tf.div(test_features, 255)

    # 分批训练
    num_preprocess_threads = 4  # 读取线程数
    batch_size = 10  # 每次训练数据量
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
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout概率
    baiduModel = BaiduModel()
    y = baiduModel.create_model(xs, keep_prob)

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
        sess.run(tf.global_variables_initializer())  # 初始化变量
        #加载已训练数据
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.data_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        # 使用start_queue_runners之后，才会开始填充队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(1000):
            train_features_batch_data, train_labels_batch_data = sess.run(
                [train_features, train_labels])
            # print(train_features_batch_data)
            sess.run(train_step, feed_dict={
                     xs: train_features_batch_data, ys: train_labels_batch_data, keep_prob: 0.5})
            print(sess.run(cross_entropy, feed_dict={
                xs: train_features_batch_data, ys: train_labels_batch_data, keep_prob: 1}))
            if step % 10 == 0:
                test_features_batch_data, test_labels_batch_data = sess.run(
                    [test_features, test_labels])
                predict_num = sess.run(accuracy, feed_dict={
                    xs: test_features_batch_data, ys: test_labels_batch_data, keep_prob: 1})
                print(step, predict_num)
                # checkpoint_path = os.path.join(FLAGS.data_dir, 'model.ckpt')
                # saver.save(sess, checkpoint_path)
                if predict_num == 1:
                    pass
