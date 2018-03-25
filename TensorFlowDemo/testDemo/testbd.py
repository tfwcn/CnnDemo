import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./models')
from models.BaiduModel import BaiduModel

if __name__ == "__main__":
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
    # 随机文件列表
    train_features, train_labels = tf.train.slice_input_producer(
        [data['文件路径'][0:-500].values, data['标签'][0:-500].values], shuffle=True)
    train_features = tf.read_file(train_features)
    print(train_features)
    print(train_labels)
    test_features, test_labels = tf.train.slice_input_producer(
        [data['文件路径'][-500:].values, data['标签'][-500:].values], shuffle=True)
    test_features = tf.read_file(test_features)
    print(test_features)
    print(test_labels)
    train_features = tf.image.decode_jpeg(
        train_features, channels=3)  # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    # train_features = tf.image.rgb_to_grayscale(train_features)  # 转灰度
    train_features = tf.image.resize_images(train_features, [160, 60])  # 缩放图片
    test_features = tf.image.decode_jpeg(
        train_features, channels=3)  # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    # test_features = tf.image.rgb_to_grayscale(train_features)  # 转灰度
    test_features = tf.image.resize_images(train_features, [160, 60])  # 缩放图片
