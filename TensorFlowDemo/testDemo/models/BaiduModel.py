import tensorflow as tf
import math
from .ModelHelper import ModelHelper


class BaiduModel():
    def __init__(self):
        # 初始化
        self.modelHelper = ModelHelper()
        self.width = 160
        self.height = 60
        # 4位验证码
        self.charNum = 4
        # 26字母+10数字
        self.classes = 36
        pass

    def create_model(self, features, keep_prob):
        # first layer
        w_conv1 = self.modelHelper.weight_variable([5, 5, 1, 32])
        b_conv1 = self.modelHelper.bias_variable([32])
        h_conv1 = tf.nn.tanh(tf.nn.bias_add(
            self.modelHelper.conv2d(features, w_conv1), b_conv1))
        h_pool1 = self.modelHelper.max_pool_2x2(h_conv1)
        h_dropout1 = tf.nn.dropout(h_pool1, keep_prob)
        conv_width = math.ceil(self.width/2)
        conv_height = math.ceil(self.height/2)

        # second layer
        w_conv2 = self.modelHelper.weight_variable([5, 5, 32, 64])
        b_conv2 = self.modelHelper.bias_variable([64])
        h_conv2 = tf.nn.tanh(tf.nn.bias_add(
            self.modelHelper.conv2d(h_dropout1, w_conv2), b_conv2))
        h_pool2 = self.modelHelper.max_pool_2x2(h_conv2)
        h_dropout2 = tf.nn.dropout(h_pool2, keep_prob)
        conv_width = math.ceil(conv_width/2)
        conv_height = math.ceil(conv_height/2)

        # third layer
        w_conv3 = self.modelHelper.weight_variable([5, 5, 64, 64])
        b_conv3 = self.modelHelper.bias_variable([64])
        h_conv3 = tf.nn.tanh(tf.nn.bias_add(
            self.modelHelper.conv2d(h_dropout2, w_conv3), b_conv3))
        h_pool3 = self.modelHelper.max_pool_2x2(h_conv3)
        h_dropout3 = tf.nn.dropout(h_pool3, keep_prob)
        conv_width = math.ceil(conv_width/2)
        conv_height = math.ceil(conv_height/2)

        # first fully layer
        conv_width = int(conv_width)
        conv_height = int(conv_height)
        w_fc1 = self.modelHelper.weight_variable(
            [64*conv_width*conv_height, 1024])
        b_fc1 = self.modelHelper.bias_variable([1024])
        h_dropout3_flat = tf.reshape(
            h_dropout3, [-1, 64*conv_width*conv_height])
        h_fc1 = tf.nn.tanh(tf.nn.bias_add(
            tf.matmul(h_dropout3_flat, w_fc1), b_fc1))
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # second fully layer
        w_fc2 = self.modelHelper.weight_variable(
            [1024, self.charNum*self.classes])
        b_fc2 = self.modelHelper.bias_variable([self.charNum*self.classes])
        y_conv = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)

        return y_conv
