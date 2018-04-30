import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./models')
from models.BaiduModel import BaiduModel
from models.ModelHelper import ModelHelper

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', './data/2018-04-06-2',
                           """Path to the MNIST data directory.""")


class BdPredictHelper():
    def __init__(self):
        # 初始化
        self.char_list = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.modelHelper = ModelHelper()
        self.xs = None
        self.ys = None
        self.keep_prob = None
        self.predict_index = None
        self.sess = None
        return

    def init(self):
        # 生成神经网络结构
        self.xs = tf.placeholder(tf.float32, [None, 160, 60, 1])
        self.ys = tf.placeholder(tf.float32, [None, 4*36])
        self.keep_prob = tf.placeholder(
            tf.float32, name="keep_prob")  # dropout概率
        baiduModel = BaiduModel()
        y = baiduModel.create_model(self.xs, self.keep_prob)

        predict = tf.reshape(
            y, [-1, baiduModel.char_num, baiduModel.classes])
        self.predict_index = tf.argmax(predict, 2)

        # 开始训练
        # with tf.Session() as sess:
        if True:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())  # 初始化变量
            # 加载已训练数据
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAGS.data_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("读取模型完成")
        return

    def predict(self, path_file):
        # 读取文件列表
        # path_file = "./img2/2kxw.jpg"
        # labels = "2kxw"
        # labels = self.modelHelper.hot_one_one(labels, self.char_list, 4)
        # 取后500条记录昨测试集，shuffle=True随机文件列表
        # test_features = tf.read_file(path_file)
        # 读取图片
        test_features = tf.image.decode_jpeg(
            path_file, channels=3)  # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
        test_features = tf.image.rgb_to_grayscale(test_features)  # 转灰度
        test_features = tf.image.resize_images(
            test_features, [160, 60])  # 缩放图片
        test_features = tf.div(test_features, 255)
        test_features = tf.reshape(test_features, [-1, 160, 60, 1])

        # 开始训练
        # with tf.Session() as sess:
        if True:
            img = self.sess.run(test_features)
            # predict_num = sess.run(accuracy, feed_dict={
            #     xs: img, ys: labels, keep_prob: 1})
            # print(predict_num)
            predict_value = self.sess.run(self.predict_index, feed_dict={
                self.xs: img, self.keep_prob: 1})
            result = str(self.char_list[predict_value[0][0]])
            result += str(self.char_list[predict_value[0][1]])
            result += str(self.char_list[predict_value[0][2]])
            result += str(self.char_list[predict_value[0][3]])
            # print(labels)
            # print(result)
            return result

    def close(self):
        self.sess.close()


if __name__ == "__main__":
    bdPredictHelper = BdPredictHelper()
    bdPredictHelper.init()
    pathDir = os.listdir('./data/img')
    import shutil
    for allDir in pathDir:
        label = allDir[0:4]
        print(allDir)
        result = bdPredictHelper.predict(open("./data/img/"+allDir, 'rb').read())
        if label == result:
            shutil.copyfile("./data/img/"+allDir, "./data/img_true/"+allDir)  # 复制文件
        else:
            shutil.copyfile("./data/img/"+allDir, "./data/img_false/"+allDir)  # 复制文件
    print(result)
