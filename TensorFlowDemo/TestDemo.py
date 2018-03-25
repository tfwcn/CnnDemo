#import input_data
import tensorflow as tf
import os
import scipy.io as sio
import scipy.misc as misc
import numpy
import math

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'e:/imdb_crop',
                           """Path to the MNIST data directory.""")


def run(image):
    """训练"""
    #tf.reset_default_graph()#重置图
    #image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)#读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    sess = tf.InteractiveSession()#启动Session，与底层通信
    print(image.eval(session=sess))#识别
    return
        
