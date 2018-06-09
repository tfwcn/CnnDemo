#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import align.detect_face
import scipy.io as sio
import math
import pandas as pd
import fnmatch
import cv2

class picture_train():
    # 构造函数 初始化
    def __init__(self):
        
        self.image_size=227
        self.margin=40
        self.gpu_memory_fraction=0.3
        # 加载模型
        #self.g1=tf.Graph()
        #self.sess1 = tf.Session(graph=self.g1)
        self.isRun = 0
        self.FLAGS = tf.app.flags.FLAGS
        tf.app.flags.DEFINE_string('data_dir', '/home/hillsun/jzy/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification',
                           """Path to the data directory.""")
        tf.app.flags.DEFINE_string('model_dir', '/home/hillsun/dxh/PictureClassifier/model_data/',
                           """Path to the model directory.""")
        #扣人脸
        self.gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            
        self.graph2=tf.Graph()
        self.sess2 = tf.Session(graph=self.graph2, config=tf.ConfigProto(
            gpu_options=self.gpu_options, log_device_placement=False))
        with self.graph2.as_default():
            with self.sess2.as_default():
                # tf.ConfigProto一般用在创建session的时候。用来对session进行参数配置
                # 多层前馈神经元网络，用来检测人脸
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(
                    self.sess2, None)
        return

    def dense_to_one_hot(self,labels_dense, num_classes=2):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = len(labels_dense) #数组长度
        index_offset = np.arange(num_labels) * num_classes #根据数组生成序号
        labels_one_hot = np.zeros([num_labels, num_classes]) #初始化二维数组
        for i in range(num_labels):#赋值
            if math.isnan(labels_dense[i]):
                continue
            labels_one_hot.flat[int(index_offset[i] + labels_dense[i])] = 1 #一维化赋值
        return labels_one_hot.tolist()

    def readFiles2(self):
        #读取地址和性别
        dataset0=pd.read_csv(self.FLAGS.data_dir+"/fold_0_data.txt", sep="\t")
        dataset1=pd.read_csv(self.FLAGS.data_dir+"/fold_1_data.txt", sep="\t")
        dataset2=pd.read_csv(self.FLAGS.data_dir+"/fold_2_data.txt", sep="\t")
        dataset3=pd.read_csv(self.FLAGS.data_dir+"/fold_3_data.txt", sep="\t")
        dataset4=pd.read_csv(self.FLAGS.data_dir+"/fold_4_data.txt", sep="\t")
        #将4个文件合成1个文件
        dataset=pd.concat([dataset0, dataset1, dataset2, dataset3, dataset4], axis=0)
        dataset=dataset[dataset.gender.notnull()&dataset.face_id.notnull()]
        user_id=dataset["user_id"]
        original_image=dataset["original_image"]
        face_id=dataset["face_id"]
        gender=dataset["gender"]
        age=dataset["age"]
        #整理
        # print(len(dataset))
        # print(user_id.iloc[0])
        # print(original_image)
        # print(face_id)
        # print(gender)
        TEST_IMAGE_PATHS = [ os.path.join(self.FLAGS.data_dir,'aligned/%s/landmark_aligned_face.%s.%s' % (user_id.iloc[i],face_id.iloc[i],original_image.iloc[i])) for i in range(0, len(dataset)) ]
        TEST_GENDER = [ 1 if gender.iloc[i]=='f' else 0 for i in range(0, len(dataset)) ]
        TEST_GENDER = self.dense_to_one_hot(TEST_GENDER,2) #转化成1维数组

        # print(TEST_IMAGE_PATHS)
        # print(TEST_GENDER)

        # slice_input_producer会产生一个文件名队列
        #filename_queue = tf.train.string_input_producer(TEST_IMAGE_PATHS) #文件列表
        filename_queue, label_queue,age_queue = tf.train.slice_input_producer([TEST_IMAGE_PATHS,TEST_GENDER,age],shuffle=True,num_epochs=None) #文件列表
        #读取文件
        #reader = tf.WholeFileReader()
        #key, value = reader.read(filename_queue)
        value = tf.read_file(filename_queue)
        label = label_queue
        image = value
        image = tf.image.decode_jpeg(image, channels=3)#读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
        #image = tf.image.grayscale_to_rgb(image)#转彩色
        image = tf.image.resize_images(image,[227,227]) #缩放图片
        age_value = age_queue
        #随机
        num_preprocess_threads = 4 #读取线程数
        batch_size = 1 #每次训练数据量
        min_queue_examples = 1 #最小数据量
        image, label, age_value = tf.train.shuffle_batch(
            [image, label,age_value],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
        print(label._shape)
        print(image._shape)
        print(age_value._shape)
        return label,image,age_value

    def run(self):
        print("aaaa")
        tf.reset_default_graph()#重置图
        print("bbbb")
        with tf.Session() as sess:
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))#tf.InteractiveSession()#启动Session，与底层通
            print("fff")
            label,image,age_value = self.readFiles2()#读取文件
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            
            for i in range(17000):
                #print("fff1")
                img_data, label_data,age_data = sess.run([image, label,age_value])
                print(label_data)
                print("age_data:%s" %age_data)
                print(age_data[0])
                num = str(age_data).index("'")
                print("num:%d"%num)
                
                s=str(age_data)[num+1:]
                num=s.index("'")
                s=s[0:num]
                print("s:%s"%s)
                if s.replace(' ', '')!="(0,2)":
                    newfile="/home/hillsun/dxh/images_xlj/"+str(i)+ "_" + str(int(label_data[0][0])) + "_" + s + ".jpg"
                    img_data = np.reshape(img_data, [227,227,3])
                    #print(img_data)
                    print(newfile)
                    cv2.imwrite(newfile, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))
                    print("write sucessfull!")
                    print("i:%d"%i)
        return

def main():
    obj=picture_train()
    obj.run()
    
if __name__ == '__main__':
    main()