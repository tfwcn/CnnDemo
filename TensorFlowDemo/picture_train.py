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

    def weight_variable(self,shape):
        """初始化权重，shape表示生成张量的维度，mean是均值，stddev是标准差。"""
        "这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。"
        "即：(生成的数-mean)>steddev*2 则重新生成"
        initial = tf.truncated_normal(shape, stddev=0.0005)#0.1的正态分布
        return tf.Variable(initial)

    def bias_variable(self,shape):
        """初始化偏置"""
        initial = tf.constant(0.0005, shape=shape)#所有值0.1
        return tf.Variable(initial)

    def conv2d(self,x, W,step,pad):
        """卷积，步长为1，,padding为0
        x为输入值，W为卷积核（权重），strides[1(固定),1[横向步长],1[竖向步长],1[固定]]
        padding ="SAME"表示输出结果大小和输入结果大小一样，即外围补零
        padding ="VALID"表示输出结果大小和输入结果小"""
        return tf.nn.conv2d(x, W, strides=[1, step, step, 1], padding=pad)

    def max_pool(self,x,kernSize,step):
        """max_pool(取池化区最大值) ，2*2，步长为2，,padding为0
        ksize[1(固定)，2(宽),2(高)，1（固定),
        strides[1(固定),1[横向步长],1[竖向步长],1[固定]]
        padding ="SAME"表示输出结果大小和输入结果大小一样，即外围补零
        padding ="VALID"表示输出结果大小和输入结果小"""
        return tf.nn.max_pool(x, ksize=[1, kernSize, kernSize, 1],
                            strides=[1, step, step, 1], padding='SAME')

    def load_and_align_data(self,fin):
        minsize = 20 # minimum size of face 人脸最小值
        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709 # scale factor 比例因子
        
        #print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)

        img = misc.imread(fin, mode='RGB') #tf.image.decode_jpeg(img_bytes, channels=3)
        #截取img.shape[346,342,3]的前两个数据，即宽高
        img_size = np.asarray(img.shape)[0:2]
        #print("--------------------img_size--------------")
        #print(img_size)
        #align.detect_face.detect_face:返回5个float,前四个值是矩形左上角及右下角左边，第5个值未知（值都是0.99...）
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
        if len(bounding_boxes) < 1:
            print("未检测到人脸 ")
            return "no face"
        nrof_faces = bounding_boxes.shape[0]
        #print('找到人脸数目为：{}'.format(nrof_faces))
        #取bounding_boxes的前四列的值，即取对角坐标
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-self.margin/2, 0)
        bb[1] = np.maximum(det[1]-self.margin/2, 0)
        bb[2] = np.minimum(det[2]+self.margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+self.margin/2, img_size[0])
         #裁剪
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]

        #print('cropped:%s' % cropped)
        #调整图片大小
        aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
        #预白化
        prewhitened = self.prewhiten(aligned)
        #img_list = []
        #img_list.append(prewhitened)
        #images = np.stack(img_list)
        return aligned

    def prewhiten(self,x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y  

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

    def readFiles(self):
        #读取地址和性别
        dataset=sio.loadmat("/home/hillsun/hyj/imdb_crop/imdb.mat")
        filepath=dataset["imdb"][0,0]["full_path"][0]
        gender=dataset["imdb"][0,0]["gender"][0]
        # print('filepath:%s' % len(filepath))
        # print('gender:%s' % len(gender))
        #整理
        TEST_IMAGE_PATHS = [ os.path.join("/home/hillsun/hyj/imdb_crop", filepath[i][0]) for i in range(0, len(gender)) ]
        TEST_GENDER = [ gender[i] for i in range(0, len(gender)) ]
        TEST_GENDER = self.dense_to_one_hot(TEST_GENDER,2) #转化成1维数组

        # slice_input_producer会产生一个文件名队列
        #filename_queue = tf.train.string_input_producer(TEST_IMAGE_PATHS) #文件列表
        filename_queue, label_queue = tf.train.slice_input_producer([TEST_IMAGE_PATHS,TEST_GENDER],shuffle=True,num_epochs=None) #文件列表
        #读取文件
        #reader = tf.WholeFileReader()
        #key, value = reader.read(filename_queue)
        value = tf.read_file(filename_queue)
        label = label_queue
        image = value
        image = tf.image.decode_jpeg(image, channels=3)#读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
        #image = tf.image.grayscale_to_rgb(image)#转彩色
        image = tf.image.resize_images(image,[227,227]) #缩放图片
        #随机
        num_preprocess_threads = 4 #读取线程数
        batch_size = 100 #每次训练数据量
        min_queue_examples = 1000 #最小数据量
        print(label._shape)
        print(image._shape)
        # image, label = tf.train.shuffle_batch(
        #     [image, label],
        #     batch_size=batch_size,
        #     num_threads=num_preprocess_threads,
        #     capacity=min_queue_examples + 3 * batch_size,
        #     min_after_dequeue=min_queue_examples)
        image, label = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
        print("label.shape:%s"%label._shape)
        print("image.shape:%s"%image._shape)
        return label,image

    def readFiles3(self,img_dir):
        #读取相片文件，存入file_list
        file_list = fnmatch.filter(os.listdir(img_dir), '*.jpg')
        file_label=[]
        for file_name in file_list:
            sex=file_name.split('_')[1]
            label_value=[]
            #print("----------------------sex-------------------")
            #print(sex)
            if sex=="1":
                label_value=[1,0]
            else:
                label_value=[0,1]
            file_label.append(label_value)

        #print(file_label)
        file_names,file_labels = tf.train.slice_input_producer([file_list,file_label],shuffle=True,num_epochs=None,name="lists")

        init = tf.initialize_all_variables()
        value = tf.read_file(img_dir + file_names)
        image = value
        label = file_labels
        fileName = file_names
        image = tf.image.decode_jpeg(image, channels=3)#读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
        #image = tf.image.grayscale_to_rgb(image)#转彩色
        image = tf.image.resize_images(image,[227,227]) #缩放图片
        #随机
        num_preprocess_threads = 4 #读取线程数
        batch_size = 100 #每次训练数据量
        min_queue_examples = 5000 #最小数据量
        image, label ,fileName = tf.train.shuffle_batch(
            [image, label, fileName],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
        print(label._shape)
        print(image._shape)
        return label,image,fileName

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
        filename_queue, label_queue = tf.train.slice_input_producer([TEST_IMAGE_PATHS,TEST_GENDER],shuffle=True,num_epochs=None) #文件列表
        #读取文件
        #reader = tf.WholeFileReader()
        #key, value = reader.read(filename_queue)
        value = tf.read_file(filename_queue)
        label = label_queue
        image = value
        image = tf.image.decode_jpeg(image, channels=3)#读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
        #image = tf.image.grayscale_to_rgb(image)#转彩色
        image = tf.image.resize_images(image,[227,227]) #缩放图片
        #随机
        num_preprocess_threads = 4 #读取线程数
        batch_size = 50 #每次训练数据量
        min_queue_examples = 5000 #最小数据量
        image, label = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
        print(label._shape)
        print(image._shape)
        return label,image

    def init(self):
        x = tf.placeholder("float", shape=[None, 227, 227, 3],name="x")#定义变量，输入值
        #x = image
        print("--------------------------------x-------------------")
        print(x._shape)
        y_ = tf.placeholder("float", shape=[None, 2],name="y_")#定义变量，输出值
        #y_ = label

        #第一层
        W_conv1 = self.weight_variable([7, 7, 3, 96])#初始化权重，5*5，输入数量1，输出数量32
        b_conv1 = self.bias_variable([96])#初始化偏置，输出数量32

        x_image = tf.reshape(x, [-1,227,227,3])#转成四维向量，大小227*227，颜色通道3，对应输入数量

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1,4,"VALID") + b_conv1)#定义卷积  (100, 56, 56, 96)
        print("-----h_conv1.shape:%s"%h_conv1._shape)
        h_pool1 = self.max_pool(h_conv1,3,2)#定义池化 (100, 28, 28, 96)
        print("-----h_pool1.shape:%s"%h_pool1._shape)
        norm1 = tf.nn.local_response_normalization(h_pool1, 5, alpha=0.0001, beta=0.75, name='norm1') #归一化
        print("-----norm1.shape:%s"%norm1._shape)

        #第二层
        W_conv2 = self.weight_variable([5, 5, 96, 256])
        b_conv2 = self.bias_variable([256])
        
        h_conv2 = tf.nn.relu(self.conv2d(norm1, W_conv2,1,"VALID") + b_conv2)# (100, 24, 24, 256)
        print("-----h_conv2.shape:%s"%h_conv2._shape)
        h_pool2 = self.max_pool(h_conv2,3,2)#定义池化 (100, 12, 12, 256)
        print("-----h_pool2.shape:%s"%h_pool2._shape)
        norm2 = tf.nn.local_response_normalization(h_pool2, 5, alpha=0.0001, beta=0.75, name='norm2') #归一化
        print("-----norm2.shape:%s"%norm2._shape)

        #第三层
        W_conv3 = self.weight_variable([3, 3, 256, 384])
        b_conv3 = self.bias_variable([384])

        h_conv3 = tf.nn.relu(self.conv2d(norm2, W_conv3,1,"SAME") + b_conv3)#(100, 12, 12, 384)
        print("-----h_conv3.shape:%s"%h_conv3._shape)
        h_pool3 = self.max_pool(h_conv3,3,2)#(100, 6, 6, 384)
        print("-----h_pool3.shape:%s"%h_pool3._shape)

        #全连接层
        W_fc1 = self.weight_variable([6 * 6 * 384, 512])#定义权重，输出数量1024
        b_fc1 = self.bias_variable([512])#定义偏置，输出数量1024

        h_pool3_flat = tf.reshape(h_pool3, [-1, 6 * 6 * 384])#池化结果7*7*64转一维数组(100, 2048)
        print("-----h_pool3_flat.shape:%s"%h_pool3_flat._shape) 
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)#(100, 512)
        print("-----h_fc1.shape:%s"%h_fc1._shape)

        #dropout
        keep_prob = tf.placeholder("float",name="keep_prob")#dropout概率
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        #输出层也是全连接层
        W_fc2 = self.weight_variable([512, 2])
        b_fc2 = self.bias_variable([2])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name="y_conv")#用softmax激活函数

        print("-----y_.shape:%s"%y_._shape)
        print("-----y_conv.shape:%s"%y_conv._shape)
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv),name="cross_entropy")#损失函数，交叉熵
        #train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy,name="train_step")#ADAM优化器来做梯度最速下降，自动调整里面的变量
        train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,name="train_step")
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))#比较结果
        resultY = tf.argmax(y_conv,1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name="accuracy")#求平均
        return

    def run(self,istrain):
        print("aaaa")
        tf.reset_default_graph()#重置图
        print("aaaa1")
        print("aaaa2")
        self.init()#初始化
        print("bbbb")
        with tf.Session() as sess:
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))#tf.InteractiveSession()#启动Session，与底层通信
            print("ccc")
            saver = tf.train.Saver()
            graph = sess.graph
            keep_prob = graph.get_tensor_by_name("keep_prob:0")#dropout概率
            cross_entropy = graph.get_tensor_by_name("cross_entropy:0")#损失函数，交叉熵
            train_step = graph.get_operation_by_name("train_step")#ADAM优化器来做梯度最速下降，自动调整里面的变量
            accuracy = graph.get_tensor_by_name("accuracy:0")#求平均
            y_conv = graph.get_tensor_by_name("y_conv:0")
            x = graph.get_tensor_by_name("x:0")
            y_ = graph.get_tensor_by_name("y_:0")
            print("ccc1")
            if istrain:
                print("ddd")
                #sess.run(tf.initialize_all_variables())#初始化变量
                tf.local_variables_initializer().run()
                tf.global_variables_initializer().run()#初始化变量
                print("eee")
                #tf.summary.histogram(cross_entropy.name + '/activations',
                                                        #cross_entropy)
                #tf.summary.scalar(cross_entropy.name + '/sparsity',cross_entropy)
                #merged_summary_op = tf.summary.merge_all()
                #summary_writer = tf.summary.FileWriter("logs", sess.graph)
                print("fff")
                label,image = self.readFiles2()#读取文件
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess,coord=coord)
                
                for i in range(5000):
                    #print("fff1")
                    image_data, label_data = sess.run([image, label])
                    if i % 100 == 0:
                        #print("fff2")
                        train_accuracy = accuracy.eval(feed_dict={x:image_data,y_:label_data,keep_prob: 1.0})#识别
                        print("train_accuracy:%s"%train_accuracy)
                        #train_cross_entropy = cross_entropy.eval(feed_dict={keep_prob: 1.0})#识别
                        #print("train_cross_entropy:%s"%train_cross_entropy)
                    #print("fff3")
                    train_step.run(feed_dict={x:image_data,y_:label_data,keep_prob: 0.5})#训练

                checkpoint_path = os.path.join(self.FLAGS.model_dir, 'model.ckpt')
                print("checkpoint_path: %s" %checkpoint_path)
                saver.save(sess, checkpoint_path)
            else:
                print("self.FLAGS.model_dir:%s"%self.FLAGS.model_dir)
                ckpt = tf.train.get_checkpoint_state(self.FLAGS.model_dir)
                print("ccc3")
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    print("ckpt.model_checkpoint_path:%s"%ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("ccc32")
                else:
                    print('No checkpoint file found')
                    return
                print("ccc4")
                file_dir= "/home/hillsun/dxh/images_test/" #_test/"#"D:\\images_test\\"#
                print("ccc5")
                i=1
                for root, dirs, files in os.walk(file_dir):
                    for file in files:
                        filename = os.path.splitext(file)[0]
                        sex=filename.split('_')[0]
                        y_value=[]
                        if sex==1:
                            y_value=[[1,0]]
                        else:
                            y_value=[[0,1]]
                        filePath = file_dir + file
                        fin = open(filePath,'rb')
                        print("ccc6")
                        img = self.load_and_align_data(fin)
                        #train_step.run(feed_dict={x: img, y_: y_value, keep_prob: 0.5})
                        print("i:%s"%i)
                        print("filename:%s"%filename)
                        #img_data = tf.reshape(img, [1,227,227,3])
                        #print(img_data.shape)
                        test_image = tf.convert_to_tensor(img, dtype=tf.float32)
                        img_data = tf.reshape(test_image, [1,227,227,3])
                        img_data=img_data.eval()
                        print(sess.run(y_conv,feed_dict={x: img_data,keep_prob: 1.0}))
                        i=i+1
                        #print("test accuracy %g" % accuracy.eval(feed_dict={x: img, y_: y_value, keep_prob: 1.0}))#识别
                        #print("test accuracy %g" % accuracy.eval(feed_dict={x: img, y_: y_value, keep_prob: 1.0}))#识别
        return

    def runYzr(self,isPileupTrain,img_dir):
        print("aaaa")
        tf.reset_default_graph()#重置图
        print("aaaa1")
        print("aaaa2")
        self.init()#初始化
        print("bbbb")
        with tf.Session() as sess:
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))#tf.InteractiveSession()#启动Session，与底层通信
            print("ccc")
            saver = tf.train.Saver()
            graph = sess.graph
            keep_prob = graph.get_tensor_by_name("keep_prob:0")#dropout概率
            cross_entropy = graph.get_tensor_by_name("cross_entropy:0")#损失函数，交叉熵
            train_step = graph.get_operation_by_name("train_step")#ADAM优化器来做梯度最速下降，自动调整里面的变量
            accuracy = graph.get_tensor_by_name("accuracy:0")#求平均
            y_conv = graph.get_tensor_by_name("y_conv:0")
            x = graph.get_tensor_by_name("x:0")
            y_ = graph.get_tensor_by_name("y_:0")

            print("ccc1")
            if isPileupTrain:
                ckpt = tf.train.get_checkpoint_state(self.FLAGS.model_dir)
                print("ccc3")
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    print("ckpt.model_checkpoint_path:%s"%ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("ccc32")
                else:
                    print('No checkpoint file found')
                    return
                print("eee")
                label,image,fileName = self.readFiles3(img_dir)#读取文件
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess,coord=coord)
                
                for i in range(147):
                    image_data, label_data,file_data = sess.run([image, label,fileName])
                    if i % 100 == 0:
                        train_accuracy = accuracy.eval(feed_dict={x:image_data,y_:label_data,keep_prob: 0.5})
                        print("train_accuracy:%s"%train_accuracy)
                    train_step.run(feed_dict={x:image_data,y_:label_data,keep_prob: 0.5})#训练

                checkpoint_path = os.path.join(self.FLAGS.model_dir, 'model.ckpt')
                print("checkpoint_path: %s" %checkpoint_path)
                saver.save(sess, checkpoint_path)
            else:
                tf.local_variables_initializer().run()
                tf.global_variables_initializer().run()#初始化变量
                print("eee")
                label,image,fileName = self.readFiles3(img_dir)#读取文件
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess,coord=coord)
                
                for i in range(301):
                    image_data, label_data,file_data = sess.run([image, label,fileName])
                    if i % 100 == 0:
                        train_accuracy = accuracy.eval(feed_dict={x:image_data,y_:label_data,keep_prob: 0.5})
                        print("train_accuracy:%s"%train_accuracy)
                    train_step.run(feed_dict={x:image_data,y_:label_data,keep_prob: 0.5})#训练

                checkpoint_path = os.path.join(self.FLAGS.model_dir, 'model.ckpt')
                print("checkpoint_path: %s" %checkpoint_path)
                saver.save(sess, checkpoint_path) 
        return

    def idlen_gender(self,img_dir):
        print("aaaa")
        tf.reset_default_graph()#重置图
        print("aaaa1")
        print("aaaa2")
        self.init()#初始化
        print("bbbb")
        with tf.Session() as sess:
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))#tf.InteractiveSession()#启动Session，与底层通信
            print("ccc")
            saver = tf.train.Saver()
            graph = sess.graph
            train_step = graph.get_operation_by_name("train_step")#ADAM优化器来做梯度最速下降，自动调整里面的变量
            accuracy = graph.get_tensor_by_name("accuracy:0")#求平均
            x = graph.get_tensor_by_name("x:0")
            y_ = graph.get_tensor_by_name("y_:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")#dropout概率
        
            ckpt = tf.train.get_checkpoint_state(self.FLAGS.model_dir)
            print("ccc3")
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                print("ckpt.model_checkpoint_path:%s"%ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("ccc32")
            else:
                print('No checkpoint file found')
                return
            print("eee")
            label,image,fileName = self.readFiles3(img_dir)#读取文件
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            image_data, label_data,file_data = sess.run([image, label,fileName])
            print("label_data.len:%d"%len(label_data))
            #print(label_data)
            train_accuracy = accuracy.eval(feed_dict={x:image_data,y_:label_data,keep_prob: 1.0})#识别
            print("train_accuracy:%s"%train_accuracy)             
        return

def parse_arguments(self,argv):
    #向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项；
    #最后调用parse_args()方法进行解析；解析成功之后即可使用
    parser = argparse.ArgumentParser()
    #model为参数别名，后面执行args = parser.parse_args()后，可通过args.model调用。
    #nargs='+'：当参数有多余的情况下，多余的参数都会属于当前这个参数
    parser.add_argument('img_dir', type=str, nargs='+',
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    # parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    # parser.add_argument('--image_size', type=int,
    #     help='Image size (height, width) in pixels.', default=160)
    # parser.add_argument('--margin', type=int,
    #     help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    # parser.add_argument('--gpu_memory_fraction', type=float,
    #     help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

def main(argv):
    if argv == None:
        print('please input image path!')
    else:
        print(argv[1])
    obj=picture_train()
    obj.idlen_gender(argv[1])
    #obj.runYzr(True,argv[1])
    #obj.run(True)
    
if __name__ == '__main__':
    main(sys.argv)