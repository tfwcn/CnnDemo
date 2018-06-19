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


def readFiles1():
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
    batch_size = 1  # 每次训练数据量
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

def init4():
    """初始化神经网络"""
    x = tf.placeholder(
        "float", shape=[None, 160, 160, 3], name="x")  # 定义变量，输入值
    y_ = tf.placeholder("float", shape=[None, 2], name="y_")  # 定义变量，输出值
    keep_prob = tf.placeholder("float", name="keep_prob")  # dropout概率

    logits, endpoints = inception.inception_resnet_v2(x, 2)
    # logits, endpoints = inception.inception_v4(x, 2)

    print(logits.name)
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) +
    #                        b_fc2, name="y_conv")  # 用softmax激活函数
    # print(y_.shape)
    # print(y_conv.shape)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(features = y_, logits = logits, name="cross_entropy")
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

def run(istrain):
    """训练"""
    tf.reset_default_graph()  # 重置图
    label, image = readFiles2()  # 读取文件
    init4()  # 初始化
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)) as sess:  # 启动Session，与底层通信
        graph = sess.graph
        # 加载变量和操作
        x = graph.get_tensor_by_name("x:0")  # 定义变量，输入值
        y_ = graph.get_tensor_by_name("y_:0")  # 定义变量，输出值
        keep_prob = graph.get_tensor_by_name("keep_prob:0")  # dropout概率
        cross_entropy = graph.get_tensor_by_name("cross_entropy:0")  # 损失函数，交叉熵
        train_step = graph.get_operation_by_name(
            "train_step")  # ADAM优化器来做梯度最速下降，自动调整里面的变量
        accuracy = graph.get_tensor_by_name("accuracy:0")  # 求平均
        saver = tf.train.Saver()
        if istrain:
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
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(
                        feed_dict={x: image_value, y_: label_value, keep_prob: 1.0})  # 识别
                    print("step %d, training accuracy %g" %
                          (i, train_accuracy))
                    print('训练数：%d %s' %
                          (i, cross_entropy.eval(feed_dict={x: image_value, y_: label_value, keep_prob: 1})))
                if i % 1000 == 0:
                    checkpoint_path = os.path.join('.', 'model.ckpt')
                    saver.save(sess, checkpoint_path)
                #summary_str = sess.run(merged_summary_op,feed_dict={keep_prob: 1.0})
                #summary_writer.add_summary(summary_str, i)
                train_step.run(
                    feed_dict={x: image_value, y_: label_value, keep_prob: 0.8})  # 训练
            # 停止填充队列
            coord.request_stop()
            coord.join(threads)
            # Create a saver.
            #saver = tf.train.Saver(tf.all_variables())
            checkpoint_path = os.path.join('.', 'model.ckpt')
            saver.save(sess, checkpoint_path)
        else:
            ckpt = tf.train.get_checkpoint_state('.')
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            print("test accuracy %g" % accuracy.eval(feed_dict={
                x: mnist.test.images[:100], y_: mnist.test.labels[:100], keep_prob: 1.0}))  # 识别
    return


def train():
    """训练"""
    tf.reset_default_graph()  # 重置图
    label, image = readFiles4()  # 读取文件
    readFiles3()
    init4()  # 初始化
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
        # y_conv = graph.get_tensor_by_name("y_conv:0")  # 定义变量，输出值
        y_conv = graph.get_tensor_by_name("InceptionResnetV2/Logits/Logits/BiasAdd:0")  # 定义变量，输出值
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
    init4()  # 初始化
    readFiles3()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)) as sess:  # 启动Session，与底层通信
        graph = sess.graph
        # 加载变量和操作
        x = graph.get_tensor_by_name("x:0")  # 定义变量，输入值
        y_ = graph.get_tensor_by_name("y_:0")  # 定义变量，输入值
        # y_conv = graph.get_tensor_by_name("y_conv:0")  # 定义变量，输出值
        y_conv = graph.get_tensor_by_name("InceptionResnetV2/Logits/Logits/BiasAdd:0")  # 定义变量，输出值
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
