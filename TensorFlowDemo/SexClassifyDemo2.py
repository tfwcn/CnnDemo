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

#mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)#加载图片

def weight_variable(shape,name):
    """初始化权重"""
    initial = tf.truncated_normal(shape, stddev=0.1)#0.1的正态分布
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    """初始化偏置"""
    initial = tf.constant(0.1, shape=shape)#所有值0.1
    return tf.Variable(initial,name=name)

def conv2d(x, W):
    """卷积，步长为1，,padding为0"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x,name):
    """池化，2*2，步长为2，,padding为0"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME',name=name)

def dense_to_one_hot(labels_dense, num_classes=2):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = len(labels_dense) #数组长度
  index_offset = numpy.arange(num_labels) * num_classes #根据数组生成序号
  labels_one_hot = numpy.zeros([num_labels, num_classes]) #初始化二维数组
  for i in range(num_labels):#赋值
    if math.isnan(labels_dense[i]):
        continue
    labels_one_hot.flat[int(index_offset[i] + labels_dense[i])] = 1 #一维化赋值
  return labels_one_hot.tolist()

def readFiles():
    #读取地址和性别
    dataset=sio.loadmat(FLAGS.data_dir+"/imdb.mat")
    filepath=dataset["imdb"][0,0]["full_path"][0]
    gender=dataset["imdb"][0,0]["gender"][0]
    #整理
    TEST_IMAGE_PATHS = [ os.path.join(FLAGS.data_dir, filepath[i][0]) for i in range(0, len(gender)) ]
    TEST_GENDER = [ gender[i] for i in range(0, len(gender)) ]
    TEST_GENDER = dense_to_one_hot(TEST_GENDER,2) #转化成1维数组
    # slice_input_producer会产生一个文件名队列
    #filename_queue = tf.train.string_input_producer(TEST_IMAGE_PATHS) #文件列表
    filename_queue, label_queue = tf.train.slice_input_producer([TEST_IMAGE_PATHS,TEST_GENDER],shuffle=True,num_epochs=3) #文件列表
    #读取文件
    #reader = tf.WholeFileReader()
    #key, value = reader.read(filename_queue)
    value = tf.read_file(filename_queue)
    label = label_queue
    image = value
    image = tf.image.decode_jpeg(image, channels=3)#读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    #image = tf.image.grayscale_to_rgb(image)#转彩色
    image = tf.image.resize_images(image,[160,160]) #缩放图片
    #随机
    num_preprocess_threads = 4 #读取线程数
    batch_size = 20 #每次训练数据量
    min_queue_examples = 10000 #最小数据量
    image, label = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    print(label._shape)
    print(image._shape)
    return label,image

def init(label,image):
    """初始化神经网络"""
    #tf.reset_default_graph()#重置图
    #x = tf.placeholder("float", shape=[None,960,540,3],name="x")#定义变量，输入值
    #y_ = tf.placeholder("float", shape=[None, 2],name="y_")#定义变量，输出值
    x = image
    y_ = label
    #第一层960*540*32
    #x_image = tf.reshape(x,
    #[-1,960,540,3],name="x_image")#转成四维向量，大小960*540，颜色通道3，对应输入数量

    h_conv1 = slim.conv2d(x, 32, [5, 5], scope='Conv2d_1_5x5',
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.1))
    h_pool1 = slim.max_pool2d(h_conv1, [2, 2], scope='MaxPool_1_2x2', stride=2)

    #第二层480*270*64
    h_conv2 = slim.conv2d(h_pool1, 64, [5, 5], scope='Conv2d_2_5x5',
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.1))
    h_pool2 = slim.max_pool2d(h_conv2, [2, 2], scope='MaxPool_2_2x2', stride=2)
    
    #全连接层20*20*512
    W_fc1 = weight_variable([20 * 20 * 320, 512],name="W_fc1")#定义权重，输出数量1024
    b_fc1 = bias_variable([512],name="b_fc1")#定义偏置，输出数量1024

    h_pool2_flat = tf.reshape(mixed_4a, [-1, 20 * 20 * 320],name="h_pool2_flat")#池化结果7*7*64转一维数组
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,name="h_fc1")#用relu激活函数

    #dropout
    keep_prob = tf.placeholder("float",name="keep_prob")#dropout概率
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name="h_fc1_drop")

    #输出层也是全连接层
    W_fc2 = weight_variable([512, 2],name="W_fc2")
    b_fc2 = bias_variable([2],name="b_fc2")

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name="y_conv")#用softmax激活函数

    cross_entropy = tf.multiply(tf.reduce_sum(y_ * tf.log(y_conv)),-1,name="cross_entropy")#损失函数，交叉熵
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,name="train_step")#ADAM优化器来做梯度最速下降，自动调整里面的变量
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,name="train_step")#ADAM优化器来做梯度最速下降，自动调整里面的变量
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1),name="correct_prediction")#比较结果
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name="accuracy")#求平均
    return

def run(istrain):
    """训练"""
    tf.reset_default_graph()#重置图
    label,image = readFiles()#读取文件
    init(label,image)#初始化
    sess = tf.InteractiveSession()#启动Session，与底层通信
    graph = sess.graph
    #加载变量和操作
    #x = graph.get_tensor_by_name("x:0")#定义变量，输入值
    #y_ = graph.get_tensor_by_name("y_:0")#定义变量，输出值
    keep_prob = graph.get_tensor_by_name("keep_prob:0")#dropout概率
    cross_entropy = graph.get_tensor_by_name("cross_entropy:0")#损失函数，交叉熵
    train_step = graph.get_operation_by_name("train_step")#ADAM优化器来做梯度最速下降，自动调整里面的变量
    accuracy = graph.get_tensor_by_name("accuracy:0")#求平均
    saver = tf.train.Saver()
    if istrain:
        #sess.run(tf.initialize_all_variables())#初始化变量
        tf.initialize_all_variables().run()#初始化变量
        tf.initialize_local_variables().run()#初始化变量
        #加载已训练数据
        ckpt = tf.train.get_checkpoint_state(FLAGS.data_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        #记录训练数据
        #tf.summary.histogram(cross_entropy.name + '/activations',
                                                   #cross_entropy)
        tf.summary.scalar(cross_entropy.name + '/sparsity',cross_entropy)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.data_dir, graph)
        # 使用start_queue_runners之后，才会开始填充队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        #训练
        for i in range(10000):
            #batch = photo,label_value = sess.run([image,label])
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={keep_prob: 1.0})#识别
                print("step %d, training accuracy %g" % (i, train_accuracy))
                checkpoint_path = os.path.join(FLAGS.data_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path)
            summary_str = sess.run(merged_summary_op,feed_dict={keep_prob: 1.0})
            summary_writer.add_summary(summary_str, i)
            train_step.run(feed_dict={keep_prob: 1})#训练
            print(i)
        #停止填充队列
        coord.request_stop()
        coord.join(threads)
        # Create a saver.
        #saver = tf.train.Saver(tf.all_variables())
        checkpoint_path = os.path.join(FLAGS.data_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path)
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.data_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
       
        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images[:100], y_: mnist.test.labels[:100], keep_prob: 1.0}))#识别
    return
        
def train():
    """训练"""
    run(True)
    return
        
def predict():
    """识别"""
    run(False)
    return

def saveTest():
    """保存变量测试"""
    tf.reset_default_graph()#重置图
    # Create some variables.
    v1 = tf.Variable(tf.constant(-1,shape=[2,3]), name="v1")
    v2 = tf.Variable(tf.constant(1,shape=[2,3]), name="v2")
    #...
    # Add an op to initialize the variables.
    init_op = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, save the
    # variables to disk.
    with tf.Session() as sess:
      sess.run(init_op)
      # Do some work with the model.
      #..
      # Save the variables to disk.
      save_path = saver.save(sess,  FLAGS.data_dir + '/model.ckpt')
      print("Model saved in file: ", save_path)

def loadTest():
    """读取变量测试"""
    tf.reset_default_graph()#重置图
    # Create some variables.
    v1 = tf.Variable(tf.constant(0,shape=[2,3]), name="v1")
    v2 = tf.Variable(tf.constant(0,shape=[2,3]), name="v2")
    #...

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

def main(argv=None):  #运行
    train()
    #predict()
    #saveTest()
    #loadTest()
if __name__ == '__main__':
    tf.app.run()