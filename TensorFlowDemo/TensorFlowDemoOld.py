import input_data
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data/MNIST_data',
                           """Path to the MNIST data directory.""")

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)#加载图片
def weight_variable(shape):
    """初始化权重"""
    initial = tf.truncated_normal(shape, stddev=0.1)#0.1的正态分布
    return tf.Variable(initial)

def bias_variable(shape):
    """初始化偏置"""
    initial = tf.constant(0.1, shape=shape)#所有值0.1
    return tf.Variable(initial)

def conv2d(x, W):
    """卷积，步长为1，,padding为0"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """池化，2*2，步长为2，,padding为0"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def run(istrain):
    """训练"""
    tf.reset_default_graph()#重置图
    x = tf.placeholder("float", shape=[None, 784],name="x")#定义变量，输入值
    y_ = tf.placeholder("float", shape=[None, 10],name="y_")#定义变量，输出值
    #第一层
    W_conv1 = weight_variable([5, 5, 1, 32])#初始化权重，5*5，输入数量1，输出数量32
    b_conv1 = bias_variable([32])#初始化偏置，输出数量32

    x_image = tf.reshape(x, [-1,28,28,1])#转成四维向量，大小28*28，颜色通道1，对应输入数量

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#定义卷积
    h_pool1 = max_pool_2x2(h_conv1)#定义池化

    #第二层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#用relu激活函数
    h_pool2 = max_pool_2x2(h_conv2)

    #全连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])#定义权重，输出数量1024
    b_fc1 = bias_variable([1024])#定义偏置，输出数量1024

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])#池化结果7*7*64转一维数组
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)#用relu激活函数

    #dropout
    keep_prob = tf.placeholder("float",name="keep_prob")#dropout概率
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #输出层也是全连接层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)#用softmax激活函数

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))#损失函数，交叉熵
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#ADAM优化器来做梯度最速下降，自动调整里面的变量
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))#比较结果
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name="accuracy")#求平均
    sess = tf.InteractiveSession()#启动Session，与底层通信
    saver = tf.train.Saver()
    if istrain:
        #sess.run(tf.initialize_all_variables())#初始化变量
        tf.initialize_all_variables().run()#初始化变量
        #tf.summary.histogram(cross_entropy.name + '/activations',
                                                   #cross_entropy)
        tf.summary.scalar(cross_entropy.name + '/sparsity',cross_entropy)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.data_dir, sess.graph)
        for i in range(10000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})#识别
                print("step %d, training accuracy %g" % (i, train_accuracy))
            summary_str = sess.run(merged_summary_op,feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            summary_writer.add_summary(summary_str, i)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})#训练

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
    predict()
    #saveTest()
    #loadTest()

if __name__ == '__main__':
    tf.app.run()