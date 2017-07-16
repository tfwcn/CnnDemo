import input_data
import tensorflow as tf
import os

def weight_variable(shape):
    """初始化权重"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """初始化偏置"""
    initial = tf.constant(0.1, shape=shape)
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
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)#加载图片
    x = tf.placeholder("float", shape=[None, 784])#定义变量，输入值
    y_ = tf.placeholder("float", shape=[None, 10])#定义变量，输出值
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
    keep_prob = tf.placeholder("float")#dropout概率
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #输出层也是全连接层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)#用softmax激活函数

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))#损失函数，交叉熵
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#ADAM优化器来做梯度最速下降
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))#比较结果
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#求平均
    sess = tf.InteractiveSession()#启动Session，与底层通信
    if istrain:
        sess.run(tf.initialize_all_variables())#初始化变量
        for i in range(100):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})#识别
                print("step %d, training accuracy %g" % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})#训练

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        checkpoint_path = os.path.join('/tmp/cifar10_train', 'model.ckpt')
        saver.save(sess, checkpoint_path)
    else:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('/tmp/cifar10_train')
        if ckpt and ckpt.model_checkpoint_path:
          # Restores from checkpoint
          saver.restore(sess, ckpt.model_checkpoint_path)
        else:
          print('No checkpoint file found')
          return
        print("test accuracy %g" % sess.run(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))#识别
    return
        
def train():
    """训练"""
    run(True)
    return
        
def predict():
    """识别"""
    run(False)
    return

def main(argv=None):  #运行
    train()
    predict()
    return


if __name__ == '__main__':
    tf.app.run()