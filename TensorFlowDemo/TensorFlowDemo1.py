import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)#加载图片
sess = tf.InteractiveSession()#启动Session，与底层通信
x = tf.placeholder("float", shape=[None, 784])#定义变量，输入值
y_ = tf.placeholder("float", shape=[None, 10])#定义变量，输出值
W = tf.Variable(tf.zeros([784,10]))#权重
b = tf.Variable(tf.zeros([10]))#偏置
sess.run(tf.initialize_all_variables())#初始化全部变量
y = tf.nn.softmax(tf.matmul(x,W) + b)#前向传播公式
cross_entropy = -tf.reduce_sum(y_*tf.log(y))#损失函数，交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#训练步骤，梯度下降法，步长0.01，,最小偏差值为交叉熵
for i in range(1000):#循环1000次
    batch = mnist.train.next_batch(50)#每次50个集合
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})#运行一步
    print(i)#输出次数

#验证
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))#比较结果是否相同
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#把bool值转float，求结果平均
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))#输出结果

