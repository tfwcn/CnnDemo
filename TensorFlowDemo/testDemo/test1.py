import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float)
y_data = x_data*0.1+3

''' 创建神经网络结构 '''
w = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))

y = w * x_data + b

loss = tf.reduce_mean(tf.square(y-y_data))
# 梯度下降法
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
''' 结束创建神经网络结构 '''

with tf.Session() as sess:
    sess.run(init)

    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step,sess.run(w),sess.run(b),sess.run(loss))
