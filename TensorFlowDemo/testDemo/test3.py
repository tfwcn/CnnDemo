import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = "layer%s" % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            w = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layer_name+"/weights", w)
        with tf.name_scope("biases"):
            b = tf.Variable(tf.zeros([1, out_size]))
            tf.summary.histogram(layer_name+"/biases", b)

        with tf.name_scope("wx_plus_b"):
            wx_plus_b = tf.matmul(inputs, w)+b
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data)-0.5 + noise
print(x_data.shape, y_data.shape)

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
l1 = add_layer(xs, 1, 10, 1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, 2, activation_function=None)
print(l1.shape, prediction.shape)

with tf.name_scope("loss"):
    # reduction_indices处理维度，这里1代表第二维度
    loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(ys-prediction), reduction_indices=[1]))
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    summary_merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("./log/", sess.graph)
    sess.run(init)

    fig = plt.figure()  # 创建窗口
    ax = fig.add_subplot(1, 1, 1)  # 添加视图
    ax.scatter(x_data, y_data)  # 添加散点图片
    plt.ion()
    plt.show()

    for step in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if step % 50 == 0:
            print(step, sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            summary_writer.add_summary(
                sess.run(summary_merged, feed_dict={xs: x_data, ys: y_data}), step)
            if len(ax.lines) > 0:
                ax.lines.remove(ax.lines[0])
            ax.plot(x_data, prediction_value, "r-", LineWidth=5)  # 线宽5
            plt.pause(0.1)
    plt.pause(5)
