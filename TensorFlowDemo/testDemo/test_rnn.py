import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28
n_steps = 28
# 隐藏层神经元数量
n_hidden_unis = 128
n_classes = 10

# (-1,28,28)
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# (-1,10)
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    # (28,128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),
    # (128,10)
    'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))
}

biases = {
    # (128)
    'in': tf.Variable(tf.constant(0.1, tf.float32, [n_hidden_unis])),
    # (10)
    'out': tf.Variable(tf.constant(0.1, tf.float32, [n_classes]))
}


def RNN(X, weights, biases):
    # 输入判断
    # (-1*28,28)
    X = tf.reshape(X, [-1, n_inputs])
    print(X.shape)
    # (-1*28,128)
    X_in = tf.matmul(X, weights['in'])+biases['in']
    print(X_in.shape)
    # (-1,28,128)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_unis])
    print(X_in.shape)

    # LSTMCell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
        n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, tf.float32)

    # 输出判断
    # (28,128,28),time_major:x_in数量维度放在0(False)还是1(True)
    outputs, states = tf.nn.dynamic_rnn(
        lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # results=tf.matmul(states[1],weights['out'])+biases['out']
    # unstack拆分维度(128,28*28),transpose转维度(28,128,28)转(128,28,28)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out'])+biases['out']

    # results = None
    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        # (128,784) (128,10)
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # (128,28*28)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys
            }))
        step += 1
