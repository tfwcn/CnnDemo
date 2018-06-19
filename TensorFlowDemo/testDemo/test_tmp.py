import tensorflow as tf
import numpy as np

y=np.array([[2.0,4.0,6.0],[2.0,4.0,8.0]])
y_train=np.array([[1.0,2.0,3.0],[1.0,2.0,3.0]])
s=tf.square(y-y_train)
# loss=tf.reduce_mean(s)
loss = tf.losses.softmax_cross_entropy(onehot_labels = y, logits = y_train)
cross_entropy = tf.losses.get_total_loss(name="cross_entropy")    #obtain the regularization losses as well
cross_entropy2 = tf.multiply(tf.reduce_sum(
    tf.nn.softmax(y * tf.log(y_train))), -1, name="cross_entropy")  # 损失函数，交叉熵
sess=tf.Session()
print(sess.run(s))
print(sess.run(loss))
print(sess.run(cross_entropy))
print(sess.run(cross_entropy2))