import tensorflow as tf

x = tf.Variable(0)

y = tf.constant(1)

new_value = tf.add(x, y)
update = tf.assign(x, new_value)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in range(3):
        print(step, sess.run(update))
