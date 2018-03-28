import tensorflow as tf


class ModelHelper():
    def __init__(self):
        # 初始化
        # self.x = 0
        pass

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape):
        # shape[a,b,c,d] a*b卷积核，c输入维度，d输出维度
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def hot_one(self, data_list, data_dir, repeat_num=1):
        out_data = []
        for list_index in range(0, len(data_list)):
            out_data += [[(1 if data_list[list_index][dir_index // len(data_dir)] == data_dir[dir_index % len(data_dir)] else 0)
                          for dir_index in range(0, len(data_dir) * repeat_num)]]
        return out_data
