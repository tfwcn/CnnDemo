from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
from keras import backend, optimizers, losses, metrics
import tensorflow as tf
import os
import numpy
import math


def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = len(labels_dense)  # 数组长度
    index_offset = numpy.arange(num_labels) * num_classes  # 根据数组生成序号
    labels_one_hot = numpy.zeros([num_labels, num_classes])  # 初始化二维数组
    for i in range(num_labels):  # 赋值
        if math.isnan(labels_dense[i]):
            continue
        labels_one_hot.flat[int(
            index_offset[i] + labels_dense[i])] = 1  # 一维化赋值
    return labels_one_hot.tolist()


def readFiles(file_dir):
    # 读取文件列表
    # file_dir="/home/hillsun/dxh/images_hyj_7243/man_4314/"
    # file_dir="/home/hillsun/dxh/images_yzr_avg/"
    # pathDir = os.listdir(file_dir)[0:2900]
    pathDir = os.listdir(file_dir)
    # 定义特征与标签
    TEST_IMAGE_PATHS = [file_dir+path for path in pathDir]
    TEST_GENDER = [int(path.split('_')[1])-1 for path in pathDir]
    # # 读取文件列表
    # file_dir="/home/hillsun/dxh/images_hyj_7243/women_2929/"
    # pathDir = os.listdir(file_dir)[0:2900]
    # # 定义特征与标签
    # TEST_IMAGE_PATHS += [file_dir+path for path in pathDir]
    # TEST_GENDER += [int(path.split('_')[1])-1 for path in pathDir]
    TEST_GENDER = dense_to_one_hot(TEST_GENDER, 2)  # 转化成1维数组

    # print(TEST_IMAGE_PATHS)
    # print(TEST_GENDER)

    # slice_input_producer会产生一个文件名队列
    # filename_queue = tf.train.string_input_producer(TEST_IMAGE_PATHS) #文件列表
    filename_queue, label_queue = tf.train.slice_input_producer(
        [TEST_IMAGE_PATHS, TEST_GENDER], shuffle=True, num_epochs=None)  # 文件列表
    # 读取文件
    #reader = tf.WholeFileReader()
    #key, value = reader.read(filename_queue)
    value = tf.read_file(filename_queue)
    label = label_queue
    image = value
    # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.grayscale_to_rgb(image)#转彩色
    image = tf.image.resize_images(image, [160, 160])  # 缩放图片
    # 随机
    # num_preprocess_threads = 4  # 读取线程数
    # batch_size = 50  # 每次训练数据量
    # min_queue_examples = 5000  # 最小数据量
    # image, label = tf.train.shuffle_batch(
    #     [image, label],
    #     batch_size=batch_size,
    #     num_threads=num_preprocess_threads,
    #     capacity=min_queue_examples + 3 * batch_size,
    #     min_after_dequeue=min_queue_examples)
    # print(label._shape)
    # print(image._shape)
    image = tf.divide(image, 255.0)
    return label, image


def readFiles2():
    """识别用"""
    filename = tf.placeholder(tf.string, name="filename")  # 定义变量，输入值
    # 读取文件
    image = tf.read_file(filename)
    # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.grayscale_to_rgb(image)#转彩色
    image = tf.image.resize_images(image, [160, 160])  # 缩放图片
    image = tf.reshape(image, [-1, 160, 160, 3], name="input_image")
    image = tf.divide(image, 255.0)
    return filename, image


def train():
    """训练"""
    file_dir = "/home/hillsun/dxh/images_yzr_avg/"
    label, image = readFiles(file_dir)  # 读取文件
    file_dir2 = "/home/hillsun/dxh/images_align_240/"
    label2, image2 = readFiles(file_dir2)  # 读取文件
    keep_prob = 0.6  # dropout概率
    model = Sequential()
    model.add(Conv2D(96, 7, 7, (4, 4)))
    model.add(Activation(backend.relu))
    model.add(MaxPool2D((3, 3), (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(keep_prob))

    model.add(Conv2D(80, 5, 5))
    model.add(Activation(backend.relu))
    model.add(MaxPool2D((3, 3), (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(keep_prob))

    model.add(Conv2D(120, 5, 5))
    model.add(Activation(backend.relu))
    model.add(MaxPool2D((3, 3), (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(keep_prob))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation(backend.relu))
    model.add(Dropout(keep_prob))

    model.add(Dense(2))
    model.add(Activation(backend.softmax))

    # 编译模型
    model.compile(optimizers.Adadelta,
                  losses.categorical_crossentropy, metrics.binary_accuracy)
    # 训练
    model.fit(image, label, batch_size=50, epochs=1000,
              verbose=1, validation_data=(image2, label2))
    # 识别
    score = model.evaluate(image2, label2, verbose=1)

    print("损失值：", scorep[0])
    print("准确率：", scorep[1])

    return


def main(argv=None):  # 运行
    train()
    # predict()


if __name__ == '__main__':
    tf.app.run()
