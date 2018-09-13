from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras import backend, optimizers, losses, metrics
import tensorflow as tf
import os
import numpy
import math
import h5py

# face_path = "/media/ppht/000872BB00040832"
face_path = "E:"

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


def readFilesBatch(file_dir, batch_size):
    # 读取文件列表
    TEST_IMAGE_PATHS = []
    TEST_GENDER = []
    for i in range(len(file_dir)):
        pathDir = os.listdir(file_dir[i])
        # 定义特征与标签
        TEST_IMAGE_PATHS += [file_dir[i]+path for path in pathDir]
        TEST_GENDER += [int(path.split('_')[1])-1 for path in pathDir]
    TEST_GENDER = dense_to_one_hot(TEST_GENDER, 2)  # 转化成1维数组
    # slice_input_producer会产生一个文件名队列
    filename_queue, label_queue = tf.train.slice_input_producer(
        [TEST_IMAGE_PATHS, TEST_GENDER], shuffle=True, num_epochs=None)  # 文件列表
    # 读取文件
    value = tf.read_file(filename_queue)
    label = label_queue
    image = value
    # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.grayscale_to_rgb(image)#转彩色
    image = tf.image.resize_images(image, [160, 160])  # 缩放图片
    # 分批随机
    num_preprocess_threads = 4  # 读取线程数
    # batch_size = 50  # 每次训练数据量
    min_queue_examples = 5000  # 最小数据量
    image, label = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    image = tf.divide(image, 255.0)
    print("readFilesBatch label:", label.shape)
    print("readFilesBatch image:", image.shape)
    return label, image


def readFilesOne():
    """读取单个文件，识别用"""
    filename = tf.placeholder(tf.string)  # 定义变量，输入值
    # 读取文件
    image = tf.read_file(filename)
    # 读取图片，含彩色与灰度。彩色一定要decode_jpeg方法
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.grayscale_to_rgb(image)#转彩色
    image = tf.image.resize_images(image, [160, 160])  # 缩放图片
    image = tf.reshape(image, [-1, 160, 160, 3])
    image = tf.divide(image, 255.0)
    return filename, image

def createModel():
    keep_prob = 0.6  # dropout概率
    model = Sequential()
    model.add(Conv2D(30, (7, 7), strides=(
        4, 4), input_shape=(160, 160, 3), name="cnn1"))
    model.add(Activation(backend.relu))
    # model.add(MaxPool2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(30, (3, 3), strides=(2, 2), name="cnn2"))
    model.add(BatchNormalization())
    model.add(Dropout(keep_prob))

    model.add(Conv2D(80, (5, 5), name="cnn3"))
    model.add(Activation(backend.relu))
    # model.add(MaxPool2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(80, (3, 3), strides=(2, 2), name="cnn4"))
    model.add(BatchNormalization())
    model.add(Dropout(keep_prob))

    model.add(Conv2D(120, (5, 5), name="cnn5"))
    model.add(Activation(backend.relu))
    # model.add(MaxPool2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(120, (3, 3), strides=(2, 2), name="cnn6"))
    model.add(BatchNormalization())
    model.add(Dropout(keep_prob))

    model.add(Flatten())
    model.add(Dense(100, name="dnn1"))
    model.add(Activation(backend.relu))
    model.add(Dropout(keep_prob))

    model.add(Dense(2, name="dnn2"))
    model.add(Activation(backend.softmax))

    # 编译模型
    model.compile(optimizers.Adadelta(),
                    losses.categorical_crossentropy, [metrics.binary_accuracy])
    
    return model

def train():
    """训练"""
    ###############
    # 开始建立CNN模型
    ###############
    # 加载模型
    model = createModel()
    if os.path.isfile("my_model.h5"):
        model = tf.keras.models.load_model("my_model.h5")

    file_dir = [face_path + "/face/face0/女_2929/",
                face_path + "/face/face0/男_2929/"]
    label, image = readFilesBatch(file_dir, 50)  # 读取文件

    file_dir2 = [face_path + "/face/images_test_500/man/",
                 face_path + "/face/images_test_500/women/"]
    label2, image2 = readFilesBatch(file_dir2, 250)  # 读取文件

    print("训练")
    # 训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 使用start_queue_runners之后，才会开始填充队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        image_value, label_value = image, label
        image2_value, label2_value = image2, label2
        dataset = tf.data.Dataset.from_tensors((image_value, label_value))
        dataset = dataset.repeat()

        val_dataset = tf.data.Dataset.from_tensors(
            (image2_value, label2_value))
        val_dataset = val_dataset.repeat()
        # 训练 steps_per_epoch(每次训练图片批次数) epochs(循环次数) validation_steps(测试图片数)
        model.fit(dataset, steps_per_epoch=100, validation_steps=2, epochs=10,
                  validation_data=val_dataset)
        # # 识别
        # score = model.evaluate_generator(datagen.flow(image, label, batch_size=32))
        score = model.evaluate(val_dataset, steps=2)
        print("损失值：", score[0])
        print("准确率：", score[1])
        # 停止填充队列
        coord.request_stop()
        coord.join(threads)
        # 识别
        predict_filename, predict_image = readFilesOne()  # 读取文件
        predict_image_val = sess.run(predict_image, feed_dict={
                                     predict_filename: face_path + "/face/face0/359_a85b2b5f-3c83-412c-8224-771fead119b0.jpg"})
        score = model.predict(predict_image_val, steps=1)
        print("识别", score)
        predict_image_val = sess.run(predict_image, feed_dict={
                                     predict_filename: face_path + "/face/face0/359_6146e5d3-21c0-4f85-bfdb-24e161226ddc.jpg"})
        score = model.predict(predict_image_val, steps=1)
        print("识别", score)
        # 保存模型
        tf.keras.models.save_model(model, "my_model.h5")
    return


def predict():
    """识别"""
    ###############
    # 开始建立CNN模型
    ###############
    # 加载模型
    print("加载模型")
    model = createModel()
    model = tf.keras.models.load_model("my_model.h5")

    file_dir = [face_path + "/face/images_test_500/man/",
                face_path + "/face/images_test_500/women/"]
    label, image = readFilesBatch(file_dir, 250)  # 读取文件

    print("训练")
    # 训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 使用start_queue_runners之后，才会开始填充队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        image_value, label_value = image, label
        dataset = tf.data.Dataset.from_tensors((image_value, label_value))
        dataset = dataset.repeat()
        # # 识别
        score = model.evaluate(dataset, steps=2)
        print("损失值：", score[0])
        print("准确率：", score[1])
        # 停止填充队列
        coord.request_stop()
        coord.join(threads)
        # 识别
        predict_filename, predict_image = readFilesOne()  # 读取文件
        predict_image_val = sess.run(predict_image, feed_dict={
                                     predict_filename: face_path + "/face/face0/359_a85b2b5f-3c83-412c-8224-771fead119b0.jpg"})
        score = model.predict(predict_image_val, steps=1)
        print("识别", score)
        predict_image_val = sess.run(predict_image, feed_dict={
                                     predict_filename: face_path + "/face/face0/359_6146e5d3-21c0-4f85-bfdb-24e161226ddc.jpg"})
        score = model.predict(predict_image_val, steps=1)
        print("识别", score)
    return


def main(argv=None):  # 运行
    train()
    # predict()
    # f = h5py.File('my_model.h5', 'r+')
    # for key in f.keys():
    #     print(key)
    #     print(f[key])
    # f.close()


if __name__ == '__main__':
    tf.app.run()
