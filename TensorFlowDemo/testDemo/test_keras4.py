import keras as K
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('labels_path')
parser.add_argument('-t', '--train', default="1")
args = parser.parse_args()

face_path = args.labels_path


def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = len(labels_dense)  # 数组长度
    index_offset = np.arange(num_labels) * num_classes  # 根据数组生成序号
    labels_one_hot = np.zeros([num_labels, num_classes])  # 初始化二维数组
    for i in range(num_labels):  # 赋值
        if np.isnan(labels_dense[i]):
            continue
        labels_one_hot.flat[int(
            index_offset[i] + labels_dense[i])] = 1  # 一维化赋值
    return labels_one_hot.tolist()


def randomHSV(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # 通道拆分
    (h, s, v) = cv2.split(hsv)
    # 0.8概率随机颜色
    if np.random.random() < 0.8:
        h_value = np.random.uniform(0, 360)
        # print("h_value",type(h_value))
        h = (np.random.uniform(-10, 10, size=h.shape).astype(np.float32) + h_value)
        h = np.maximum(h, 0)
        h = np.minimum(h, 360)
        # print("h",type(h[0][0]))
        # print("s",type(s[0][0]))
        # print("v",type(v[0][0]))
    # 合并通道
    hsv = cv2.merge([h, s, v])
    # print("hsv",hsv.shape)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def readFilesBatch(file_dir):
    # 读取文件列表
    TEST_IMAGE_PATHS = []
    TEST_GENDER = []
    for i in range(len(file_dir)):
        pathDir = os.listdir(file_dir[i])
        # 定义特征与标签
        TEST_IMAGE_PATHS += [file_dir[i]+path for path in pathDir]
        TEST_GENDER += [int(path[0]) for path in pathDir]

    features = pd.Series(TEST_IMAGE_PATHS)
    labels = pd.Series(TEST_GENDER)
    # 把列添加到表格
    data = pd.DataFrame({'文件路径': features, '标签': labels})
    # 随机排序
    # data = data.reindex(np.random.permutation(data.index))
    data = data.sample(frac=1)
    # print(data)

    features = data['文件路径'].values
    labels = data['标签'].values
    # print(len(labels), len(char_list))
    labels = dense_to_one_hot(labels, 10)  # 转化成1维数组
    return features, labels


def generate_arrays_from_file(features, labels, batch_size, istrain=True):
    cnt = 0
    X = []
    Y = []
    datagen = K.preprocessing.image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=[0.7, 1],
        channel_shift_range=0.2,
        shear_range=0.3,
        # validation_split=0.1
    )
    while 1:
        for line in range(len(features)):
            # create Numpy arrays of input data
            # and labels, from each line in the file
            # print("features:", features[line])
            x = K.preprocessing.image.load_img(
                features[line], target_size=(60, 60))
            x = K.preprocessing.image.img_to_array(
                x, data_format="channels_last")
            x = x.astype('float32')
            x /= 255.0
            x = randomHSV(x)
            # print("x:",x.shape)
            y = labels[line]
            # print("x:", line, features[line])
            # print("y:", line, labels[line])
            X.append(x)
            Y.append(y)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                # print("batch")
                # print("x:", np.array(X).shape)
                # print("y:", np.array(Y).shape)
                X = np.array(X)
                Y = np.array(Y)
                if istrain:
                    gen = datagen.flow(X, Y, batch_size=batch_size)
                    yield next(gen)
                else:
                    yield X, Y
                X = []
                Y = []


def readFilesOne(filename):
    """读取单个文件，识别用"""
    features = K.preprocessing.image.load_img(filename, target_size=(60, 60))
    features = K.preprocessing.image.img_to_array(
        features, data_format="channels_last")
    features = features.astype('float32')
    features /= 255.0
    features = np.reshape(features, (-1, 60, 60, 3))
    return features


def createModel():
    keep_prob = 0.9  # dropout概率
    input_value = K.Input((60, 60, 3), name="input")
    print("input_value:", input_value.shape)
    # 卷积
    x = K.layers.Conv2D(32, (7, 7), strides=(4, 4),
                        activation=K.backend.relu,
                        # kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.05),
                        name="conv1", padding="same")(input_value)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.MaxPool2D((3, 3), strides=(2, 2), name="pool1")(x)
    # x = K.layers.Dropout(keep_prob, name="dropout1")(x)

    x = K.layers.Conv2D(64, (5, 5),
                        activation=K.backend.relu,
                        # kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.05),
                        name="conv2", padding="same")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.MaxPool2D((3, 3), strides=(2, 2), name="pool2")(x)
    x2 = K.layers.Conv2D(128, (5, 5),
                         activation=K.backend.relu,
                         # kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.05),
                         name="conv2_2", padding="same")(x)
    x2 = K.layers.BatchNormalization()(x2)
    # x = K.layers.Dropout(keep_prob, name="dropout2")(x)

    x = K.layers.Conv2D(128, (3, 3),
                        activation=K.backend.relu,
                        # kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.05),
                        name="conv3", padding="same")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Conv2D(128, (2, 2),
                        activation=K.backend.relu,
                        # kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.05),
                        name="conv3_2", padding="same")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Add()([x, x2])
    x = K.layers.Activation(K.backend.relu)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.MaxPool2D((3, 3), strides=(2, 2), name="pool3")(x)
    # x = K.layers.Dropout(keep_prob, name="dropout3")(x)

    # 转一维
    x = K.layers.Flatten(name="Flatten")(x)
    x = K.layers.Dense(100,
                       # activation=K.backend.relu,
                       # kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.05),
                       name="dn1")(x)
    x = K.layers.Activation(K.backend.relu)(x)
    x = K.layers.BatchNormalization()(x)
    # x = K.layers.Dropout(keep_prob, name="dropout4")(x)

    output_value = K.layers.Dense(10, activation=K.backend.softmax,
                                  # kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.05),
                                  name="dn2")(x)

    model = K.Model(inputs=input_value, outputs=output_value, name="test")

    return model


class MyCallback(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # print("MyCallback",self)
        # print("MyCallback",epoch)
        print("MyCallback", logs)
        # print("MyCallback self.model",self.model)
        if logs["val_binary_accuracy"] >= 1:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_batch_end(self, batch, logs=None):
        # print("MyCallback on_batch_end",logs)
        # # 准确率低于0.95，有0.8概率重新训练
        # if logs["binary_accuracy"]<0.95 and np.random.random()<0.8:
        #     logs["batch"]-=1
        #     batch-=1
        #     print("MyCallback on_batch_end2",logs)
        pass


def train():
    """训练"""
    ###############
    # 开始建立CNN模型
    ###############
    # 加载模型
    model = K.Model()
    if os.path.isfile("data/my_model4.h5"):
        model = K.models.load_model("data/my_model4.h5")
        print("加载模型文件")
    else:
        model = createModel()

    # 编译模型
    model.compile(K.optimizers.Adadelta(lr=1e-3),
                  K.losses.categorical_crossentropy, [K.metrics.binary_accuracy])

    # 读取文件
    file_dir = [face_path+"labels/0-9/train/"]
    features, labels = readFilesBatch(file_dir)
    # print("labels",labels)

    file_dir2 = [face_path+"labels/0-9/test/"]
    features2, labels2 = readFilesBatch(file_dir2)

    print("训练")
    # 训练
    # 准确率达到1时，提前停止训练
    # early_stopping = K.callbacks.EarlyStopping(monitor='is_ok', patience=0,min_delta=0, verbose=0, mode='max')
    callback1 = MyCallback()
    # 动态降低学习速率
    callback2 = K.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=10, verbose=0, mode='min', min_delta=0.001, cooldown=0, min_lr=0)
    model.fit_generator(generate_arrays_from_file(features, labels, 50), steps_per_epoch=100, validation_steps=1, epochs=200,
                        validation_data=generate_arrays_from_file(features2, labels2, len(labels2), False), callbacks=[callback1, callback2])

    print("识别")
    score = model.evaluate_generator(
        generate_arrays_from_file(features2, labels2, len(labels2), False), steps=1)
    print("损失值：", score[0])
    print("准确率：", score[1])

    K.models.save_model(model, "data/my_model4.h5")

    pathDir = os.listdir(file_dir2[0])
    for path in pathDir:
        predict_image_val = readFilesOne(file_dir2[0]+path)
        score = model.predict(predict_image_val, steps=1)
        print("识别路径", path)
        print("识别", np.argmax(score, axis=1))
        plt.imshow(predict_image_val[0])  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.show()
    return


def predict():
    """识别"""
    ###############
    # 开始建立CNN模型
    ###############
    # 加载模型
    print("加载模型")
    model = K.Model()
    model = K.models.load_model("data/my_model4.h5")

    # 读取文件
    file_dir2 = [face_path+"labels/0-9/test/"]
    features2, labels2 = readFilesBatch(file_dir2)

    # pathDir = os.listdir(file_dir2[0])
    # for i in range(len(features2)):
    #     print("标签：", labels2[i])
    #     predict_image_val = readFilesOne(features2[i])
    #     plt.imshow(predict_image_val[0])  # 显示图片
    #     plt.axis('off')  # 不显示坐标轴
    #     plt.show()

    # print("识别", labels2)
    score = model.evaluate_generator(
        generate_arrays_from_file(features2, labels2, len(labels2), False), steps=1, max_q_size=1)
    print("损失值：", score[0])
    print("准确率：", score[1])

    pathDir = os.listdir(file_dir2[0])
    for path in pathDir:
        predict_image_val = readFilesOne(file_dir2[0]+path)
        score = model.predict(predict_image_val, steps=1)
        print("识别路径", path)
        print("识别", np.argmax(score, axis=1))
        plt.imshow(predict_image_val[0])  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.show()
    return


def main(argv=None):  # 运行
    if args.train == "1":
        train()
    else:
        predict()
    # 读取文件
    # file_dir = ["D:/document/Share/labels/0-9/train/"]
    # features, labels = readFilesBatch(file_dir)
    # x, y = generate_arrays_from_file(features, labels, 20)
    # print("x2:", x.shape)
    # for i in x:
    #     plt.imshow(i)  # 显示图片
    #     plt.axis('off')  # 不显示坐标轴
    #     plt.show()


if __name__ == '__main__':
    main()
