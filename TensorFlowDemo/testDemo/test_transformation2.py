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


def randomTransformation(rgb):
    h, w = rgb.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
    pts1 = np.float32([[0, 0], [0, h-1*0.5], [w-1*0.6, h-1*0.7], [w-1*0.8, 0]])
    M = cv2.getPerspectiveTransform(pts, pts1)
    dst = cv2.warpPerspective(rgb, M, rgb.shape[:2])
    # print("hsv",hsv.shape)
    return rgb, M


def readFilesBatch(file_dir):
    # 读取文件列表
    TEST_IMAGE_PATHS = []
    for i in range(len(file_dir)):
        pathDir = os.listdir(file_dir[i])
        # 定义特征与标签
        TEST_IMAGE_PATHS += [file_dir[i]+path for path in pathDir]

    features = pd.Series(TEST_IMAGE_PATHS)
    # 把列添加到表格
    data = pd.DataFrame({'文件路径': features})
    # 随机排序
    # data = data.reindex(np.random.permutation(data.index))
    data = data.sample(frac=1)
    # print(data)

    features = data['文件路径'].values
    return features


def generate_arrays_from_file(features, batch_size, istrain=True):
    cnt = 0
    X = []
    Y = []
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
            x, y = randomTransformation(x)
            # print("x:",x.shape)
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
    input_value = K.Input((60, 60, 3), name="input")
    print("input_value:", input_value.shape)
    # 卷积
    x = K.layers.Conv2D(32, (7, 7), strides=(4, 4),
                        activation=K.backend.relu,
                        name="conv1", padding="same")(input_value)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.MaxPool2D((3, 3), strides=(2, 2), name="pool1")(x)

    x = K.layers.Conv2D(64, (5, 5),
                        activation=K.backend.relu,
                        name="conv2", padding="same")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.MaxPool2D((3, 3), strides=(2, 2), name="pool2")(x)

    x2 = K.layers.Conv2D(64, (1, 1),
                         activation=K.backend.relu,
                         name="conv2_2", padding="same")(x)
    x2 = K.layers.BatchNormalization()(x2)
    x2 = K.layers.Conv2D(64, (3, 3),
                         activation=K.backend.relu,
                         name="conv2_3", padding="same")(x2)
    x2 = K.layers.BatchNormalization()(x2)
    x2 = K.layers.Conv2D(64, (1, 1),
                         activation=K.backend.relu,
                         name="conv2_4", padding="same")(x2)
    x2 = K.layers.BatchNormalization()(x2)

    x3 = x

    x = K.layers.Conv2D(64, (3, 3),
                        activation=K.backend.relu,
                        name="conv3", padding="same")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Conv2D(64, (2, 2),
                        activation=K.backend.relu,
                        name="conv3_2", padding="same")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Add()([x, x2, x3])
    x = K.layers.Activation(K.backend.relu)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Conv2D(128, (3, 3),
                        activation=K.backend.relu,
                        name="conv4", padding="same")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.MaxPool2D((3, 3), strides=(2, 2), name="pool3")(x)

    # FCN转一维
    x = K.layers.Conv2D(100, x.shape,
                        activation=K.backend.relu,
                        name="conv5", padding="valid")(x)
    x = K.layers.BatchNormalization()(x)

    x = K.layers.Conv2D(6, x.shape,
                        activation=K.backend.softmax,
                        name="conv6", padding="valid")(x)
    output_value = K.layers.Reshape((3, 2))(x)
    print("output_value.shape", output_value.shape)

    model = K.Model(inputs=input_value, outputs=output_value, name="test")

    return model


class MyCallback(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # print("MyCallback",self)
        # print("MyCallback",epoch)
        print("MyCallback", logs)
        # print("MyCallback self.model",self.model)
        if logs["val_binary_accuracy"] >= 0.997:
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
    if os.path.isfile("data/my_model_transformation2.h5"):
        model = K.models.load_model("data/my_model_transformation2.h5")
        print("加载模型文件")
    else:
        model = createModel()

    # 编译模型
    model.compile(K.optimizers.SGD(lr=1e-3, momentum=0.5, decay=0.001),
                  K.losses.categorical_crossentropy, [K.metrics.binary_accuracy])

    # 读取文件
    file_dir = [face_path+"labels/0-9/train/"]
    features = readFilesBatch(file_dir)
    # print("labels",labels)

    file_dir2 = [face_path+"labels/0-9/test/"]
    features2 = readFilesBatch(file_dir2)

    print("训练")
    # 训练
    # 准确率达到1时，提前停止训练
    # early_stopping = K.callbacks.EarlyStopping(monitor='is_ok', patience=0,min_delta=0, verbose=0, mode='max')
    callback1 = MyCallback()
    # 动态降低学习速率
    callback2 = K.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.8, patience=10, verbose=0, mode='min', min_delta=0.001, cooldown=0, min_lr=0)
    model.fit_generator(generate_arrays_from_file(features, 50), steps_per_epoch=100, validation_steps=1, epochs=100,
                        validation_data=generate_arrays_from_file(features2, len(features2), False), callbacks=[callback1, callback2])

    print("识别")
    score = model.evaluate_generator(
        generate_arrays_from_file(features2, len(features2), False), steps=1)
    print("损失值：", score[0])
    print("准确率：", score[1])

    K.models.save_model(model, "data/my_model_transformation2.h5")

    pathDir = os.listdir(file_dir2[0])
    for path in pathDir:
        predict_image_val = readFilesOne(file_dir2[0]+path)
        score = model.predict(predict_image_val, steps=1)
        print("识别路径", path)
        print("识别", np.argmax(score, axis=1))
        # plt.imshow(predict_image_val[0])  # 显示图片
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()
    return


def predict():
    """识别"""
    ###############
    # 开始建立CNN模型
    ###############
    # 加载模型
    print("加载模型")
    model = K.Model()
    model = K.models.load_model("data/my_model_transformation2.h5")

    # 读取文件
    file_dir2 = [face_path+"labels/0-9/test/"]
    features2 = readFilesBatch(file_dir2)

    # pathDir = os.listdir(file_dir2[0])
    # for i in range(len(features2)):
    #     print("标签：", labels2[i])
    #     predict_image_val = readFilesOne(features2[i])
    #     plt.imshow(predict_image_val[0])  # 显示图片
    #     plt.axis('off')  # 不显示坐标轴
    #     plt.show()

    # print("识别", labels2)
    score = model.evaluate_generator(
        generate_arrays_from_file(features2, len(features2), False), steps=1, max_q_size=1)
    print("损失值：", score[0])
    print("准确率：", score[1])

    pathDir = os.listdir(file_dir2[0])
    for path in pathDir:
        predict_image_val = readFilesOne(file_dir2[0]+path)
        score = model.predict(predict_image_val, steps=1)
        print("识别路径", path)
        print("识别", np.argmax(score, axis=1))
        # plt.imshow(predict_image_val[0])  # 显示图片
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()
    return


def main(argv=None):  # 运行
    # if args.train == "1":
    #     train()
    # else:
    #     predict()
    # 读取文件
    file_dir = [face_path]
    features = readFilesBatch(file_dir)
    x, y = generate_arrays_from_file(features, 20)
    print("x2:", x.shape)
    for i in x:
        plt.imshow(i)  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.show()


if __name__ == '__main__':
    main()
