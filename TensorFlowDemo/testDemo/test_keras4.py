import keras as K
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

face_path = "D:/document/Share"
# face_path = "E:"

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
    data = data.reindex(np.random.permutation(data.index))

    features = data['文件路径'].values
    labels = data['标签'].values
    # print(len(labels), len(char_list))
    labels = dense_to_one_hot(labels, 10)  # 转化成1维数组
    return features, labels


def generate_arrays_from_file(features, labels, batch_size):
    cnt = 0
    X = []
    Y = []
    datagen = K.preprocessing.image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=[0.8,1.2],
        channel_shift_range=0.2)
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
                gen = datagen.flow(X, Y, batch_size=batch_size)
                yield next(gen)
                X = []
                Y = []


def readFilesOne(filename):
    """读取单个文件，识别用"""
    features = K.preprocessing.image.load_img(filename, target_size=(60, 60))
    features = K.preprocessing.image.img_to_array(features, data_format="channels_last")
    features = features.astype('float32')
    features /= 255.0
    features = np.reshape(features, (-1, 60, 60, 3))
    return features


def createModel():
    keep_prob = 0.5  # dropout概率
    input_value = K.Input((60, 60, 3), name="input")
    print("input_value:", input_value.shape)
    # 反向卷积，大小扩大一倍
    x = K.layers.Conv2D(16, (7, 7), activation="relu", name="conv1", padding="same")(input_value)
    x = K.layers.MaxPool2D((2, 2), strides=(2, 2), name="pool1")(x)
    x = K.layers.Dropout(keep_prob, name="dropout1")(x)
    # x = K.layers.BatchNormalization(name="bn1")(x)

    x = K.layers.Conv2D(32, (3, 3), activation="relu", name="conv2", padding="same")(x)
    x = K.layers.MaxPool2D((2, 2), strides=(2, 2), name="pool2")(x)
    x = K.layers.Dropout(keep_prob, name="dropout2")(x)
    # x = K.layers.BatchNormalization(name="bn2")(x)

    x = K.layers.Conv2D(64, (3, 3), activation="relu", name="conv3", padding="same")(x)
    x = K.layers.MaxPool2D((2, 2), strides=(2, 2), name="pool3")(x)
    x = K.layers.Dropout(keep_prob, name="dropout3")(x)
    # x = K.layers.BatchNormalization(name="bn3")(x)
    # 转一维
    x = K.layers.Flatten(name="Flatten")(x)
    x = K.layers.Dense(100, activation="relu", name="dn1")(x)
    x = K.layers.Dropout(keep_prob, name="dropout4")(x)

    x = K.layers.Dense(10, activation="softmax", name="dn2")(x)
    output_value = K.layers.Dropout(keep_prob, name="dropout5")(x)

    model = K.Model(inputs=input_value, outputs=output_value, name="test")

    # 编译模型
    model.compile(K.optimizers.Adadelta(lr=1e-4),
                  K.losses.categorical_crossentropy, [K.metrics.binary_accuracy])

    return model


def train():
    """训练"""
    ###############
    # 开始建立CNN模型
    ###############
    # 加载模型
    model = K.Model()
    if os.path.isfile("data/my_model.h5"):
        model = load_model("data/my_model.h5")
        print("加载模型文件")
    else:
        model = createModel()

    # 读取文件
    file_dir = ["D:/document/Share/labels/0-9/train/"]
    features, labels = readFilesBatch(file_dir)
    # print("labels",labels)

    file_dir2 = ["D:/document/Share/labels/0-9/train/"]
    features2, labels2 = readFilesBatch(file_dir2)

    print("训练")
    # 训练
    model.fit_generator(generate_arrays_from_file(features, labels, 50), steps_per_epoch=100, validation_steps=1, epochs=10,
                        validation_data=generate_arrays_from_file(features2, labels2, 10))

    print("识别")
    score = model.evaluate_generator(
        generate_arrays_from_file(features2, labels2, 500), steps=1)
    print("损失值：", score[0])
    print("准确率：", score[1])

    predict_image_val = readFilesOne("D:/document/Share/labels/0-9/train/8.png")
    score = model.predict(predict_image_val, steps=1)
    print("识别", score)

    predict_image_val = readFilesOne("D:/document/Share/labels/0-9/train/5.png")
    score = model.predict(predict_image_val, steps=1)
    print("识别", score)
    save_model(model, "data/my_model.h5")
    return


def predict():
    """识别"""
    ###############
    # 开始建立CNN模型
    ###############
    # 加载模型
    print("加载模型")
    model = K.Model()
    model = load_model("data/my_model.h5")

    # 读取文件
    file_dir2 = ["D:/document/Share/labels/0-9/train/"]
    features2, labels2 = readFilesBatch(file_dir2)
    print("识别")
    score = model.evaluate_generator(
        generate_arrays_from_file(features2, labels2, 10), steps=1)
    print("损失值：", score[0])
    print("准确率：", score[1])

    predict_image_val = readFilesOne("D:/document/Share/labels/0-9/train/6.png")
    score = model.predict(predict_image_val, steps=1)
    print("识别", score)

    predict_image_val = readFilesOne("D:/document/Share/labels/0-9/train/9.png")
    score = model.predict(predict_image_val, steps=1)
    print("识别", score)
    return


def main(argv=None):  # 运行
    train()
    # predict()
    # predict_image_val = readFilesOne("D:/document/Share/labels/0-9/train/8.png")


if __name__ == '__main__':
    main()
