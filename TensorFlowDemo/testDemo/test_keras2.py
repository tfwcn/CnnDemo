from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
from keras import backend, optimizers, losses, metrics
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import save_model, load_model
import os
import numpy
import math
import pandas as pd

# face_path = "/media/ppht/000872BB00040832"
face_path = "D:/document/Share"


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


def readFilesBatch(file_dir):
    # 读取文件列表
    TEST_IMAGE_PATHS = []
    TEST_GENDER = []
    for i in range(len(file_dir)):
        pathDir = os.listdir(file_dir[i])
        # 定义特征与标签
        TEST_IMAGE_PATHS += [file_dir[i]+path for path in pathDir]
        TEST_GENDER += [int(path.split('_')[1])-1 for path in pathDir]

    features = pd.Series(TEST_IMAGE_PATHS)
    labels = pd.Series(TEST_GENDER)
    # 把列添加到表格
    data = pd.DataFrame({'文件路径': features, '标签': labels})
    # 随机排序
    data = data.reindex(numpy.random.permutation(data.index))

    features = data['文件路径'].values
    labels = data['标签'].values
    # print(len(labels), len(char_list))
    labels = dense_to_one_hot(labels, 2)  # 转化成1维数组
    return features, labels


def readFilesOne(filename):
    """读取单个文件，识别用"""
    features = load_img(filename, target_size=(160, 160))
    features = img_to_array(features, data_format="channels_last")
    features = features.astype('float32')
    features /= 255.0
    features = numpy.reshape(features, (-1, 160, 160, 3))
    return features


def generate_arrays_from_file(features, labels, batch_size):
    cnt = 0
    X = []
    Y = []
    datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)
    while 1:
        for line in range(len(features)):
            # create Numpy arrays of input data
            # and labels, from each line in the file
            # print("features:", features[line])
            x = load_img(features[line], target_size=(160, 160))
            x = img_to_array(x, data_format="channels_last")
            x = x.astype('float32')
            x /= 255.0
            # print("x:", x)
            y = labels[line]
            # print("x:", line, features[line])
            # print("y:", line, labels[line])
            X.append(x)
            Y.append(y)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                # print("batch")
                # print("x:", numpy.array(X).shape)
                # print("y:", numpy.array(Y).shape)
                # yield (numpy.array(X), numpy.array(Y))
                X = numpy.array(X)
                Y = numpy.array(Y)
                gen = datagen.flow(X, Y, batch_size=batch_size)
                yield next(gen)
                X = []
                Y = []


def createModel():
    keep_prob = 0.5  # dropout概率
    model = Sequential()
    model.add(Conv2D(30, (7, 7), strides=(
        4, 4), input_shape=(160, 160, 3), name="cnn1"))
    model.add(Activation(backend.relu))
    # model.add(MaxPool2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(30, (3, 3), strides=(2, 2), name="cnn2"))
    model.add(Dropout(keep_prob))
    model.add(BatchNormalization())

    model.add(Conv2D(80, (5, 5), name="cnn3"))
    model.add(Activation(backend.relu))
    # model.add(MaxPool2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(80, (3, 3), strides=(2, 2), name="cnn4"))
    model.add(Dropout(keep_prob))
    model.add(BatchNormalization())

    model.add(Conv2D(120, (5, 5), name="cnn5"))
    model.add(Activation(backend.relu))
    # model.add(MaxPool2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(120, (3, 3), strides=(2, 2), name="cnn6"))
    model.add(Dropout(keep_prob))
    model.add(BatchNormalization())

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
    model = Sequential()
    if os.path.isfile("data/my_model.h5"):
        model = load_model("data/my_model.h5")
        print("加载模型文件")
    else:
        model = createModel()

    # 读取文件
    file_dir = [face_path + "/face/face0/女_2929/",
                face_path + "/face/face0/男_2929/"]
    features, labels = readFilesBatch(file_dir)

    file_dir2 = [face_path + "/face/images_test_500/man/",
                 face_path + "/face/images_test_500/women/"]
    features2, labels2 = readFilesBatch(file_dir2)

    print("训练")
    # 训练
    model.fit_generator(generate_arrays_from_file(features, labels, 50), steps_per_epoch=100, validation_steps=1, epochs=30,
                        validation_data=generate_arrays_from_file(features2, labels2, 500))

    print("识别")
    score = model.evaluate_generator(
        generate_arrays_from_file(features2, labels2, 500), steps=1)
    print("损失值：", score[0])
    print("准确率：", score[1])

    predict_image_val = readFilesOne(
        face_path + "/face/face0/359_a85b2b5f-3c83-412c-8224-771fead119b0.jpg")
    score = model.predict(predict_image_val, steps=1)
    print("识别", score)

    predict_image_val = readFilesOne(
        face_path + "/face/face0/359_6146e5d3-21c0-4f85-bfdb-24e161226ddc.jpg")
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
    model = Sequential()
    model = load_model("data/my_model.h5")

    # 读取文件
    file_dir2 = [face_path + "/face/images_test_500/man/",
                 face_path + "/face/images_test_500/women/"]
    features2, labels2 = readFilesBatch(file_dir2)
    print("识别")
    score = model.evaluate_generator(
        generate_arrays_from_file(features2, labels2, 500), steps=1)
    print("损失值：", score[0])
    print("准确率：", score[1])

    predict_image_val = readFilesOne(
        face_path + "/face/face0/359_a85b2b5f-3c83-412c-8224-771fead119b0.jpg")
    score = model.predict(predict_image_val, steps=1)
    print("识别", score)

    predict_image_val = readFilesOne(
        face_path + "/face/face0/359_6146e5d3-21c0-4f85-bfdb-24e161226ddc.jpg")
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
    main()
