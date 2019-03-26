import keras as K
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('face_path')
parser.add_argument('-t', '--train', default="1")
args = parser.parse_args()

face_path = args.face_path


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
    data = data.reindex(np.random.permutation(data.index))

    features = data['文件路径'].values
    return features


def generate_arrays_from_file(features, batch_size):
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
            y = K.preprocessing.image.load_img(
                features[line], target_size=(120, 120))
            y = K.preprocessing.image.img_to_array(
                y, data_format="channels_last")
            y = y.astype('float32')
            y /= 255.0
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
                yield (np.array(X), np.array(Y))
                X = []
                Y = []


def createModel():
    input_value = K.Input((60, 60, 3), name="input")
    print("input_value:", input_value.shape)
    # 反向卷积，大小扩大一倍
    x1 = K.layers.Conv2D(64, (1, 1), activation="relu", padding="same",
                                            name="x1_conv1")(input_value)
    x1 = K.layers.BatchNormalization()(x1)
    x1 = K.layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                                            name="x1_conv2")(x1)
    x1 = K.layers.BatchNormalization()(x1)
    x1 = K.layers.Conv2D(64, (1, 1), activation="relu", padding="same",
                                            name="x1_conv3")(x1)
    x1 = K.layers.BatchNormalization()(x1)

    # 并列x2
    x2 = K.layers.Conv2D(64, (5, 5), activation="relu", padding="same",
                                            name="x2_conv1")(input_value)
    x2 = K.layers.BatchNormalization()(x2)

    # 并列x3
    x3 = K.layers.Conv2D(64, (2, 2), activation="relu", padding="same",
                                            name="x3_conv1")(input_value)
    x3 = K.layers.BatchNormalization()(x3)
    x3 = K.layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                                            name="x3_conv2")(x3)
    x3 = K.layers.BatchNormalization()(x3)

    output_value = K.layers.Add()([x1, x2, x3])
    # 反卷积
    # output_value = K.layers.Conv2DTranspose(128, (2, 2), strides=2, activation="relu",
    #                                         name="mrcnn_mask_deconv")(output_value)
    output_value = K.layers.UpSampling2D((2, 2),
                                            name="mrcnn_mask_deconv")(output_value)
    output_value = K.layers.BatchNormalization()(output_value)
    output_value = K.layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                                            name="output_value_conv1")(output_value)
    output_value = K.layers.BatchNormalization()(output_value)
    output_value = K.layers.Conv2D(128, (1, 1), activation="relu", padding="same",
                                            name="output_value_conv2")(output_value)
    output_value = K.layers.BatchNormalization()(output_value)
    # (1,1)卷积提取特征，sigmoid激活函数，值范围转到0-1
    # 回归模型不用激活函数
    output_value = K.layers.Conv2D(3, (1, 1), strides=1, activation=None,
                                   name="mrcnn_mask")(output_value)
    model = K.Model(inputs=input_value, outputs=output_value, name="test")
    return model


def train():
    """训练"""
    ###############
    # 开始建立CNN模型
    ###############
    # 加载模型
    model = createModel()

    # 编译模型, compile主要完成损失函数和优化器的一些配置，是为训练服务的。
    model.compile(K.optimizers.Adam(),
                  K.losses.mean_squared_error, [K.metrics.mean_squared_error])

    if os.path.isfile("data/my_model3.h5"):
        model = K.models.load_model("data/my_model3.h5")
        print("加载模型文件")

    # 读取文件
    file_dir = [face_path + "/face0/女_2929/",
                face_path + "/face0/男_2929/"]
    features = readFilesBatch(file_dir)

    file_dir2 = [face_path + "/face0/女_250/",
                 face_path + "/face0/男_250/"]
    features2 = readFilesBatch(file_dir2)

    print("训练")
    # 训练
    model.fit_generator(generate_arrays_from_file(features, 50), steps_per_epoch=100, validation_steps=10, epochs=10,
                        validation_data=generate_arrays_from_file(features2, 50))

    print("识别")
    score = model.evaluate_generator(
        generate_arrays_from_file(features2, 50), steps=10)
    print("损失值：", score[0])
    print("准确率：", score[1])
    return


def predict():
    """识别"""
    ###############
    # 开始建立CNN模型
    ###############
    # 加载模型
    print("加载模型")
    model = createModel()
    model = K.models.load_model("data/my_model3.h5")

    # 读取文件
    print("识别")

    predict_image_val2 = K.preprocessing.image.load_img(
        face_path + "/face0/359_a85b2b5f-3c83-412c-8224-771fead119b0.jpg", target_size=(120, 120))
    predict_image_val2 = K.preprocessing.image.img_to_array(
        predict_image_val2, data_format="channels_last")
    predict_image_val2 = predict_image_val2.astype('float32')
    predict_image_val2 /= 255.0
    predict_image_val = K.preprocessing.image.load_img(
        face_path + "/face0/359_a85b2b5f-3c83-412c-8224-771fead119b0.jpg", target_size=(60, 60))
    predict_image_val = K.preprocessing.image.img_to_array(
        predict_image_val, data_format="channels_last")
    predict_image_val = predict_image_val.astype('float32')
    predict_image_val /= 255.0
    old_image = predict_image_val
    predict_image_val = np.reshape(predict_image_val, (-1, 60, 60, 3))
    score = model.predict(predict_image_val, steps=1)
    score *= 255
    score = np.reshape(score, (120, 120, 3))
    score = score.astype("int32")
    score = np.maximum(score,0)
    score = np.minimum(score,255)
    print("识别", score.shape)
    plt.imshow(old_image)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    plt.imshow(predict_image_val2)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    plt.imshow(score)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    predict_image_val2 = K.preprocessing.image.load_img(
        face_path + "/face0/359_6146e5d3-21c0-4f85-bfdb-24e161226ddc.jpg", target_size=(120, 120))
    predict_image_val2 = K.preprocessing.image.img_to_array(
        predict_image_val2, data_format="channels_last")
    predict_image_val2 = predict_image_val2.astype('float32')
    predict_image_val2 /= 255.0
    predict_image_val = K.preprocessing.image.load_img(
        face_path + "/face0/359_6146e5d3-21c0-4f85-bfdb-24e161226ddc.jpg", target_size=(60, 60))
    predict_image_val = K.preprocessing.image.img_to_array(
        predict_image_val, data_format="channels_last")
    predict_image_val = predict_image_val.astype('float32')
    predict_image_val /= 255.0
    old_image = predict_image_val
    predict_image_val = np.reshape(predict_image_val, (-1, 60, 60, 3))
    score = model.predict(predict_image_val, steps=1)
    score *= 255
    score = np.reshape(score, (120, 120, 3))
    score = score.astype("int32")
    print("识别", score.shape)
    plt.imshow(old_image)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    plt.imshow(predict_image_val2)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    plt.imshow(score)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    K.models.save_model(model, "data/my_model3.h5")
    return


def main(argv=None):  # 运行
    if args.train == "1":
        train()
    else:
        predict()
    # f = h5py.File('my_model.h5', 'r+')
    # for key in f.keys():
    #     print(key)
    #     print(f[key])
    # f.close()


if __name__ == '__main__':
    main()
