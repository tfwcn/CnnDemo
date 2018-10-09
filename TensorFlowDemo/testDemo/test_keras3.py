import keras as K
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# face_path = "D:/document/Share"
face_path = "E:"


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
    output_value = K.layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu",
                                            name="mrcnn_mask_deconv")(input_value)
    # (1,1)卷积提取特征，sigmoid激活函数，值范围转到0-1
    output_value = K.layers.Conv2D(3, (1, 1), strides=1, activation="sigmoid",
                                   name="mrcnn_mask")(output_value)
    model = K.Model(inputs=input_value, outputs=output_value, name="test")

    # 编译模型
    model.compile(K.optimizers.Adadelta(),
                  K.losses.categorical_crossentropy, [K.metrics.binary_accuracy])

    return model


def train():
    """训练"""
    ###############
    # 开始建立CNN模型
    ###############
    # 加载模型
    model = createModel()
    if os.path.isfile("my_model.h5"):
        model = K.models.load_model("my_model.h5")
        print("加载模型文件")

    # 读取文件
    file_dir = [face_path + "/face/face0/女_2929/",
                face_path + "/face/face0/男_2929/"]
    features = readFilesBatch(file_dir)

    file_dir2 = [face_path + "/face/images_test_500/man/",
                 face_path + "/face/images_test_500/women/"]
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

    predict_image_val = K.preprocessing.image.load_img(
        face_path + "/face/face0/359_a85b2b5f-3c83-412c-8224-771fead119b0.jpg", target_size=(60, 60))
    predict_image_val = K.preprocessing.image.img_to_array(
        predict_image_val, data_format="channels_last")
    predict_image_val = predict_image_val.astype('float32')
    predict_image_val /= 255.0
    predict_image_val = np.reshape(predict_image_val, (-1, 60, 60, 3))
    score = model.predict(predict_image_val, steps=1)
    score *= 255
    score = np.reshape(score, (120, 120, 3))
    score = score.astype("int32")
    print("识别", score.shape)
    plt.imshow(score)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    predict_image_val = K.preprocessing.image.load_img(
        face_path + "/face/face0/359_6146e5d3-21c0-4f85-bfdb-24e161226ddc.jpg", target_size=(60, 60))
    predict_image_val = K.preprocessing.image.img_to_array(
        predict_image_val, data_format="channels_last")
    predict_image_val = predict_image_val.astype('float32')
    predict_image_val /= 255.0
    predict_image_val = np.reshape(predict_image_val, (-1, 60, 60, 3))
    score = model.predict(predict_image_val, steps=1)
    score *= 255
    score = np.reshape(score, (120, 120, 3))
    score = score.astype("int32")
    print("识别", score.shape)
    plt.imshow(score)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    K.models.save_model(model, "my_model.h5")
    return


def predict():
    """识别"""
    ###############
    # 开始建立CNN模型
    ###############
    # 加载模型
    print("加载模型")
    model = createModel()
    model = K.models.load_model("my_model.h5")

    # 读取文件
    file_dir2 = [face_path + "/face/images_test_500/man/",
                 face_path + "/face/images_test_500/women/"]
    features2 = readFilesBatch(file_dir2)
    print("识别")
    score = model.evaluate_generator(
        generate_arrays_from_file(features2, 50), steps=10)
    print("损失值：", score[0])
    print("准确率：", score[1])

    predict_image_val = K.preprocessing.image.load_img(
        face_path + "/face/face0/359_a85b2b5f-3c83-412c-8224-771fead119b0.jpg", target_size=(60, 60))
    predict_image_val = K.preprocessing.image.img_to_array(
        predict_image_val, data_format="channels_last")
    predict_image_val = predict_image_val.astype('float32')
    predict_image_val /= 255.0
    predict_image_val = np.reshape(predict_image_val, (-1, 60, 60, 3))
    score = model.predict(predict_image_val, steps=1)
    score *= 255
    score = np.reshape(score, (120, 120, 3))
    score = score.astype("int32")
    print("识别", score.shape)
    plt.imshow(score)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    predict_image_val = K.preprocessing.image.load_img(
        face_path + "/face/face0/359_6146e5d3-21c0-4f85-bfdb-24e161226ddc.jpg", target_size=(60, 60))
    predict_image_val = K.preprocessing.image.img_to_array(
        predict_image_val, data_format="channels_last")
    predict_image_val = predict_image_val.astype('float32')
    predict_image_val /= 255.0
    predict_image_val = np.reshape(predict_image_val, (-1, 60, 60, 3))
    score = model.predict(predict_image_val, steps=1)
    score *= 255
    score = np.reshape(score, (120, 120, 3))
    score = score.astype("int32")
    print("识别", score.shape)
    plt.imshow(score)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
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
