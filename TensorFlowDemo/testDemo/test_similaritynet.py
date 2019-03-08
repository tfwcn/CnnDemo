import tensorflow as tf
import keras as K
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv2
import argparse
from models.SimilarityModel import SimilarityModel
from models.ImageHelper import ImageHelper

parser = argparse.ArgumentParser()
parser.add_argument('labels_path')
parser.add_argument('-t', '--train', default="1")
parser.add_argument('-e', '--epochs', default="10")
args = parser.parse_args()

face_path = args.labels_path

image_size = (512, 512)

imageHelper = ImageHelper()

# 相似度模型
similarityModel = SimilarityModel()


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
    X1 = []
    X2 = []
    X3 = []
    Y1 = []
    Y2 = []
    Y3 = []
    while 1:
        for line in range(len(features)):
            # create Numpy arrays of input data
            # and labels, from each line in the file
            # print("features:", features[line])
            if os.path.isfile(features[line][0]) == False:
                continue
            if os.path.isfile(features[line][1]) == False:
                continue
            if istrain and os.path.isfile(features[line][2]) == False:
                continue
            # x = K.preprocessing.image.load_img(
            #     features[line], target_size=(image_size[0], image_size[1]))
            x1 = K.preprocessing.image.load_img(
                features[line][0])
            x1 = K.preprocessing.image.img_to_array(
                x1, data_format="channels_last")
            x1 = x1.astype('float32')
            x1 /= 255.0
            x1 = imageHelper.resize_padding_zero(x1, 512, 512)

            x2 = K.preprocessing.image.load_img(
                features[line][1])
            x2 = K.preprocessing.image.img_to_array(
                x2, data_format="channels_last")
            x2 = x2.astype('float32')
            x2 /= 255.0
            x2 = imageHelper.resize_padding_zero(x2, 512, 512)

            X1.append(x1)
            X2.append(x2)

            if istrain:
                x3 = K.preprocessing.image.load_img(
                    features[line][2])
                x3 = K.preprocessing.image.img_to_array(
                    x3, data_format="channels_last")
                x3 = x3.astype('float32')
                x3 /= 255.0
                x3 = imageHelper.resize_padding_zero(x3, 512, 512)
                X3.append(x3)

            # Y是固定值0
            Y1.append(0)
            Y2.append(1)
            Y3.append(0)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                X1 = np.array(X1)
                X2 = np.array(X2)
                X3 = np.array(X3)
                Y1 = np.array(Y1)
                Y2 = np.array(Y2)
                Y3 = np.array(Y3)
                # print('Y', Y.shape)
                if istrain:
                    yield [X1, X2, X3], [Y1]
                else:
                    yield [X1, X2], [Y1]
                X1 = []
                X2 = []
                X3 = []
                Y1 = []
                Y2 = []
                Y3 = []


def readFilesOne(filename):
    """读取单个文件，识别用"""
    features = K.preprocessing.image.load_img(filename)
    features = K.preprocessing.image.img_to_array(
        features, data_format="channels_last")
    features = features.astype('float32')
    features /= 255.0
    features = imageHelper.resize_padding_zero(features, 512, 512)
    features = np.reshape(features, (-1, image_size[0], image_size[1], 3))
    return features


def createModel():
    # 使用InceptionV3网络,构建不带分类器的预训练模型
    # 输入特征
    input_value = K.layers.Input(
        shape=(image_size[0], image_size[1], 3), name="input_value")
    # 锚点
    input_anchor_value = K.layers.Input(
        shape=(image_size[0], image_size[1], 3), name="input_anchor_value")
    # 正样本
    input_positive_value = K.layers.Input(
        shape=(image_size[0], image_size[1], 3), name="input_positive_value")
    # 负样本
    input_negative_value = K.layers.Input(
        shape=(image_size[0], image_size[1], 3), name="input_negative_value")

    # 接收图片，回归结果
    model = similarityModel.create_model(
        input_value, input_anchor_value, input_positive_value, input_negative_value)
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
    print("训练")

    # 设置显存占用自适应
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_session = tf.Session(config=tf_config)
    K.backend.set_session(tf_session)

    # 加载模型
    model = createModel()
    if os.path.isfile("data/my_model_similaritynet1.h5"):
        model.load_weights("data/my_model_similaritynet1.h5", by_name=True)
        print("加载模型文件")

    # 编译模型
    similarityModel.compile()

    # 测试模型
    predict_image_val = readFilesOne(os.path.join(face_path, 'Ai_Sugiyama_0001.jpg'))
    score = similarityModel.model1.predict([predict_image_val], steps=1)
    print("识别A1", score)
    predict_image_val = readFilesOne(os.path.join(face_path, 'Ai_Sugiyama_0002.jpg'))
    score = similarityModel.model2.predict([predict_image_val], steps=1)
    print("识别A2", score)
    predict_image_val = readFilesOne(os.path.join(face_path, 'Akbar_Hashemi_Rafsanjani_0001.jpg'))
    score = similarityModel.model3.predict([predict_image_val], steps=1)
    print("识别A3", score)

    # 读取文件
    file_dir = [face_path]
    # features = readFilesBatch(file_dir)
    file_dir=face_path
    features = [[os.path.join(file_dir, 'Ai_Sugiyama_0001.jpg'), os.path.join(file_dir, 'Ai_Sugiyama_0002.jpg'),
                 os.path.join(file_dir, 'Akbar_Hashemi_Rafsanjani_0001.jpg')],
                [os.path.join(file_dir, 'Ai_Sugiyama_0003.jpg'), os.path.join(file_dir, 'Ai_Sugiyama_0004.jpg'),
                    os.path.join(file_dir, 'Akbar_Hashemi_Rafsanjani_0002.jpg')],
                [os.path.join(file_dir, 'Ai_Sugiyama_0005.jpg'), os.path.join(file_dir, 'Ai_Sugiyama_0002.jpg'),
                    os.path.join(file_dir, 'Akbar_Hashemi_Rafsanjani_0003.jpg')],
                [os.path.join(file_dir, 'Akbar_Hashemi_Rafsanjani_0001.jpg'), os.path.join(file_dir, 'Akbar_Hashemi_Rafsanjani_0003.jpg'),
                 os.path.join(file_dir, 'Ai_Sugiyama_0002.jpg')]]
    # print("labels",labels)

    file_dir2 = [face_path]
    file_dir2=face_path
    # features2 = readFilesBatch(file_dir2)
    features2 = [[os.path.join(file_dir2, 'Ai_Sugiyama_0001.jpg'), os.path.join(file_dir2, 'Ai_Sugiyama_0002.jpg')],
                [os.path.join(file_dir2, 'Ai_Sugiyama_0003.jpg'), os.path.join(file_dir2, 'Akbar_Hashemi_Rafsanjani_0002.jpg')],
                [os.path.join(file_dir2, 'Ai_Sugiyama_0005.jpg'), os.path.join(file_dir2, 'Ai_Sugiyama_0002.jpg')],
                [os.path.join(file_dir2, 'Akbar_Hashemi_Rafsanjani_0001.jpg'), os.path.join(file_dir2, 'Ai_Sugiyama_0002.jpg')]]
    # print("features2",features2)

    # 训练
    # 准确率达到1时，提前停止训练
    # early_stopping = K.callbacks.EarlyStopping(monitor='is_ok', patience=0,min_delta=0, verbose=0, mode='max')
    callback1 = MyCallback()
    # 动态降低学习速率
    callback2 = K.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.8, patience=10, verbose=0, mode='min', min_delta=0.001, cooldown=0, min_lr=0)
    # model.fit_generator(generate_arrays_from_file(features, 10), steps_per_epoch=100, validation_steps=1, epochs=int(args.epochs),
    #                     validation_data=generate_arrays_from_file(features2, len(features2), False), callbacks=[callback1])
    tb_cb = K.callbacks.TensorBoard(log_dir='./logs/similarityModel', write_images=1)
    model.fit_generator(generate_arrays_from_file(features, 3), steps_per_epoch=100, validation_steps=len(features), epochs=int(args.epochs),
                        validation_data=generate_arrays_from_file(features, 1), callbacks=[tb_cb])

    # 测试模型
    predict_image_val = readFilesOne(os.path.join(face_path, 'Ai_Sugiyama_0001.jpg'))
    score = similarityModel.model1.predict([predict_image_val], steps=1)
    print("识别1", score)
    predict_image_val = readFilesOne(os.path.join(face_path, 'Ai_Sugiyama_0002.jpg'))
    score = similarityModel.model2.predict([predict_image_val], steps=1)
    print("识别2", score)
    predict_image_val = readFilesOne(os.path.join(face_path, 'Akbar_Hashemi_Rafsanjani_0001.jpg'))
    score = similarityModel.model3.predict([predict_image_val], steps=1)
    print("识别3", score)

    # print("识别")
    # score = similarityModel.model_predict.evaluate_generator(
    #     generate_arrays_from_file(features2, len(features2), False), steps=1)
    # print("识别0", score)

    model.save_weights("data/my_model_similaritynet1.h5")

    predict_image_val = readFilesOne(os.path.join(face_path, 'Ai_Sugiyama_0001.jpg'))
    predict_image_val2 = readFilesOne(os.path.join(face_path, 'Ai_Sugiyama_0002.jpg'))
    score = similarityModel.model_predict.predict([predict_image_val,predict_image_val2], steps=1)
    print("比对识别1", score)
    predict_image_val = readFilesOne(os.path.join(face_path, 'Ai_Sugiyama_0001.jpg'))
    predict_image_val2 = readFilesOne(os.path.join(face_path, 'Akbar_Hashemi_Rafsanjani_0001.jpg'))
    score = similarityModel.model_predict.predict([predict_image_val,predict_image_val2], steps=1)
    print("比对识别2", score)
    return


def predict():
    """识别"""
    ###############
    # 开始建立CNN模型
    ###############
    # 加载模型
    model = createModel()
    if os.path.isfile("data/my_model_transformation3.h5"):
        model.load_weights("data/my_model_transformation3.h5", by_name=True)
        print("加载模型文件")

    # 编译模型
    model.compile(K.optimizers.Adadelta(lr=1e-4),
                  K.losses.categorical_crossentropy, [K.metrics.binary_accuracy])

    # 读取文件
    file_dir2 = [face_path+"test/"]
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
        generate_arrays_from_file(features2, len(features2), False), steps=1)
    print("损失值：", score[0])
    print("准确率：", score[1])

    pathDir = os.listdir(file_dir2[0])
    for path in pathDir:
        predict_image_val = readFilesOne(file_dir2[0]+path)
        score = model.predict(predict_image_val, steps=1)
        predict_image_val = predict_image_val.reshape(
            (image_size[0], image_size[1], 3))
        print("识别路径", path)
        print("识别", score[0])
        h, w = predict_image_val.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
        pts1 = np.float32(
            score[0] * ((w-1, h-1), (w-1, h-1), (w-1, h-1), (w-1, h-1)))
        M = cv2.getPerspectiveTransform(pts1, pts)
        dst = cv2.warpPerspective(
            predict_image_val, M, predict_image_val.shape[:2])
        plt.imshow(predict_image_val)  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.show()
        plt.imshow(dst)  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.show()
    return


def main(argv=None):  # 运行
    if args.train == "1":
        train()
    else:
        predict()


if __name__ == '__main__':
    main()
