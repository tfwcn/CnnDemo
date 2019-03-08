import tensorflow as tf
import keras as K
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv2
import argparse
from models.RegressModel import RegressModel
from models.ImageHelper import ImageHelper

parser = argparse.ArgumentParser()
parser.add_argument('labels_path')
parser.add_argument('-t', '--train', default="1")
parser.add_argument('-e', '--epochs', default="10")
args = parser.parse_args()

face_path = args.labels_path

image_size = (512,512)

imageHelper = ImageHelper()

def randomTransformation(rgb):
    h, w = rgb.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
    x1 = np.maximum((np.random.random()-0.6)*w, -w*0.2)
    y1 = np.maximum((np.random.random()-0.6)*h, -h*0.2)
    x2 = np.maximum((np.random.random()-0.6)*w, -w*0.2)
    y2 = np.minimum((np.random.random()+0.6)*h, h-1)
    x3 = np.minimum((np.random.random()+0.6)*w, w-1)
    y3 = np.minimum((np.random.random()+0.6)*h, h-1)
    x4 = np.minimum((np.random.random()+0.6)*w, w-1)
    y4 = np.maximum((np.random.random()-0.6)*h, -h*0.2)
    # print([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    M = cv2.getPerspectiveTransform(pts, pts1)
    dst = cv2.warpPerspective(rgb, M, rgb.shape[:2])
    # print("pts1_1",pts1)
    pts1 = np.float32(pts1 / ((w-1,h-1),(w-1,h-1),(w-1,h-1),(w-1,h-1)))
    # print("pts1_2",pts1)
    # print("M", M)
    return dst, pts1


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
            if os.path.isfile(features[line]) == False:
                continue
            # x = K.preprocessing.image.load_img(
            #     features[line], target_size=(image_size[0], image_size[1]))
            x = K.preprocessing.image.load_img(
                features[line])
            x = K.preprocessing.image.img_to_array(
                x, data_format="channels_last")
            x = x.astype('float32')
            x /= 255.0
            x, y = randomTransformation(x)
            x = imageHelper.resize_padding_zero(x,512,512)
            # print("y:",y.shape)
            # y = np.reshape(y, (9))
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
    features = K.preprocessing.image.load_img(filename)
    features = K.preprocessing.image.img_to_array(
        features, data_format="channels_last")
    features = features.astype('float32')
    features /= 255.0
    features = imageHelper.resize_padding_zero(features,512,512)
    # features, y = randomTransformation(features)
    features = np.reshape(features, (-1, image_size[0], image_size[1], 3))
    return features


def createModel():
    # 使用InceptionV3网络,构建不带分类器的预训练模型
    input_value = K.layers.Input(shape=(image_size[0], image_size[1], 3), name="input_value")

    # 接收图片，回归结果
    regressModel = RegressModel(256)
    model = regressModel.create_model(input_value)
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

def myloss(y_true, y_pred):
    # print("y_true",y_true)
    # print("y_pred",y_pred)
    # diff = K.layers.Subtract(y_true - y_pred)
    # # 小于1=1，否则0
    # less_than_one = K.backend.cast(K.backend.less(diff, 1.0), "float32")
    # loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return (y_true - y_pred)

def train():
    """训练"""
    ###############
    # 开始建立CNN模型
    ###############
    print("训练")
    # 加载模型
    model = createModel()
    if os.path.isfile("data/my_model_transformation3.h5"):
        model.load_weights("data/my_model_transformation3.h5", by_name=True)
        print("加载模型文件")

    # 编译模型
    # model.compile(K.optimizers.Adadelta(lr=1e-4),
    #               K.losses.categorical_crossentropy, [K.metrics.categorical_crossentropy])
    model.compile(K.optimizers.Adadelta(lr=0.005),
                  K.losses.mean_squared_error, [K.metrics.mean_squared_error])

    # 读取文件
    file_dir = [face_path]
    features = readFilesBatch(file_dir)
    # print("labels",labels)

    file_dir2 = [face_path+"test/"]
    features2 = readFilesBatch(file_dir2)
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
    model.fit_generator(generate_arrays_from_file(features, 10), steps_per_epoch=100, validation_steps=1, epochs=int(args.epochs),
                        validation_data=generate_arrays_from_file(features2, len(features2), False))

    print("识别")
    score = model.evaluate_generator(
        generate_arrays_from_file(features2, len(features2), False), steps=1)
    print("损失值：", score[0])
    print("准确率：", score[1])

    model.save_weights("data/my_model_transformation3.h5")

    pathDir = os.listdir(file_dir2[0])
    for path in pathDir:
        predict_image_val = readFilesOne(file_dir2[0]+path)
        score = model.predict(predict_image_val, steps=1)
        predict_image_val = predict_image_val.reshape((image_size[0], image_size[1], 3))
        print("识别路径", path)
        print("识别", score[0])
        # dst = cv2.warpPerspective(
        #     predict_image_val, np.reshape(score[0], (3, 3)), predict_image_val.shape[:2])
        # 透视矩阵变换
        h, w = predict_image_val.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
        pts1 = np.float32(score[0] * ((w-1,h-1),(w-1,h-1),(w-1,h-1),(w-1,h-1)))
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
        predict_image_val = predict_image_val.reshape((image_size[0], image_size[1], 3))
        print("识别路径", path)
        print("识别", score[0])
        h, w = predict_image_val.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
        pts1 = np.float32(score[0] * ((w-1,h-1),(w-1,h-1),(w-1,h-1),(w-1,h-1)))
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
    # 读取文件
    # file_dir = [face_path]
    # features = readFilesBatch(file_dir)
    # x, y = next(generate_arrays_from_file(features, 20))
    # print("x2:", x.shape)
    # print("y:", y)
    # for i in x:
    #     plt.imshow(i)  # 显示图片
    #     plt.axis('off')  # 不显示坐标轴
    #     plt.show()
    
    # x = K.preprocessing.image.load_img(
    #     "E:/Labels/yibiao/140a4f4536923f24d736e88b2fd2ed7f.jpg", target_size=(image_size[0], image_size[1]))
    # x = K.preprocessing.image.img_to_array(
    #     x, data_format="channels_last")
    # x = x.astype('float32')
    # x /= 255.0
    # x, y = randomTransformation(x)
    # predict_image_val=x
    # h, w = predict_image_val.shape[:2]
    # pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
    # pts1 = np.float32(y * ((w-1,h-1),(w-1,h-1),(w-1,h-1),(w-1,h-1)))
    # M = cv2.getPerspectiveTransform(pts1, pts)
    # dst = cv2.warpPerspective(
    #     predict_image_val, M, predict_image_val.shape[:2])
    # plt.imshow(predict_image_val)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
    # plt.imshow(dst)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()


if __name__ == '__main__':
    main()
