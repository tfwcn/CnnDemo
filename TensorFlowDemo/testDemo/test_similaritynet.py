import tensorflow as tf
import keras as K
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv2
import argparse
import random
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
    '''读取文件列表'''
    image_paths = []
    image_ids = []
    now_id = 0
    for i in range(len(file_dir)):
        # 读取所有文件夹
        pathDir = os.listdir(file_dir[i])
        # print(pathDir)
        for path in pathDir:
            pathDir2 = os.listdir(os.path.join(file_dir[i], path))
            for path2 in pathDir2:
                # 定义特征与标签
                image_paths += [os.path.join(file_dir[i], path, path2)]
                image_ids += [now_id]
            now_id += 1

    features = pd.Series(image_paths)
    labels = pd.Series(image_ids)
    # 把列添加到表格,同人图片ID相同
    data = pd.DataFrame({'image_paths': features, 'image_ids': labels})
    # 随机排序
    # data = data.reindex(np.random.permutation(data.index))
    # data = data.sample(frac=1)
    # print(data)
    print('图片总数：', len(data))
    return data


def hasImagesById(data, image_path, image_id):
    '''通过图片id判断是否有其他相同图片'''
    return len(data[(data['image_ids'] == image_id) & (data['image_paths'] != image_path)]) > 0


def getRandomImage(data):
    '''获取随机图片，有多张相同图片'''
    image = data.iloc[random.randint(0, len(data) - 1)]
    # print(image['image_paths'], image['image_ids'])
    while hasImagesById(data, image['image_paths'], image['image_ids']) == False:
        image = data.iloc[random.randint(0, len(data) - 1)]
        # print(image['image_paths'], image['image_ids'])
    return image


def findImagesById(data, image_path, image_id, is_same):
    '''通过图片id找图片列表'''
    if is_same:
        return data[(data['image_ids'] == image_id) & (data['image_paths'] != image_path)]
    else:
        return data[data['image_ids'] != image_id]


def findImageById(data, image_path, image_id, is_same):
    '''通过图片id随机找图片（一张）'''
    images = findImagesById(data, image_path, image_id, is_same)
    image_one = images.iloc[random.randint(0, len(images) - 1)]
    return image_one


def generate_arrays_from_file(data, batch_size, istrain=True):
    cnt = 0
    X1 = []
    X2 = []
    X3 = []
    Y1 = []
    Y2 = []
    Y3 = []
    while 1:
        # create Numpy arrays of input data
        # and labels, from each line in the file
        # print("features:", features[line])
        # 找一张有多张相同图片的图片
        image1 = getRandomImage(data)
        # 找一张相同的
        image2 = findImageById(
            data, image1['image_paths'], image1['image_ids'], True)
        # print(image2['image_paths'], image2['image_ids'])
        # 找一张不同的
        image3 = findImageById(
            data, image1['image_paths'], image1['image_ids'], False)
        # print(image3['image_paths'], image3['image_ids'])
        # x = K.preprocessing.image.load_img(
        #     features[line], target_size=(image_size[0], image_size[1]))
        x1 = K.preprocessing.image.load_img(image1['image_paths'])
        x1 = K.preprocessing.image.img_to_array(
            x1, data_format="channels_last")
        x1 = x1.astype('float32')
        x1 /= 255.0
        x1 = imageHelper.resize_padding_zero(x1, 512, 512)

        x2 = K.preprocessing.image.load_img(image2['image_paths'])
        x2 = K.preprocessing.image.img_to_array(
            x2, data_format="channels_last")
        x2 = x2.astype('float32')
        x2 /= 255.0
        x2 = imageHelper.resize_padding_zero(x2, 512, 512)

        X1.append(x1)
        X2.append(x2)

        if istrain:
            x3 = K.preprocessing.image.load_img(image3['image_paths'])
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

    # 读取文件
    file_dir = [face_path]
    data = readFilesBatch(file_dir)
    # file_dir=face_path
    # features = [[os.path.join(file_dir, 'Ai_Sugiyama_0001.jpg'), os.path.join(file_dir, 'Ai_Sugiyama_0002.jpg'),
    #              os.path.join(file_dir, 'Akbar_Hashemi_Rafsanjani_0001.jpg')],
    #             [os.path.join(file_dir, 'Ai_Sugiyama_0003.jpg'), os.path.join(file_dir, 'Ai_Sugiyama_0004.jpg'),
    #                 os.path.join(file_dir, 'Akbar_Hashemi_Rafsanjani_0002.jpg')],
    #             [os.path.join(file_dir, 'Ai_Sugiyama_0005.jpg'), os.path.join(file_dir, 'Ai_Sugiyama_0002.jpg'),
    #                 os.path.join(file_dir, 'Akbar_Hashemi_Rafsanjani_0003.jpg')],
    #             [os.path.join(file_dir, 'Akbar_Hashemi_Rafsanjani_0001.jpg'), os.path.join(file_dir, 'Akbar_Hashemi_Rafsanjani_0003.jpg'),
    #              os.path.join(file_dir, 'Ai_Sugiyama_0002.jpg')]]
    # print("labels",labels)

    # file_dir2 = [face_path]
    # features2 = readFilesBatch(file_dir2)
    # file_dir2 = face_path
    # features2 = [[os.path.join(file_dir2, 'Ai_Sugiyama_0001.jpg'), os.path.join(file_dir2, 'Ai_Sugiyama_0002.jpg')],
    #              [os.path.join(file_dir2, 'Ai_Sugiyama_0003.jpg'), os.path.join(
    #                  file_dir2, 'Akbar_Hashemi_Rafsanjani_0002.jpg')],
    #              [os.path.join(file_dir2, 'Ai_Sugiyama_0005.jpg'),
    #               os.path.join(file_dir2, 'Ai_Sugiyama_0002.jpg')],
    #              [os.path.join(file_dir2, 'Akbar_Hashemi_Rafsanjani_0001.jpg'), os.path.join(file_dir2, 'Ai_Sugiyama_0002.jpg')]]
    # print("features2",features2)

    # 测试模型
    image1 = getRandomImage(data)
    predict_image_val = readFilesOne(image1['image_paths'])
    score = similarityModel.model1.predict([predict_image_val], steps=1)
    print("识别A1", image1['image_paths'], score)
    image2 = getRandomImage(data)
    predict_image_val = readFilesOne(image2['image_paths'])
    score = similarityModel.model2.predict([predict_image_val], steps=1)
    print("识别A2", image2['image_paths'], score)
    image3 = getRandomImage(data)
    predict_image_val = readFilesOne(image3['image_paths'])
    score = similarityModel.model3.predict([predict_image_val], steps=1)
    print("识别A3", image3['image_paths'], score)

    # 训练
    # 准确率达到1时，提前停止训练
    # early_stopping = K.callbacks.EarlyStopping(monitor='is_ok', patience=0,min_delta=0, verbose=0, mode='max')
    callback1 = MyCallback()
    # 动态降低学习速率
    callback2 = K.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.8, patience=10, verbose=0, mode='min', min_delta=0.001, cooldown=0, min_lr=0)
    # model.fit_generator(generate_arrays_from_file(features, 10), steps_per_epoch=100, validation_steps=1, epochs=int(args.epochs),
    #                     validation_data=generate_arrays_from_file(features2, len(features2), False), callbacks=[callback1])
    tb_cb = K.callbacks.TensorBoard(
        log_dir='./logs/similarityModel', write_images=1)
    # model.fit_generator(generate_arrays_from_file(data, 1), steps_per_epoch=100, validation_steps=len(data), epochs=int(args.epochs),
    #                     validation_data=generate_arrays_from_file(data, 1))
    model.fit_generator(generate_arrays_from_file(data, 1), steps_per_epoch=100, epochs=int(args.epochs), callbacks=[tb_cb])

    # 测试模型
    predict_image_val = readFilesOne(image1['image_paths'])
    score = similarityModel.model1.predict([predict_image_val], steps=1)
    print("识别A1", image1['image_paths'], score)
    predict_image_val = readFilesOne(image2['image_paths'])
    score = similarityModel.model2.predict([predict_image_val], steps=1)
    print("识别A2", image2['image_paths'], score)
    predict_image_val = readFilesOne(image3['image_paths'])
    score = similarityModel.model3.predict([predict_image_val], steps=1)
    print("识别A3", image3['image_paths'], score)

    # print("识别")
    # score = similarityModel.model_predict.evaluate_generator(
    #     generate_arrays_from_file(features2, len(features2), False), steps=1)
    # print("识别0", score)

    model.save_weights("data/my_model_similaritynet1.h5")

    image1 = getRandomImage(data)
    # 找一张相同的
    image2 = findImageById(
        data, image1['image_paths'], image1['image_ids'], True)
    predict_image_val = readFilesOne(image1['image_paths'])
    predict_image_val2 = readFilesOne(image2['image_paths'])
    score = similarityModel.model_predict.predict(
        [predict_image_val, predict_image_val2], steps=1)
    print("比对识别1", image1['image_paths'], image2['image_paths'], score)
    # 找一张不同的
    image2 = findImageById(
        data, image1['image_paths'], image1['image_ids'], False)
    predict_image_val = readFilesOne(image1['image_paths'])
    predict_image_val2 = readFilesOne(image2['image_paths'])
    score = similarityModel.model_predict.predict(
        [predict_image_val, predict_image_val2], steps=1)
    print("比对识别2", image1['image_paths'], image2['image_paths'], score)
    return


def predict():
    """识别"""
    ###############
    # 开始建立CNN模型
    ###############

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

    # 读取文件
    file_dir = [face_path]
    data = readFilesBatch(file_dir)

    # 测试模型
    image1 = getRandomImage(data)
    predict_image_val = readFilesOne(image1['image_paths'])
    score = similarityModel.model1.predict([predict_image_val], steps=1)
    print("识别A1", image1['image_paths'], score)
    image2 = getRandomImage(data)
    predict_image_val = readFilesOne(image2['image_paths'])
    score = similarityModel.model2.predict([predict_image_val], steps=1)
    print("识别A2", image2['image_paths'], score)
    image3 = getRandomImage(data)
    predict_image_val = readFilesOne(image3['image_paths'])
    score = similarityModel.model3.predict([predict_image_val], steps=1)
    print("识别A3", image3['image_paths'], score)

    image1 = getRandomImage(data)
    # 找一张相同的
    image2 = findImageById(
        data, image1['image_paths'], image1['image_ids'], True)
    predict_image_val = readFilesOne(image1['image_paths'])
    predict_image_val2 = readFilesOne(image2['image_paths'])
    score = similarityModel.model_predict.predict(
        [predict_image_val, predict_image_val2], steps=1)
    print("比对识别1", image1['image_paths'], image2['image_paths'], score)
    # 找一张不同的
    image2 = findImageById(
        data, image1['image_paths'], image1['image_ids'], False)
    predict_image_val = readFilesOne(image1['image_paths'])
    predict_image_val2 = readFilesOne(image2['image_paths'])
    score = similarityModel.model_predict.predict(
        [predict_image_val, predict_image_val2], steps=1)
    print("比对识别2", image1['image_paths'], image2['image_paths'], score)
    return


def main(argv=None):  # 运行
    if args.train == "1":
        train()
    else:
        predict()


if __name__ == '__main__':
    main()
