import tensorflow as tf
import keras as K
import numpy as np
import os
import argparse
import pandas as pd

# 读取参数
parser = argparse.ArgumentParser()
# 图片文件夹路径
parser.add_argument('file_path')
# 是训练还是识别，1：训练，0：识别
parser.add_argument('-t', '--train', default="1")
# 训练循环次数
parser.add_argument('-e', '--epochs', default="10")
# 例如启动命令：python ./SexClassifyDemo_keras.py '图片文件夹路径' -t=1 -e=10
# 例如启动命令：python ./SexClassifyDemo_keras.py '图片文件夹路径' --train=1 --epochs=10

# 所有参数存到args变量
args = parser.parse_args()

# 拿取图片文件夹路径
file_path = args.file_path


def readFilesBatch(file_dir):
    # file_dir是路径的数组，['路径1','路径2']
    # 从文件夹读取完整的图片路径，用来在训练过程中读取图片
    TEST_IMAGE_PATHS = []
    TEST_LABELS = []
    # 循环所有路径
    for i in range(len(file_dir)):
        # 读路径下的文件
        pathDir = os.listdir(file_dir[i])
        # 定义特征与标签，TEST_IMAGE_PATHS包含图片路径的列表，TEST_LABELS相对的标签的列表
        # 这里是性别分类，所以标签里 0.男 1.女
        TEST_IMAGE_PATHS += [file_dir[i]+path for path in pathDir]
        TEST_LABELS += [int(path.split('_')[1])-1 for path in pathDir]

    # 这里建立pandas.Series,类似与数据库的列数据
    # features特征列的数据，labels标签列的数据
    features = pd.Series(TEST_IMAGE_PATHS)
    labels = pd.Series(TEST_LABELS)
    # 把列添加到表格
    data = pd.DataFrame({'文件路径': features, '标签': labels})
    # 产生随机排序的数据，用于数据打乱
    data = data.sample(frac=1)
    # print(data)
    # 拿打乱顺序后的表数据
    features = data['文件路径'].values
    labels = data['标签'].values
    # one_hot，把0转成[1,0],把1转成[0,1]
    # 第二参数为分类数
    labels = K.utils.to_categorical(labels, 2)
    return features, labels


def generate_arrays_from_file(features, labels, batch_size):
    # 训练过程中批量读取图片
    cnt = 0
    X = []
    Y = []
    while 1:
        for line in range(len(features)):
            # 判断是否文件
            if os.path.isfile(features[line]) == False:
                continue
            # 读取图片
            x = K.preprocessing.image.load_img(
                features[line], target_size=(160, 160))
            # 图片转数组，按[宽，高，通道数]格式
            x = K.preprocessing.image.img_to_array(
                x, data_format="channels_last")
            # 把图片颜色归一化，由0-255转成0-1
            x = x.astype('float32')
            x /= 255.0

            # 加到数组中
            X.append(x)

            # Y是标签
            Y.append(labels[line])
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                X = np.array(X)
                Y = np.array(Y)
                # 返回一批结果，方法还没结束
                # X(3,160,160,3),Y(3,2)
                print('X', X)
                print('Y', Y)
                yield X, Y
                X = []
                Y = []


def readFilesOne(filename):
    """读取单个文件，识别用"""
    features = K.preprocessing.image.load_img(filename, target_size=(160, 160))
    features = K.preprocessing.image.img_to_array(
        features, data_format="channels_last")
    features = features.astype('float32')
    features /= 255.0
    features = np.reshape(features, (-1, 160, 160, 3))
    return features


def createModel():
    # 输入特征
    input_value = K.layers.Input(
        shape=(160, 160, 3), name="input_value")
    # 第一层卷积,卷积层厚度64,卷积核大小(7,7),偏移量(4,4),激活函数为relu。
    # padding='valid'为正常卷积，卷积后大小变小，'same'为卷积后大小与输入大小相同。
    x = K.layers.Conv2D(64, (7, 7),
                        strides=(4, 4),
                        padding='valid',
                        kernel_initializer=K.initializers.normal(
                            0.0, 0.01),  # 初始化权重范围（-0.01-0.01）
                        activation=K.activations.relu,
                        name='conv1')(input_value)
    # 池化，(3,3)池化核大小，偏移量(2,2)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), name='max_pool1')(x)
    # Dropout,忽略掉0.3的神经元
    x = K.layers.Dropout(0.3, name='dropout1')(x)

    # 第二层卷积,卷积层厚度64,卷积核大小(5,5),偏移量(1,1)。
    # padding='valid'为正常卷积，卷积后大小变小，'same'为卷积后大小与输入大小相同。
    x = K.layers.Conv2D(200, (5, 5),
                        padding='valid',
                        kernel_initializer=K.initializers.normal(0.0, 0.01),
                        activation=K.activations.relu,
                        name='conv2')(input_value)
    # 池化，(3,3)池化核大小，偏移量(2,2)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), name='max_pool2')(x)
    # Dropout,忽略掉0.3的神经元
    x = K.layers.Dropout(0.3, name='dropout2')(x)

    # 二维数据转一维
    x = K.layers.Flatten()(x)

    # 全连接层1，长度128
    x = K.layers.Dense(128, activation=K.activations.relu, name='dense1')(x)
    # 全连接层2，长度128
    x = K.layers.Dense(2, activation=K.activations.softmax, name='dense2')(x)

    model = K.Model(input=input_value, output=x, name='SexClassifyModel')

    return model


def train():
    """训练"""
    print('开始训练')
    # 设置显存占用自适应
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_session = tf.Session(config=tf_config)
    K.backend.set_session(tf_session)

    # 读取文件
    file_dir = [file_path]
    # 读路径下的文件夹
    pathDir = os.listdir(file_path)
    for path in pathDir:
        tmp_dir = os.path.join(file_path, path)
        if os.path.isdir(tmp_dir):
            file_dir += [tmp_dir]
    features, labels = readFilesBatch(file_dir)

    # 构造模型
    model = createModel()
    # 加载模型权重
    if os.path.isfile("./SexClassifyDemo_keras.h5"):
        model.load_weights("./SexClassifyDemo_keras.h5", by_name=True)
        print("加载模型文件")

    # 编译模型,梯度下降用Adadelta，loss用交叉熵。
    # metrics为测试准确率时的参数，不参与训练。
    model.compile(K.optimizers.Adadelta(lr=0.005),
                  loss=[K.losses.categorical_crossentropy],
                  metrics=[K.metrics.categorical_crossentropy])

    # 开始训练模型，steps_per_epoch训练步数，epochs训练次数，validation_steps识别步数，
    # 每训练一批测试准确率一次
    # generate_arrays_from_file训练或测试数据生成器
    model.fit_generator(generate_arrays_from_file(features, labels, 3), steps_per_epoch=100,
                        validation_steps=len(features), epochs=int(args.epochs),
                        validation_data=generate_arrays_from_file(features, labels, 1))

    # 保存权重
    model.save_weights('./SexClassifyDemo_keras.h5')


def predict():
    """识别"""
    print('开始识别')
    # 构造模型
    model = createModel()
    # 加载模型权重
    if os.path.isfile("./SexClassifyDemo_keras.h5"):
        model.load_weights("./SexClassifyDemo_keras.h5", by_name=True)
        print("加载模型文件")
    # 读取图片
    predict_image_val = readFilesOne(file_path)
    # 识别，score为返回的结果
    score = model.predict([predict_image_val], steps=1)
    print('识别结果：', score)


def main():  # 运行
    print('args', args)
    if args.train == "1":
        train()
    else:
        predict()


if __name__ == '__main__':
    main()
