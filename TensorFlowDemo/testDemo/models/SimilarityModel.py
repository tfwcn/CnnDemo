import keras as K
import tensorflow as tf


class SimilarityModel():
    # 图片相似度
    def __init__(self):
        self.alpha = 0.2

    def create_model_one(self, input_value):
        """单个模型，用于特征提取"""
        inputs = input_value
        # resnet50模型
        inputs = self.resnet50(inputs)
        print("image_model.output", inputs.shape)
        # 压缩维度
        inputs = K.layers.Conv2D(512, (3, 3),
                                 activation=K.activations.relu,
                                 name="SimilarityModel_Conv1", padding="same")(inputs)
        inputs = K.layers.MaxPooling2D(
            (2, 2), name="SimilarityModel_Pool3")(inputs)
        print("inputs1", inputs.shape)
        inputs = K.layers.Conv2D(512, (int(inputs.shape[1]), int(inputs.shape[2])),
                                 activation=K.activations.relu,
                                 name="SimilarityModel_Conv2", padding="valid")(inputs)
        inputs = K.layers.Conv2D(256, (1, 1),
                                 activation=K.activations.linear,
                                 name="SimilarityModel_Conv3", padding="same")(inputs)
        # 扁平化
        # inputs = K.layers.Flatten()(inputs)
        # inputs = K.layers.Dense(self.embedding_dim,
        #                         activation=K.activations.softmax)(inputs)
        # print("inputs1",inputs.shape)
        # inputs = K.layers.Dense(8,
        #                         activation=K.activations.linear)(inputs)
        print("inputs2", inputs.shape)
        inputs = K.layers.Lambda(lambda x: tf.reshape(x, (-1, 256)))(inputs)
        print("inputs3", inputs.shape)
        output_value = inputs
        print("output_value", output_value.shape)
        # 共享权重模型
        moedl = K.Model(inputs=input_value,
                                 outputs=output_value, name="SimilarityModel_One")

        return moedl

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = K.layers.Conv2D(filters1, (1, 1),
                            kernel_initializer='he_normal',
                            name=conv_name_base + '2a')(input_tensor)
        x = K.layers.BatchNormalization(name=bn_name_base + '2a')(x)
        x = K.layers.Activation('relu')(x)

        x = K.layers.Conv2D(filters2, kernel_size,
                            padding='same',
                            kernel_initializer='he_normal',
                            name=conv_name_base + '2b')(x)
        x = K.layers.BatchNormalization(name=bn_name_base + '2b')(x)
        x = K.layers.Activation('relu')(x)

        x = K.layers.Conv2D(filters3, (1, 1),
                            kernel_initializer='he_normal',
                            name=conv_name_base + '2c')(x)
        x = K.layers.BatchNormalization(name=bn_name_base + '2c')(x)

        x = K.layers.add([x, input_tensor])
        x = K.layers.Activation('relu')(x)
        return x

    def conv_block(self, input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2)):
        """A block that has a conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.

        # Returns
            Output tensor for the block.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = K.layers.Conv2D(filters1, (1, 1), strides=strides,
                            kernel_initializer='he_normal',
                            name=conv_name_base + '2a')(input_tensor)
        x = K.layers.BatchNormalization(name=bn_name_base + '2a')(x)
        x = K.layers.Activation('relu')(x)

        x = K.layers.Conv2D(filters2, kernel_size, padding='same',
                            kernel_initializer='he_normal',
                            name=conv_name_base + '2b')(x)
        x = K.layers.BatchNormalization(name=bn_name_base + '2b')(x)
        x = K.layers.Activation('relu')(x)

        x = K.layers.Conv2D(filters3, (1, 1),
                            kernel_initializer='he_normal',
                            name=conv_name_base + '2c')(x)
        x = K.layers.BatchNormalization(name=bn_name_base + '2c')(x)

        shortcut = K.layers.Conv2D(filters3, (1, 1), strides=strides,
                                   kernel_initializer='he_normal',
                                   name=conv_name_base + '1')(input_tensor)
        shortcut = K.layers.BatchNormalization(
            name=bn_name_base + '1')(shortcut)

        x = K.layers.add([x, shortcut])
        x = K.layers.Activation('relu')(x)
        return x

    def resnet50(self, img_input):
        """resnet50模型"""
        x = K.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = K.layers.Conv2D(64, (7, 7),
                            strides=(2, 2),
                            padding='valid',
                            kernel_initializer='he_normal',
                            name='conv1')(x)
        x = K.layers.BatchNormalization(name='bn_conv1')(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = K.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256],
                            stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        return x

    def create_model(self, input_value, input_anchor_value, input_positive_value, input_negative_value):
        self.input_value = input_value
        self.input_anchor_value = input_anchor_value
        self.input_positive_value = input_positive_value
        self.input_negative_value = input_negative_value
        # 创建共享模型
        self.moedl_one = self.create_model_one(self.input_value)
        # 特征值对应输出
        self.output_anchor_value = self.moedl_one(self.input_anchor_value)
        self.output_positive_value = self.moedl_one(self.input_positive_value)
        self.output_negative_value = self.moedl_one(self.input_negative_value)
        print('input_anchor_value.shape',self.input_anchor_value.shape)
        print('output_anchor_value.shape',self.output_anchor_value.shape)
        # loss当作输出来处理
        loss = K.layers.Lambda(lambda x: self.triplet_loss(x[0], x[1], x[2]),name='loss')(
            [self.output_anchor_value, self.output_positive_value, self.output_negative_value])
        # 正比较
        loss_positive = K.layers.Lambda(lambda x: self.euclidean_distance(x[0], x[1]),name='loss_positive')(
            [self.output_anchor_value, self.output_positive_value])
        # 负比较
        loss_negative = K.layers.Lambda(lambda x: self.euclidean_distance(x[0], x[1]),name='loss_negative')(
            [self.output_anchor_value, self.output_negative_value])


        self.model = K.Model(inputs=[self.input_anchor_value, self.input_positive_value, self.input_negative_value],
                             outputs=[loss],
                             name="SimilarityModel")

        # 用于显示相似度的模型
        self.model_predict = K.Model(inputs=[self.input_anchor_value, self.input_positive_value],
                             outputs=[loss_positive],
                             name="SimilarityModel_Predict")

        #测试是否共享权重
        self.model1 = K.Model(inputs=self.input_anchor_value,
                             outputs=self.output_anchor_value,
                             name="SimilarityModel1")
        self.model2 = K.Model(inputs=self.input_positive_value,
                             outputs=self.output_positive_value,
                             name="SimilarityModel2")
        self.model3 = K.Model(inputs=self.input_negative_value,
                             outputs=self.output_negative_value,
                             name="SimilarityModel3")

        return self.model
    

    def compile(self):
        """编译模型"""
        def lossfun(y_true, y_pred):
            print('loss', y_true, y_pred)
            return y_pred
        def lossfun2(y_true, y_pred):
            print('loss', y_true, y_pred)
            return y_pred-1

        self.model.compile(K.optimizers.Adadelta(lr=0.005),
                           loss=[lossfun],
                           metrics=[K.metrics.mean_squared_error])
        self.model_predict.compile(K.optimizers.Adadelta(lr=0.005),
                           loss=[K.losses.mean_squared_error],metrics=[K.metrics.mean_squared_error])
                           
        #测试是否共享权重
        self.model1.compile(K.optimizers.Adadelta(lr=0.005),
                           loss=[lossfun],metrics=[K.metrics.mean_squared_error])
        self.model2.compile(K.optimizers.Adadelta(lr=0.005),
                           loss=[lossfun],metrics=[K.metrics.mean_squared_error])
        self.model3.compile(K.optimizers.Adadelta(lr=0.005),
                           loss=[lossfun],metrics=[K.metrics.mean_squared_error])

    def triplet_loss(self, anchor, positive, negative):
        """Calculate the triplet loss according to the FaceNet paper

        Args:
        anchor: the embeddings for the anchor images.
        positive: the embeddings for the positive images.
        negative: the embeddings for the negative images.

        Returns:
        the triplet loss according to the FaceNet paper as a float tensor.
        """
        print('anchor',anchor.shape)
        with tf.variable_scope('triplet_loss'):
            pos_dist = tf.reduce_sum(
                tf.square(tf.subtract(anchor, positive)), 1)
            neg_dist = tf.reduce_sum(
                tf.square(tf.subtract(anchor, negative)), 1)
            # print('neg_dist',neg_dist.shape)

            basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.alpha)
            # print('basic_loss',basic_loss.shape)
            loss = tf.reshape(tf.maximum(basic_loss, 0.0),(-1,1))
            print('loss',loss.shape)

        return loss

    def euclidean_distance(self, anchor, positive):
        # 欧氏距离
        with tf.variable_scope('euclidean_distance'):
            pos_dist = tf.reduce_sum(
                tf.square(tf.subtract(anchor, positive)), 1)

            loss = tf.reshape(tf.maximum(pos_dist, 0.0),(-1,1))

        return loss


