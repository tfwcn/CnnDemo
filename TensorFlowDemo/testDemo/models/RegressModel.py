import keras as K
import tensorflow as tf


class RegressModel():
    # 图片回归
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def create_model(self, input_value):
        inputs = input_value
        # 使用InceptionV3网络,构建不带分类器的预训练模型
        # image_model = K.applications.inception_v3.InceptionV3(input_tensor=input_value, weights=None, include_top=False)
        # image_model = K.applications.resnet50.ResNet50(
        #     input_tensor=input_value, weights=None, include_top=False)
        # inputs = image_model.output
        # print("image_model.output", inputs.shape)
        # 压缩维度
        inputs = K.layers.Conv2D(64, (7, 7), strides=(4, 4),
                            activation=K.backend.relu,
                            name="conv1", padding="same",
                            kernel_regularizer=K.regularizers.l2(0.05))(inputs)
        inputs = K.layers.MaxPooling2D((3, 3), strides=(2, 2), name="pool1")(inputs)

        inputs = self.layers_block(inputs,128,"block1")
        inputs = K.layers.MaxPooling2D(
            (2, 2), name="RegressModel_Pool1")(inputs)

        inputs = self.layers_block(inputs,256,"block2")
        inputs = K.layers.MaxPooling2D(
            (2, 2), name="RegressModel_Pool2")(inputs)

        inputs = self.layers_block(inputs,512,"block3")
        # inputs = K.layers.MaxPooling2D(
        #     (2, 2), name="RegressModel_Pool3")(inputs)
        print("inputs1", inputs.shape)
        inputs = K.layers.Conv2D(512, (int(inputs.shape[1]), int(inputs.shape[2])),
                                 activation=K.activations.relu,
                                 name="RegressModel_Conv1", padding="valid",
                                 kernel_regularizer=K.regularizers.l2(0.05))(inputs)
        inputs = K.layers.Conv2D(8, (1, 1),
                                 activation=K.activations.linear,
                                 name="RegressModel_Conv2", padding="valid")(inputs)

        # inputs = K.layers.Flatten()(inputs)
        # inputs = K.layers.Dense(self.embedding_dim,
        #                         activation=K.activations.softmax)(inputs)
        # print("inputs1",inputs.shape)
        # inputs = K.layers.Dense(8,
        #                         activation=K.activations.linear)(inputs)
        print("inputs2",inputs.shape)
        inputs = K.layers.Lambda(lambda x: tf.reshape(x, (-1, 4, 2)))(inputs)
        print("inputs3", inputs.shape)
        output_value = inputs
        print("output_value", output_value.shape)
        model = K.Model(inputs=input_value, outputs=output_value, name="test")
        return model

    def layers_block(self,input_value,filters,name):
        inputs=input_value
        inputs2=input_value
        inputs=K.layers.Conv2D(filters, (1, 1),
                                 activation=K.activations.relu,
                                 name=name+"_Conv1", padding="same",
                                 kernel_regularizer=K.regularizers.l2(0.05))(inputs)
        # inputs = K.layers.normalization.BatchNormalization()(inputs)
        inputs2=K.layers.Conv2D(filters, (3, 3),
                                 activation=K.activations.relu,
                                 name=name+"_Conv2", padding="same",
                                 kernel_regularizer=K.regularizers.l2(0.05))(inputs2)
        # inputs2 = K.layers.normalization.BatchNormalization()(inputs2)
        inputs2=K.layers.Conv2D(filters, (1, 1),
                                 activation=K.activations.linear,
                                 name=name+"_Conv3", padding="same",
                                 kernel_regularizer=K.regularizers.l2(0.05))(inputs2)
        outputs=K.layers.Add()([inputs,inputs2])
        outputs = K.layers.normalization.BatchNormalization()(outputs)
        return outputs

    def layers_block2(self,input_value,filters,name):
        inputs=input_value
        # inputs=K.layers.Conv2D(filters, (1, 1),
        #                          activation=K.activations.relu,
        #                          name=name+"_Conv1", padding="same")(inputs)
        # inputs = K.layers.normalization.BatchNormalization()(inputs)
        inputs=K.layers.Conv2D(filters, (3, 3),
                                 activation=K.activations.relu,
                                 name=name+"_Conv2", padding="same")(inputs)
        inputs = K.layers.normalization.BatchNormalization()(inputs)
        # inputs=K.layers.Conv2D(filters, (1, 1),
        #                          activation=K.activations.relu,
        #                          name=name+"_Conv3", padding="same")(inputs)
        # inputs = K.layers.normalization.BatchNormalization()(inputs)
        outputs=inputs
        return outputs
