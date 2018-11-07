import keras as K
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse


class MyMaskRCNN():

    def __init__(self, label_path):
        self.label_path = label_path

    def creade_model():
        """创建模型"""
        # 输入图片(宽,高,通道数)
        input_img = K.Input((None, None, 3), name="input_image")
        # 输入图片信息，用来标记实际图片大小、缩放比例等
        input_img = K.Input((None)), name = "input_image_meta")
        # RPN 前景：1，背景：0 （多个）
        input_rpn_match=KL.Input(
            shape = [None, 1], name = "input_rpn_match", dtype = tf.int32)
        # RPN 框偏移量 （多个）
        input_rpn_bbox=KL.Input(
            shape = [None, 4], name = "input_rpn_bbox", dtype = tf.float32)

        # Detection GT (class IDs, bounding boxes, and masks)
        # 1. GT Class IDs (zero padded)
        # 实际分类，数量与框数量相等，里面对应分类ID
        input_gt_class_ids=KL.Input(
            shape = [None], name = "input_gt_class_ids", dtype = tf.int32)
        # 2. GT Boxes in pixels (zero padded)
        # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        # 实际图片框，x1,y1,x2,y2，左上角右下角坐标
        input_gt_boxes=KL.Input(
            shape = [None, 4], name = "input_gt_boxes", dtype = tf.float32)
        # Normalize coordinates
        # 实际图片框坐标归一化，根据输入图片大小
        gt_boxes=KL.Lambda(lambda x: norm_boxes_graph(
            x, K.shape(input_image)[1:3]))(input_gt_boxes)
        # 3. GT Masks (zero padded)
        # [batch, height, width, MAX_GT_INSTANCES]
        # mask多个，数量与框数量相等，用56*56大小，最终缩放成原图大小
        input_gt_masks = KL.Input(
            shape = [56, 56, None],
            name = "input_gt_masks", dtype = bool)
        
        # =================================
        # 创建特征提取层(只是Demo,实际结构比该结构复杂，原作者用resnet101)
        # C1
        x = K.layers.Conv2D(32, (7, 7), strides=(4, 4),
                            activation=K.backend.relu,
                            name="fpn_conv1", padding="same")(input_value)
        x = K.layers.BatchNormalization(name="fpn_bn1")(x)
        # 池化，大小变成原图1/2
        x = K.layers.MaxPool2D((3, 3), strides=(2, 2), name="fpn_pool1")(x)
        C1 = x

        # C2
        x = K.layers.Conv2D(50, (5, 5), strides=(2, 2),
                            activation=K.backend.relu,
                            name="fpn_conv2", padding="same")(x)
        x = K.layers.BatchNormalization(name="fpn_bn2")(x)
        # 池化，大小变成原图1/4
        x = K.layers.MaxPool2D((3, 3), strides=(2, 2), name="fpn_pool2")(x)
        C2 = x

        # C3
        x = K.layers.Conv2D(80, (3, 3),
                            activation=K.backend.relu,
                            name="fpn_conv3", padding="same")(x)
        x = K.layers.BatchNormalization(name="fpn_bn3")(x)
        # 池化，大小变成原图1/8
        x = K.layers.MaxPool2D((3, 3), strides=(2, 2), name="fpn_pool3")(x)
        C3 = x

        # C4
        x = K.layers.Conv2D(120, (3, 3),
                            activation=K.backend.relu,
                            name="fpn_conv4", padding="same")(x)
        x = K.layers.BatchNormalization(name="fpn_bn4")(x)
        # 池化，大小变成原图1/16
        x = K.layers.MaxPool2D((3, 3), strides=(2, 2), name="fpn_pool4")(x)
        C4 = x

        # C5
        x = K.layers.Conv2D(200, (3, 3),
                            activation=K.backend.relu,
                            name="fpn_conv5", padding="same")(x)
        x = K.layers.BatchNormalization(name="fpn_bn5")(x)
        # 池化，大小变成原图1/32
        x = K.layers.MaxPool2D((3, 3), strides=(2, 2), name="fpn_pool5")(x)
        C5 = x

        # 通过C1、C2、C3、C4、C5合成特征金字塔
        P5 = K.layers.Conv2D(256, (1, 1),
                            name="fpn_p5_c5", padding="same")(C5)
        P4 = K.layers.Add(name="fpn_p4_c4_add")([
            K.layers.UpSampling2D(size=(2, 2),
                            name="fpn_p5_upsampling")(P5),
            K.layers.Conv2D(256, (1, 1),
                            name="fpn_p4_c4", padding="same")(C4)
            ])
        P3 = K.layers.Add(name="fpn_p3_c3_add")([
            K.layers.UpSampling2D(size=(2, 2),
                            name="fpn_p4_upsampling")(P4),
            K.layers.Conv2D(256, (1, 1),
                            name="fpn_p3_c3", padding="same")(C3)
            ])
        P2 = K.layers.Add(name="fpn_p2_c2_add")([
            K.layers.UpSampling2D(size=(2, 2),
                            name="fpn_p3_upsampling")(P3),
            K.layers.Conv2D(256, (1, 1),
                            name="fpn_p2_c2", padding="same")(C2)
            ])
        # 合成的特征做一个(3*3)卷积
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3),
                       padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3),
                       padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3),
                       padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3),
                       padding="SAME", name="fpn_p5")(P5)
        # P6由P5 最大池化下取样获得
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
        # =================================

        # RPN网络
        

# 读取启动参数
parser=argparse.ArgumentParser()
parser.add_argument('label_path')
parser.add_argument('-t', '--train', default = "1")
args=parser.parse_args()

# 数据集路径
label_path=args.label_path
