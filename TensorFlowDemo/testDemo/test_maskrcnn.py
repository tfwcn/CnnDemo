import keras as K
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

class MyMaskRCNN():

    def __init__(self,label_path):
        self.label_path = label_path

    def creade_model():
        """创建模型"""
        


# 读取启动参数
parser = argparse.ArgumentParser()
parser.add_argument('label_path')
parser.add_argument('-t', '--train', default="1")
args = parser.parse_args()

# 数据集路径
label_path = args.label_path
