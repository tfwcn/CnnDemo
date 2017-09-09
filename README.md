# CnnDemo

## Cnn卷积神经网络

> 神经网络训练识别手写字体

* 主分支：传统卷积神经网络
* DCnn分支：可变卷积神经网络

### 参照资料：
1. 《Notes on Convolutional Neural Networks》
    * 原文：http://cogprints.org/5869/1/cnn_tutorial.pdf
    * 中文：http://www.cnblogs.com/shouhuxianjian/p/4529202.html

### 环境依赖：
1. python 3.5
	``` bash
	# For CPU
	pip install tensorflow
	# For GPU
	pip install tensorflow-gpu
	```
	``` bash
	pip install pillow
	pip install lxml
	pip install jupyter
	pip install matplotlib
	```

	Python35\Lib\site-packages 添加库引用文件 tensorflow.pth，内容：
	``` python
	[git源码]\tensorflow
	[git源码]\tensorflowModels
	```

2. protoc
	下载地址：https://github.com/google/protobuf/releases
	protoc.exe 复制到 C:\Windows
	``` bash
	# From tensorflow/models/
	protoc object_detection/protos/*.proto --python_out=.
	```
	
3. GPU环境
	* CUDA 8
	* CUDNN 6 for CUDA 8