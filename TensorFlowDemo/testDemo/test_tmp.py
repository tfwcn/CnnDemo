import keras as k
import tensorflow as tf
import numpy as np


a = [1, 2, 3]
b = [4, 5, 6, 6.5]
c = [7, 8, 9]
# np.meshgrid 按x,y维度，分别扩充scales,ratios，结果维度(5,3)
a1, b1 = np.meshgrid(np.array(a), np.array(b))
print("a1", a1.shape)
print("a1", a1)
print("b1", b1.shape)
print("b1", b1)
# a1先由二维转一维
c1, a2 = np.meshgrid(np.array(c), np.array(a1))
# print("c1", c1.shape)
# print("c1", c1)
print("a2", a2.shape)
print("a2", a2)
# b1先由二维转一维
c2, b2 = np.meshgrid(np.array(c), np.array(b1))
# print("c2", c2.shape)
# print("c2", c2)
print("b2", b2.shape)
print("b2", b2)
box_centers = np.stack([b2, a2], axis=2)#np.stack后面加一个维度，数量为2，b2与a2对应维度组合
box_sizes = np.stack([c2, c1], axis=2)
print("box_centers", box_centers.shape)
print("box_centers", box_centers)
# print("box_sizes", box_sizes.shape)
# print("box_sizes", box_sizes)
# box_centers = box_centers.reshape([-1, 2])
# box_sizes = box_centers.reshape([-1, 2])
# print("box_centers", box_centers.shape)
# print("box_centers", box_centers)
# print("box_sizes", box_sizes.shape)
# print("box_sizes", box_sizes)
boxes = np.concatenate([box_centers - 0.5 * box_sizes,box_centers + 0.5 * box_sizes], axis=1)
print("box_sizes", box_sizes.shape)
print("box_sizes", box_sizes)
