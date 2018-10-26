import keras as K
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

"""ROIAlign，把不同大小的图片对齐到固定大小"""

parser = argparse.ArgumentParser()
parser.add_argument('file_path')
args = parser.parse_args()

file_path = args.file_path

img = K.preprocessing.image.load_img(file_path)
img = K.preprocessing.image.img_to_array(img)
img = img.astype(np.float)
img /= 255

plt.imshow(img)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()

width = img.shape[0]
height = img.shape[1]
new_width = 50
new_height = 50
subcell = (2, 2)
new_img = np.zeros((new_width*subcell[0], new_height*subcell[1], 3))
print("img.shape", img.shape)
for x in range(new_width*subcell[0]):
    for y in range(new_height*subcell[1]):
        # 真实坐标点
        x2 = x / (new_width*subcell[0]) * width
        y2 = y / (new_height*subcell[1]) * height
        # print("x2,y2", x2, y2)

        c1 = img[math.floor(x2), math.floor(y2), :]  # 左上
        c2 = img[math.ceil(x2), math.floor(y2), :]  # 右上
        c3 = img[math.floor(x2), math.ceil(y2), :]  # 左下
        c4 = img[math.ceil(x2), math.ceil(y2), :]  # 右下
        # print("c1,c2,c3,c4", c1, c2, c3, c4)

        # 双线性插值
        cx1 = c1 + (c2-c1)*(x2 % 1)
        cx2 = c3 + (c4-c3)*(x2 % 1)
        c = cx1 + (cx2-cx1)*(y2 % 1)
        # print("cx1, cx2, c", cx1, cx2, c)
        new_img[x, y] = c

print("new_img.shape", new_img.shape)
plt.imshow(new_img)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()
