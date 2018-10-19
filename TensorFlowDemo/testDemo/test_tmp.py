import keras as K
import tensorflow as tf
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt



img = K.preprocessing.image.load_img('D:/document/Share/labels/0-9/test/2.png', target_size=(60, 60))
img = K.preprocessing.image.img_to_array(img, data_format="channels_last")
img = img.astype('float32')
img /= 255.0
# img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
# 通道拆分
(h, s, v) = cv2.split(hsv)
h = np.random.uniform(0,360,size=(hsv.shape[:2])).astype(np.float32)
print("h",type(h[0][0]))
print("s",type(s[0][0]))
print("v",type(v[0][0]))
# 合并通道
hsv = cv2.merge([h,s,v])
print("hsv",hsv.shape)
img2 = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
# print("img2:",img2)
# print("h:",h)
# print("s:",s)
# print("v:",v)
plt.imshow(img)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()
plt.imshow(hsv)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()
plt.imshow(img2)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()