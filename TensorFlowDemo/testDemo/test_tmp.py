import keras as K
import tensorflow as tf
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt


def randomHSV():
    """HSV颜色变换测试"""
    a = np.zeros([10, 10, 3])
    a[0, 0, 2] = 1
    b = a[:, :, -1]
    print(b)
    print(b.shape)
    img = K.preprocessing.image.load_img(
        'D:/document/Share/labels/0-9/test/2.png', target_size=(60, 60))
    img = K.preprocessing.image.img_to_array(img, data_format="channels_last")
    img = img.astype('float32')
    img /= 255.0
    # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 通道拆分
    (h, s, v) = cv2.split(hsv)
    h = np.random.uniform(0, 360, size=(hsv.shape[:2])).astype(np.float32)
    print("h", type(h[0][0]))
    print("s", type(s[0][0]))
    print("v", type(v[0][0]))
    # 合并通道
    hsv = cv2.merge([h, s, v])
    print("hsv", hsv.shape)
    img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
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


def testCode():
    
    a = [1,5,3]
    
    f1 = tf.maximum(a, 3)
    f2 = tf.minimum(a, 3)
    f3 = tf.argmax(a, 0)
    f4 = tf.argmin(a, 0)
    
    with tf.Session() as sess:
        print(sess.run(f1))#print f1.eval()
        print(sess.run(f2))
        print(sess.run(f3))
        print(sess.run(f4))


    # =====================

    # loss = [None] * 10
    # print("loss", loss)

    # =====================

    # x = tf.constant([[1., 1.], [2., 2.]])
    # print("x", x)
    # print("x", x.shape)

    # with tf.Session() as sess:
    #     y1=sess.run(tf.reduce_mean(x))
    #     print("1", y1)
    #     y2=sess.run(tf.reduce_mean(x, keepdims=True))
    #     print("2", y2)
    #     print("2", y2.shape)


testCode()
