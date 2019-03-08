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
    with tf.Session() as sess:
        # 常量
        a=tf.constant(((2.0,8.0),(3.0,9.0)))
        print(a.eval())
        # 常量
        b=tf.constant(((1.0,2.0),(3.0,4.0)))
        print(b.eval())
        # 常量
        c=tf.constant(((9.0,1.0),(2.0,9.0)))
        print(b.eval())
        # # 相减
        # s=tf.math.subtract(a,b)
        # print(s.eval())
        # # 求和
        # loss=tf.reduce_mean(s)
        # print(loss.eval())
        # 锚点
        anchor=a
        # 正样本
        positive=b
        # 负样本
        negative=c
        # 计算锚点与正样本的距离
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        print("pos_dist",pos_dist.eval())
        # 计算锚点与负样本的距离
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        print("neg_dist",neg_dist.eval())
        # 正距离减负距离为损失值（训练时basic_loss会尽量逼近0，最终两个距离相等，还不符合要求,要加一个数，使结果偏向正）
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), 0.2)
        print("basic_loss",basic_loss.eval())
        # 取消小于0的loss,让结果向正方向收敛
        # loss = tf.reduce_mean(tf.maximum(basic_loss,0), 0)
        loss = tf.maximum(basic_loss,0)
        print("loss",loss.eval())

    # =====================

    # a=cv2.imread('E:/Labels/yibiao/140a4f4536923f24d736e88b2fd2ed7f.jpg')
    # h,w = a.shape[:2]
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ])
    # pts1 = np.float32(pts / ((w-1,h-1),(w-1,h-1),(w-1,h-1),(w-1,h-1)))
    # print("pts",pts)
    # print("pts1",pts1)

    # =====================

    # with tf.Session() as sess:
    #     print(tf.greater(4,3).eval())
    #     print(tf.greater(3,3).eval())
    #     print(tf.greater(2,3).eval())

    # =====================

    # cap = cv2.VideoCapture(0)
    # # 打开摄像头，若打开本地视频，同opencv一样，只需将０换成("×××.avi")
    # while(1):    # get a frame   
    #     ret, frame = cap.read()    # show a frame   
    #     cv2.imshow("capture", frame)   
    #     if cv2.waitKey(1) & 0xFF == ord('q'):        
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

    # =====================
    
    # a = tf.constant([[1,2],[3,4]]) # shape (2,2)
    # ab0 = tf.pad(a, [(0,1),(2,3)])

    # with tf.Session() as sess:
    #     print("a", sess.run(a))
    #     print("0", sess.run(ab0))

    # =====================

    # 这里ab0和ab1都实现找下标的方法
    # a = tf.constant([1,2,3,4])
    # b = tf.constant([1,2])
    # c = tf.constant([1,0,0,1])
    # ab0 = tf.gather(a, b)
    # partitionsabs = tf.reduce_sum(tf.one_hot(b, tf.shape(a)[0], dtype='int32'), 0)
    # ab1 = tf.dynamic_partition(a, partitionsabs, 2)[1]

    # with tf.Session() as sess:
    #     print("a", sess.run(a))
    #     print("b", sess.run(b))
    #     print("0", sess.run(ab0))
    #     print("1", sess.run(ab1))

    # =====================

    # x = tf.minimum(6000, 4092)
    # print("x", x)
    # print("x", x.shape)

    # with tf.Session() as sess:
    #     y1=sess.run(x)
    #     print("1", y1)

    # =====================

    # input_image = K.layers.Input(
    #         shape=[None, None, 3], name="input_image")
    # # 一层
    # a = K.layers.Lambda(lambda x: tf.Variable((1,2,3)), name="anchors")(input_image)
    # # 一变量
    # anchors = K.backend.variable((1,2,3), name="anchors")
    # print(type(a))
    # print(type(anchors))

    # =====================

    # a = [1,5,3]
    
    # f1 = tf.maximum(a, 3)
    # f2 = tf.minimum(a, 3)
    # f3 = tf.argmax(a, 0)
    # f4 = tf.argmin(a, 0)
    
    # with tf.Session() as sess:
    #     print(sess.run(f1))#print f1.eval()
    #     print(sess.run(f2))
    #     print(sess.run(f3))
    #     print(sess.run(f4))


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
    #     # keepdims保持矩阵结构，长度为1
    #     y2=sess.run(tf.reduce_mean(x, keepdims=True))
    #     print("2", y2)
    #     print("2", y2.shape)


testCode()
