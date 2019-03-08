import numpy as np
import cv2

class ImageHelper():
    def resize_padding_zero(self,img,width,height):
        """等比例缩放图片"""
        (h,w) = img.shape[:2]
        s1 = w / h
        s2 = width / height
        new_img=img
        # print("原图大小",w,h)
        if s1>s2:
            new_w=width
            new_h=int(width/s1)
            pad_l=int((width-new_w)/2)
            pad_r=int(width-new_w-pad_l)
            pad_t=int((height-new_h)/2)
            pad_b=int(height-new_h-pad_t)
            new_img=cv2.resize(img,(new_w,new_h))
            new_img=np.pad(new_img,((pad_t,pad_b),(pad_l,pad_r),(0,0)),"constant",constant_values = (0,0))
        else:
            new_w=int(height*s1)
            new_h=height
            pad_l=int((width-new_w)/2)
            pad_r=int(width-new_w-pad_l)
            pad_t=int((height-new_h)/2)
            pad_b=int(height-new_h-pad_t)
            new_img=cv2.resize(img,(new_w,new_h))
            new_img=np.pad(new_img,((pad_t,pad_b),(pad_l,pad_r),(0,0)),"constant",constant_values = (0,0))
        # print("new_img",new_img.shape)
        return new_img

        
if __name__ == '__main__':
    import keras as K
    import matplotlib.pyplot as plt

    x = K.preprocessing.image.load_img(
        "E:/Labels/yibiao/一号主变保护测控盘A套4n主变本体保护装置 2018-11-26 102653_效果图.jpg")
    x = K.preprocessing.image.img_to_array(
        x, data_format="channels_last")
    x = x.astype('float32')
    x /= 255.0
    imageHelper = ImageHelper()
    new_img = imageHelper.resize_padding_zero(x,512,512)
    plt.imshow(x)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    plt.imshow(new_img)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
