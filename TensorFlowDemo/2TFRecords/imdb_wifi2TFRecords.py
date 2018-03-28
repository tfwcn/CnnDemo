import tensorflow as tf
import os
import scipy.io as sio
import numpy as np
import math
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'e:/imdb_crop',
                           """Path to the MNIST data directory.""")

def readFiles():
    '''读取文件'''
    #读取地址和性别
    dataset = sio.loadmat(FLAGS.data_dir + "/imdb.mat")
    filepath = dataset["imdb"][0,0]["full_path"][0]
    gender = dataset["imdb"][0,0]["gender"][0]
    #整理
    label = [ os.path.join(FLAGS.data_dir, filepath[i][0]) for i in range(0, len(filepath)) ]
    image = [ gender[i] for i in range(0, len(gender)) ]
    return label,image

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(images, labels, name):
  """Converts a dataset to tfrecords.保存路径与性别"""
  #num_examples = data_set.num_examples

  #if images.shape[0] != num_examples:
  #  raise ValueError('Images size %d does not match label size %d.' %
  #                   (images.shape[0], num_examples))


  filename = os.path.join(FLAGS.data_dir, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(len(images)):
    if math.isnan(labels[index]):
        continue
    #image = Image.open(images[index])#打开图片
    #(im_width, im_height) = image.size
    #image_raw = np.array(image.getdata())
    #depth = image_raw.size//(im_width*im_height)#颜色通道
    #image_raw = image_raw.reshape(
    #  (im_height, im_width, depth)).astype(np.uint8)
    #image_raw = image_raw.tobytes()
    #example = tf.train.Example(features=tf.train.Features(feature={
    #    'height': _int64_feature(im_height),
    #    'width': _int64_feature(im_width),
    #    'depth': _int64_feature(depth),
    #    'label': _int64_feature(int(labels[index])),
    #    'image_raw': _bytes_feature(image_raw)}))
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(int(labels[index])),
        'image_path': _bytes_feature(images[index].encode())}))
    writer.write(example.SerializeToString())
  writer.close()

def main(argv=None):  #运行
    label, image = readFiles()
    convert_to(label, image, "imdb")

if __name__ == '__main__':
    tf.app.run()