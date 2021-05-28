from PIL import Image
import numpy as np
import tensorflow as tf
import os
import cv2
import random
from tqdm import tqdm

# tfrecord结构
expected_features={
    'jpg':tf.io.FixedLenFeature((),tf.string,default_value=''),
    'label':tf.io.FixedLenFeature((),tf.string,default_value=''),
}
# 读取tfrecord
def parse_example(serialized_example):
    features = tf.io.parse_single_example(serialized_example, expected_features)
    jpg=tf.io.decode_jpeg(features['jpg'])
    retyped_images = tf.cast(jpg, tf.float32)
    images = tf.reshape(retyped_images, [256, 256, 3])
    float_image = tf.image.per_image_standardization(images)
    float_image.set_shape([256, 256, 3])

    label = tf.io.decode_png(features['label'])
    retyped_label = tf.cast(label, tf.int32)
    label = tf.reshape(retyped_label, [256, 256, 1])
    label.set_shape([256, 256, 1])
    return float_image,label
def tfrecords_reader_dataset(filenames,batch_size=32, n_parse_threads=5,shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename))
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_example,num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset
# 制作tfrecord
def make_record(jpg_dir="",label_dir="",train_record="",test_record="",slide=0.8):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    files=os.listdir(jpg_dir)
    random.shuffle(files)

    train_list=files[:int(len(files)*slide)]
    test_list=files[int(len(files)*slide):-1]

    # 制作训练集tfrecord
    with tf.io.TFRecordWriter(train_record) as writer:
        for file in tqdm(train_list):
            jpg_path=os.path.join(jpg_dir,file)
            label_path=os.path.join(label_dir,file.replace(".jpg",".png"))
            with tf.compat.v1.gfile.GFile(jpg_path, 'rb') as fid:
                jpg = fid.read()
            with tf.compat.v1.gfile.GFile(label_path, 'rb') as fid:
                label = fid.read()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': _bytes_feature(label),
                    'jpg': _bytes_feature(jpg),
                }))
            writer.write(example.SerializeToString())
    writer.close()

    # 制作测试集tfrecord
    with tf.io.TFRecordWriter(test_record) as writer:
        for file in tqdm(test_list):
            jpg_path = os.path.join(jpg_dir, file)
            label_path = os.path.join(label_dir, file.replace(".jpg", ".png"))
            with tf.compat.v1.gfile.GFile(jpg_path, 'rb') as fid:
                jpg = fid.read()
            with tf.compat.v1.gfile.GFile(label_path, 'rb') as fid:
                label = fid.read()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': _bytes_feature(label),
                    'jpg': _bytes_feature(jpg),
                }))
            writer.write(example.SerializeToString())
    writer.close()

    return
# 转换图像格式
class tif2jpg:
    def __init__(self,tif_dir=""):
        # 类型转换
        self.tif_dir=tif_dir
        self.Transform()

    def Transform(self):
        list=[]
        # 将tiff转换为jpg/png图像
        flag=0
        files=os.listdir(self.tif_dir)
        for file in files:
            if file.endswith(".TIF") and "y" not in file:
                # 路径声明
                tif_path=os.path.join(self.tif_dir,file)
                label_file=file[0:3]+"y"+file[3:]
                label_path=os.path.join(self.tif_dir,label_file)
                save_jpg=os.path.join("TrainData/jpg/",str(flag)+".jpg")
                save_label=os.path.join("TrainData/label/",str(flag)+".png")

                im=cv2.imread(tif_path,1)
                im2=cv2.imread(label_path,4)

                if np.max(im2) not in list:
                    list.append(np.max(im2))

                im2[im2 == 64] = 1
                im2[im2==128]=2
                im2[im2 == 192] = 3
                im2[im2 == 255] = 4

                # 筛选无意义图像
                if np.sum(im2)==0:
                    pass
                else:
                    cv2.imwrite(save_jpg, im)
                    cv2.imwrite(save_label, im2)
                    flag+=1

        return
if __name__ == '__main__':

    jpg_dir="TrainData/jpg/"
    label_dir="TrainData/label/"
    train_record="TrainData/train.tfrecord"
    test_record="TrainData/test.tfrecord"

    # 数据清洗整理
    # path = "D:/destbook/data/1/"
    # tif2jpg(path)

    # 制作tfrecord文件
    # make_record(jpg_dir,label_dir,train_record,test_record)