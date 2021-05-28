import os
from Efficiunet import UEfficientNet
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image


def Model_acc(pic_dir="",label_dir="",save_dir="",model_path=""):
    model = UEfficientNet()
    model.load_weights(model_path)
    true=0
    flag=0
    files=os.listdir(pic_dir)
    for file in tqdm(files):
        flag+=1
        jpg_path=os.path.join(pic_dir,file)
        label_path=os.path.join(label_dir,file.replace(".jpg",".png"))
        save_path=os.path.join(save_dir,file.replace(".jpg",".png"))

        ans=Compute_ans(jpg_path,model)
        # ans=np.array(Image.open(save_path))
        label=np.array(Image.open(label_path))
        true+=Compare_pic(ans,label)

        im=Image.fromarray((ans*64-1).astype("uint8"))
        im.save(save_path)

    return true/flag*100

def Compare_pic(ans,label):
    """
    计算单张图像正确率
    :param ans: 输出矩阵
    :param label: 标注图像矩阵
    :return: 正确率
    """
    true=np.where(ans==label)
    return len(true[0])/256/256

def standard(data):
    """
    图像标准化
    :param data: 输入
    :return: 输出归一化数据
    """
    out_data = (data - np.mean(data)) / max(1 / 256, np.std(data))
    return out_data

def Compute_ans(pic_path,model):
    jpg=Image.open(pic_path)
    jpg_data=standard(jpg)

    jpg_data=np.reshape(jpg_data,(1,256,256,3))

    prediction = model.predict(jpg_data)
    ans = np.argmax(prediction, axis=-1)
    return np.reshape(ans,(256,256))


if __name__ == '__main__':
    pic_dir="TrainData/jpg/"
    label_dir="TrainData/label/"
    save_dir="TrainData/save/"
    model_path="Model/model.h5"
    u=Model_acc(pic_dir,label_dir,save_dir,model_path)