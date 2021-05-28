import os
from Efficiunet import UEfficientNet
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image

def Use(jpg_path,save_path,model_path):
    # 项目部署
    model = UEfficientNet()
    model.load_weights(model_path)
    ans = Compute_ans(jpg_path, model)
    im = Image.fromarray((ans * 64 - 1).astype("uint8"))
    im.save(save_path)
    return

def standard(data):
    """
    图像标准化
    :param data: 输入
    :return: 输出归一化数据
    """
    out_data = (data - np.mean(data)) / max(1 / 256, np.std(data))
    return out_data

def Compute_ans(pic_path, model):
    # 卷积网络计算
    jpg = Image.open(pic_path)
    jpg_data = standard(jpg)

    jpg_data = np.reshape(jpg_data, (1, 256, 256, 3))

    prediction = model.predict(jpg_data)
    ans = np.argmax(prediction, axis=-1)
    return np.reshape(ans, (256, 256))


if __name__ == '__main__':
    jpg_path=input("please input jpg path end with enter:")
    save_path=input("please input save path end with enter:")
    Use(jpg_path,save_path,model_path="Model/model.h5")