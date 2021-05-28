首先将TIF文件转换为jpg/png同时将其标注为同一flag文件名
    调用MadeRecord下的tif2jpg类
    调用MadeRecord下的make_record函数制作tfrecord
搭建网络 Efficinet注意力机制网络 + Uet++编码器（包含Restnet残差模块）
    调用Efficiunet下的UEfficientNet函数返回网络结构
训练采用Adam优化器+交叉熵损失函数+余弦退火学习率+交并比IOU验证