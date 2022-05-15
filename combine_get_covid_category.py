import os
from model1_net import multi_net as model1
from model2_net import multi_net as model2
import tensorflow as tf
import numpy as np

IMG_SIZE = (224, 224)


def get_result_model2(pa, no):
    checkpoint_path = "model2/model2/cp.ckpt"
    model = model2()  # 加载模型
    model.load_weights(checkpoint_path)
    result_list = []  # 模型预测是p的概率列表 ：预测值列表
    for k in pa:
        img = tf.keras.preprocessing.image.load_img(k, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = model.predict(img_array)
        label = np.argmax(predictions)
        result_list.append(label)
    cal_result(result_list, no)


def cal_result(re_list, no):
    """
    最终结果判断
    :param re_list:
    :param no:
    :return:
    """
    pass


def get_path_model1(pa):
    checkpoint_path = "model1/model1/cp.ckpt"
    model = model1()  # 加载模型
    model.load_weights(checkpoint_path)
    estimate_list = []  # 模型预测是p的概率列表 ：预测值列表
    str_path = str(pa[0])
    no = str_path.split("/")[-1].split("_")[0]  # 得到病人的编号
    for k in pa:
        img = tf.keras.preprocessing.image.load_img(k, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = model.predict(img_array)
        score = list(predictions[0])[2]  # 预测p的值
        estimate_list.append(score)  # 预测值列表
    tf.keras.backend.clear_session()
    arr = np.array(estimate_list)
    index_path = np.argsort(-arr)[:10]  # 取最大的10个对应的索引
    index_path = list(index_path)
    path_select = [pa[j] for j in index_path]
    get_result_model2(path_select, no)


if __name__ == '__main__':
    path = ""  # 存放病人的CT图片的文件夹
    path_list = os.listdir(path)
    path_list = [path + "/" + i for i in path_list]
    get_path_model1(path_list)
