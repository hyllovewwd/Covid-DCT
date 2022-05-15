import tensorflow as tf
import numpy as np
import shutil
import os
from model1_net import multi_net
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = (224, 224)


def handle_path():
    """ 得到文件的路径"""
    path_patient = []
    path_header = 'D:/项目/queue2/oCT'
    with open('oCT_1515', 'r') as f:
        list1 = f.readlines()
        for p in list1:
            p = eval(p)
            path_patient.append(p)
    back_path = []
    for p in path_patient:
        p_0 = p[0]
        p_length = len(p)
        header = p_0.split("/")
        no_p = header[4].split("_")[0]
        path_p = []
        for n in range(0, p_length):
            path_last = str(no_p) + "_" + str(n + 1) + ".jpg"
            path = path_header + "/" + path_last
            path_p.append(path)
        back_path.append(path_p)
    print(len(back_path))
    return back_path


def load_weight_model():
    model_=multi_net()
    checkpoint_path = "model1/model1/cp.ckpt"
    model_.load_weights(checkpoint_path)
    return model_


def get_result(model_, path_list):
    """  n:1 p:2 Ni:0"""
    estimate_list = []  # 模型预测是p的概率列表 ：预测值列表
    str_path = str(path_list[0])
    no = str_path.split("/")[-1].split("_")[0]  # 得到病人的编号
    for k in path_list:
        img = tf.keras.preprocessing.image.load_img(k, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = model_.predict(img_array)
        score = list(predictions[0])[2]  # 预测p的值
        estimate_list.append(score)  # 预测值列表
    arr = np.array(estimate_list)
    index_path = np.argsort(-arr)[:10]  # 取最大的10个对应的索引
    index_path = list(index_path)
    path_select = [path_list[j] for j in index_path]
    get_image(path_select, no)


def get_image(image_paths, patient_no):
    """ 复制图片到指定路径"""
    target_path_header = "D:/项目/combine_image/oCT"
    path_dir = target_path_header + "/" + str(patient_no)
    os.mkdir(path_dir)
    number_ = 0
    for image_path in image_paths:
        number_ += 1
        target_path = path_dir + "/" + str(patient_no) + "_" + str(number_) + ".jpg"
        shutil.copy(image_path, target_path)
    print("{}号病人图片选择完毕".format(patient_no))


if __name__ == '__main__':
    path_sum = handle_path()
    print(path_sum)
    model = load_weight_model()
    number = 0
    for i in path_sum:
        get_result(model, i)

