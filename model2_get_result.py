import os
from model2_net import multi_net as model2
import tensorflow as tf
import numpy as np
from collections import Counter

IMG_SIZE = (224, 224)
num = 0


def get_result_model2(pa, no, model):
    result_list = []  # 模型预测是p的概率列表 ：预测值列表
    for k in pa:
        img = tf.keras.preprocessing.image.load_img(k, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = model.predict(img_array)
        label = int(np.argmax(predictions))
        result_list.append(label)

    # 结果的判断
    print(no, "病人的结果", result_list)
    data_nums = Counter(result_list)
    label_count = [int(data_nums[0]), int(data_nums[1]), int(data_nums[2])]
    label = label_count.index(max(label_count))
    return label


# nCT 49   [0, 0, 0, 1, 2, 1, 0, 0, 2, 0, 0, 1, 2, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 2, 0, 2, 0, 1, 1, 0, 1, 1, 1, 0, 2, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 2, 1, 1, 2, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 2, 2, 1, 0, 2, 0, 1, 1, 2, 1, 0, 0, 1, 0, 1]
# oCT 144  [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 2, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 0, 2, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1]
# tCT 11   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1]

if __name__ == '__main__':
    num = 0
    path = "D:/项目/combine_image/nCT"  # 存放病人的CT图片的文件夹
    path_list = os.listdir(path)
    checkpoint_path = "model2/model2/cp.ckpt"
    model = model2()  # 加载模型
    model.load_weights(checkpoint_path)
    label=[]
    for i in path_list:
        path_ = path + "/" + i
        path_list_ = os.listdir(path_)
        path_li = [path_ + "/" + i for i in path_list_]
        la = get_result_model2(path_li, i, model)
        label.append(la)
        print(la)
    print(label)

