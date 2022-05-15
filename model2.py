import pathlib
import random
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model2_net import multi_net
from sklearn.metrics import roc_auc_score


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = (224, 224)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    return image


def load_and_preprocess_image(path_):
    image = tf.io.read_file(path_)
    return preprocess_image(image)


def construct_batch(image_paths, image_labels):
    """
    构建总的测试集
    :return:
    """
    image_paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_paths_ds = image_paths_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    image_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(image_labels, tf.int32))
    path_label = tf.data.Dataset.zip((image_paths_ds, image_labels_ds))
    data_batch = path_label.batch(BATCH_SIZE)
    dataset = data_batch.prefetch(buffer_size=AUTOTUNE)
    return dataset


def load_image_label(paths):
    data_root = pathlib.Path(paths)
    all_image_paths_ = list(data_root.glob('*/*'))
    all_image_paths_ = [str(path_) for path_ in all_image_paths_]  # 所有的图片路径
    random.shuffle(all_image_paths_)  # 随机打乱这些数据名
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index_) for index_, name in enumerate(label_names))
    all_image_labels_ = [label_to_index[pathlib.Path(path_).parent.name]
                         for path_ in all_image_paths_]  # 所有的标签
    return all_image_paths_, all_image_labels_


if __name__ == '__main__':
    train_path_ = r"D:\项目\queue1"
    test_path_ = r"D:\项目\queue1_test"
    all_image_paths, all_image_labels = load_image_label(train_path_)  # 图片已经经过打乱
    all_image_paths_test, all_image_labels_test = load_image_label(test_path_)  # 图片已经经过打乱
    model = multi_net()

    train_dataset = construct_batch(all_image_paths, all_image_labels)
    test_dataset=construct_batch(all_image_paths_test,all_image_labels_test)
    checkpoint_path = "model2/model2" + "/" + "cp.ckpt"
    model.load_weights(checkpoint_path)
    # callback_list = [
    #     tf.keras.callbacks.EarlyStopping(
    #         monitor='accuracy',  # 监控的指标，精度
    #         patience=5  # 精度在多于五轮未改善，即停止训练
    #     ),
    #     tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                        save_best_only=True,
    #                                        save_weights_only=True,
    #                                        verbose=1,
    #                                        monitor='accuracy'
    #                                        )
    # ]
    # initial_epochs = 50
    # history = model.fit(train_dataset,
    #                     epochs=initial_epochs,
    #                     callbacks=callback_list)
    # acc = history.history['accuracy']
    # print("acc:", acc)
    # model.load_weights(checkpoint_path)  # 加载保存的最好权重
    scores = model.predict(test_dataset)
    scores_pri_t = [i.tolist() for i in scores]  # 预测到的分数
    label_pri_t = []
    for i in scores:
        """ 得到预测值对应的标签"""
        label_pri_t.append(np.argmax(i))
    print(classification_report(all_image_labels_test, label_pri_t))
    print(confusion_matrix(all_image_labels_test, label_pri_t))
    print(roc_auc_score(all_image_labels_test, scores_pri_t, multi_class='ovo'))
    print(roc_auc_score(all_image_labels_test, scores_pri_t, multi_class='ovr'))
