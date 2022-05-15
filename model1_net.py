import tensorflow as tf
from attention import se_block, cbam_block, eca_block


def multi_net():
    tf_se = tf.keras.Sequential()
    resnet_input = tf.keras.applications.resnet.preprocess_input
    vgg_input = tf.keras.applications.vgg16.preprocess_input
    mobilenet_input = tf.keras.applications.mobilenet.preprocess_input
    IMG_SHAPE = (224, 224, 3)
    model_resnet = tf.keras.applications.ResNet152(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    model_vgg = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')
    model_mobilenet = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                                      include_top=False,
                                                      weights='imagenet')
    model_resnet.trainable = True
    fine_tune_at = 505
    for layer in model_resnet.layers[:fine_tune_at]:
        layer.trainable = False
    model_vgg.trainable = True
    fine_tune_at = 15
    for layer in model_vgg.layers[:fine_tune_at]:
        layer.trainable = False
    model_mobilenet.trainable = True
    fine_tune_at = 80
    for layer in model_mobilenet.layers[:fine_tune_at]:
        layer.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(3, activation='softmax')
    inputs_1 = tf.keras.Input(shape=(224, 224, 3))
    x = tf_se(inputs_1)
    x1 = resnet_input(x)
    x1 = model_resnet(x1, training=False)
    x1 = se_block(x1)  # 加入注意力
    x2 = vgg_input(x)
    x2 = model_vgg(x2, training=False)
    x2 = cbam_block(x2)
    x3 = mobilenet_input(x)
    x3 = model_mobilenet(x3, training=False)
    x3 = eca_block(x3)
    multi_input = tf.keras.layers.concatenate([x1, x2, x3])  # 模型融合
    multi_input = global_average_layer(multi_input)
    x = tf.keras.layers.Dense(32, activation='relu')(multi_input)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    models = tf.keras.Model(inputs=inputs_1, outputs=outputs)
    models.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                   metrics=['accuracy'])
    return models


