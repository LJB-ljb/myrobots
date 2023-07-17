import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPool2D
from keras.callbacks import ModelCheckpoint
from keras import regularizers, optimizers
import os
import os.path as osp
import cv2 as cv
import numpy as np
import  matplotlib.pyplot as plt
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

  
def get_file_path_by_name(file_dir):
    """get file path by name"""

    imgs = []
    results = []
    labels = os.listdir(file_dir)
    for label in labels:
        for root, dirs, files in os.walk(osp.join(file_dir, label)):  # 获取所有文件
            for file in files:  # 遍历所有文件名
                # print(file)
                if os.path.splitext(file)[1] == '.jpg':  
                    imgs.append(os.path.join(root, file))  # 拼接处绝对路径并放入列表
                    results.append(label)
    # print('总文件数目：', len(imgs))
    return imgs, results, label



data = ImageDataGenerator(data_format=keras.backend.image_data_format())
train_generator = data.flow_from_directory("./training_data",
                                            target_size=(64, 64),
                                            color_mode='rgb',
                                            batch_size=128)

test_generator = data.flow_from_directory("./testing_data",
                                            target_size=(64, 64),
                                            color_mode='rgb',
                                            batch_size=128)



base_hidden_units = 32  # 隐藏单元数量
weight_decay = 1e-4     # L2正则化超参数

# 使用序列模型
model = Sequential()
# CONV1
model.add(Conv2D(base_hidden_units, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
# CONV2
model.add(Conv2D(base_hidden_units, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
# POOL+Dropout
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
# CONV3
model.add(Conv2D(base_hidden_units*2, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
# CONV4
model.add(Conv2D(base_hidden_units*2, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
# POOL+Dropout
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
# CONV5
model.add(Conv2D(base_hidden_units*4, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
# CONV6
model.add(Conv2D(base_hidden_units*4, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
# POOL+Dropout
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
# FC7
model.add(Flatten())
model.add(Dense(4, activation='softmax'))
model.summary()


batch_size = 128    # 批处理大小
epochs = 20        # 训练轮数

# 保存最佳权重
checkpointer = ModelCheckpoint(filepath='model.100epochs.hdf5', verbose=1, save_best_only=True)
optimizers = keras.optimizers.adam(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy' ])


history = model.fit_generator(train_generator, callbacks=[checkpointer],
                              steps_per_epoch=100, epochs=epochs, verbose=1)
scores = model.evaluate_generator(generator=test_generator,workers=4,use_multiprocessing=True,verbose=0)#验证

print('%s: %.2f' % (model.metrics_names[0], scores[0])) # Loss
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100)) # metrics1


# 绘制模型准确率学习曲线
plt.plot(history.history['accuracy'], label='train')
plt.legend()
plt.show()
