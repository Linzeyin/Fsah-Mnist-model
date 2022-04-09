# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import Sequential
import os
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from pltshow import plot_image,plot_value_array

print(tf.version.VERSION)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = tf.expand_dims(train_images,(int)(-1))#增加数据维度
test_images = tf.expand_dims(test_images,(int)(-1))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
K.image_dim_ordering='th'

model = keras.Sequential([
    keras.layers.Conv2D(64, kernel_size=2, strides=(1,1),padding='same', activation='relu',input_shape=(28,28,1)),
    keras.layers.MaxPool2D(pool_size=(2, 2),strides=None,padding='same'),
    keras.layers.Dropout(0.3),

    keras.layers.Conv2D(32, kernel_size=3, strides=(1,1),padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2),strides=None,padding='same'),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])#构建模型
model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])#模型编译指令


model.fit(train_images, train_labels, epochs=10,batch_size=1000, validation_split=0.2, verbose=1)#模型训练指令
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

#checkpoint_path = "E:/file/tensorflow file/model.ckptt"
model.save('E:/file/tensorflow file/model.ckptt')
